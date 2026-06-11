from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from src_v2.config import SegmentAutoencoderConfig, TargetRollConfig
from src_v2.models.segment_semi_crf import (
    IntervalBoundaryPredictor,
    IntervalScorer,
    reconstruct_semi_crf_segments_to_roll,
)
from src_v2.models.transformer import Transformer


class SegmentEncoder(nn.Module):
    # ピアノロールのセグメントを潜在分布（mu, logvar）へ圧縮するVAEエンコーダー
    def __init__(self, config: SegmentAutoencoderConfig, roll_config: TargetRollConfig) -> None:
        super().__init__()
        self.config = config
        self.roll_config = roll_config
        self.frame_projection = nn.Linear(roll_config.feature_dim, config.d_model)
        self.frame_positions = nn.Parameter(torch.zeros(1, roll_config.frames_per_phrase, config.d_model))
        # セグメント全体の情報を集約するためのクエリトークン
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_latent_queries, config.d_model))
        self.encoder = Transformer(
            input_dim=config.d_model,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            num_layers=config.encoder_layers,
            ffn_hidden_size_factor=config.ff_multiplier,
            dropout=config.dropout,
            output_norm=True,
        )
        bottleneck_dim = config.d_model * config.num_latent_queries
        self.to_mu = nn.Linear(bottleneck_dim, config.latent_dim)
        self.to_logvar = nn.Linear(bottleneck_dim, config.latent_dim)
        nn.init.zeros_(self.to_logvar.weight)
        nn.init.zeros_(self.to_logvar.bias)

    def forward(self, segments: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # segments: [batch_size, num_segments, num_frames, feature_dim]
        batch_size, num_segments, num_frames, _ = segments.shape
        # 1. 各フレーム特徴を d_model 次元へ射影し、時間位置の情報を加える
        # [batch_size * num_segments, num_frames, d_model]
        hidden = self.frame_projection(rearrange(segments, "b s f d -> (b s) f d"))
        hidden = hidden + self.frame_positions[:, :num_frames]

        # 2. 先頭に query token を差し込み、Transformer にセグメント全体の情報を集約させる
        # [batch_size * num_segments, num_latent_queries + num_frames, d_model]
        queries = self.query_tokens.expand(batch_size * num_segments, -1, -1)
        hidden = torch.cat([queries, hidden], dim=1)
        encoded = self.encoder(hidden)

        # 3. query 部分だけを取り出し、潜在分布のパラメータ mu / logvar に変換する
        # mu, logvar: [batch_size, num_segments, latent_dim]
        queried = rearrange(encoded[:, : self.config.num_latent_queries], "bs q d -> bs (q d)")
        mu = rearrange(self.to_mu(queried), "(b s) l -> b s l", b=batch_size)
        logvar = rearrange(self.to_logvar(queried), "(b s) l -> b s l", b=batch_size)
        return mu, logvar


class SegmentDecoder(nn.Module):
    # 潜在表現から semi-CRF デコード用の区間特徴を復元するVAEデコーダー
    def __init__(self, config: SegmentAutoencoderConfig, roll_config: TargetRollConfig) -> None:
        super().__init__()
        self.config = config
        self.roll_config = roll_config
        self.num_memory_tokens = config.num_latent_queries
        # 潜在ベクトルを cross-attention が読むメモリートークン列に変換する
        self.latent_to_memory = nn.Sequential(
            nn.Linear(config.latent_dim, config.d_model * 2),
            nn.SiLU(),
            nn.Linear(config.d_model * 2, config.d_model * self.num_memory_tokens),
        )
        # 各フレーム位置に対応するデコード用クエリ
        self.frame_queries = nn.Parameter(torch.zeros(1, roll_config.frames_per_phrase, config.d_model))
        self.decoder = Transformer(
            input_dim=config.d_model,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            num_layers=config.decoder_layers,
            ffn_hidden_size_factor=config.ff_multiplier,
            dropout=config.dropout,
            use_cross_attention=True,
            output_norm=True,
        )

        if config.decoder_mode == "frame":
            self.frame_head = nn.Linear(config.d_model, roll_config.feature_dim)
        else:
            self.pedal_head = nn.Linear(config.d_model, 1)
            # 各ピッチ・各フレームごとの semi-CRF 特徴量を出力する
            self.pitch_feature_head = nn.Linear(
                config.d_model,
                roll_config.pitch_count * config.semi_crf_pitch_feature_dim,
            )
            self.interval_scorer = IntervalScorer(
                input_dim=config.semi_crf_pitch_feature_dim,
                head_dim=config.semi_crf_head_dim,
            )
            self.interval_boundary_predictor = (
                IntervalBoundaryPredictor(
                    input_dim=config.semi_crf_pitch_feature_dim,
                    dropout=config.dropout,
                )
                if config.use_interval_boundary_head
                else None
            )
            # 区間に対応する velocity をフレーム特徴から予測する
            self.velocity_head = nn.Sequential(
                nn.Linear(config.semi_crf_pitch_feature_dim, 1),
                nn.Sigmoid(),
            )

    def forward(self, latents: torch.Tensor) -> dict[str, torch.Tensor]:
        # latents: [batch_size, num_segments, latent_dim]
        batch_size, num_segments, _ = latents.shape
        # 1. 潜在表現を cross-attention 用の memory token 列へ変換する
        # [batch_size * num_segments, num_memory_tokens, d_model]
        memory = rearrange(
            self.latent_to_memory(latents),
            "b s (m d) -> (b s) m d",
            m=self.num_memory_tokens,
        )

        # 2. デコード先の各フレーム位置に対応する query を用意する
        # [batch_size * num_segments, frames_per_phrase, d_model]
        queries = self.frame_queries.expand(batch_size * num_segments, -1, -1)

        # 3. 各フレーム query が latent 由来の memory を参照して特徴を取り出す
        # [batch_size * num_segments, frames_per_phrase, d_model]
        decoded = self.decoder(queries, context=memory)

        if self.config.decoder_mode == "frame":
            # Frame単位で直接出力を予測する
            frame_logits = rearrange(self.frame_head(decoded), "(b s) f d -> b s f d", b=batch_size)
            return {"frame_logits": frame_logits}

        # 4. 各ヘッドで semi-CRF に必要な出力へ変換する
        # pedal_logits: [batch_size, num_segments, frames_per_phrase, 1]
        pedal_logits = rearrange(self.pedal_head(decoded), "(b s) f 1 -> b s f 1", b=batch_size)

        # pitch_features: [batch_size, num_segments, frames_per_phrase, pitch_count, semi_crf_pitch_feature_dim]
        pitch_features = rearrange(
            self.pitch_feature_head(decoded),
            "(b s) f (p d) -> b s f p d",
            b=batch_size,
            p=self.roll_config.pitch_count,
            d=self.config.semi_crf_pitch_feature_dim,
        )

        # interval_query / interval_key / interval_diag は semi-CRF の区間スコア計算に使う
        interval_query, interval_key, interval_diag = self.interval_scorer(pitch_features)

        # velocity: [batch_size, num_segments, frames_per_phrase, pitch_count]
        velocity = rearrange(self.velocity_head(pitch_features), "b s f p 1 -> b s f p")
        return {
            "pitch_features": pitch_features,
            "interval_query": interval_query,
            "interval_key": interval_key,
            "interval_diag": interval_diag,
            "pedal_logits": pedal_logits,
            "velocity": velocity,
        }


class SegmentLatentAutoencoder(nn.Module):
    # セグメント単位で動くVAEの全体モデル
    def __init__(self, config: SegmentAutoencoderConfig, roll_config: TargetRollConfig) -> None:
        super().__init__()
        self.config = config
        self.roll_config = roll_config
        self.encoder = SegmentEncoder(config, roll_config)
        self.decoder = SegmentDecoder(config, roll_config)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # 再パラメータ化トリックで潜在変数をサンプリングする
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def encode(
        self,
        segments: torch.Tensor,
        sample_posterior: bool = True,
        variational: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # セグメント列を潜在表現へエンコードする
        mu, logvar = self.encoder(segments)
        if variational:
            # VAE 学習時はサンプリングし、決定論的に使う場合は平均をそのまま使う
            latents = self.reparameterize(mu, logvar) if sample_posterior else mu
        else:
            logvar = torch.zeros_like(mu)
            latents = mu
        return latents, mu, logvar

    def decode(self, latents: torch.Tensor) -> dict[str, torch.Tensor]:
        # 潜在表現を semi-CRF 用の出力へデコードする
        return self.decoder(latents)

    @torch.no_grad()
    def reconstruct_roll(
        self,
        outputs: dict[str, torch.Tensor],
        segment_times: torch.Tensor,
        segment_valid_lengths: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        if self.config.decoder_mode == "frame":
            raise NotImplementedError("Frame mode decoding for reconstruction is not implemented.")

        # 推論時は 1 曲ぶんの出力を区間デコードしてロールへ戻す
        if int(outputs["pitch_features"].shape[0]) != 1:
            raise ValueError("reconstruct_roll expects outputs for a single song batch")

        segment_outputs = {
            "pitch_features": outputs["pitch_features"][0].detach(),
            "interval_query": outputs["interval_query"][0].detach(),
            "interval_key": outputs["interval_key"][0].detach(),
            "interval_diag": outputs["interval_diag"][0].detach(),
            "pedal_logits": outputs["pedal_logits"][0].detach(),
            "velocity": outputs["velocity"][0].detach(),
        }
        return reconstruct_semi_crf_segments_to_roll(
            segment_outputs,
            self.config,
            self.roll_config,
            segment_times.cpu(),
            segment_valid_lengths.cpu(),
            num_frames,
            self.decoder.interval_boundary_predictor,
        )

    def forward(
        self,
        segments: torch.Tensor,
        sample_posterior: bool = True,
        variational: bool = True,
    ) -> dict[str, torch.Tensor]:
        latents, mu, logvar = self.encode(
            segments,
            sample_posterior=sample_posterior,
            variational=variational,
        )
        recon = self.decode(latents)
        recon["latents"] = latents
        recon["mu"] = mu
        recon["logvar"] = logvar
        recon["variational"] = variational
        return recon
