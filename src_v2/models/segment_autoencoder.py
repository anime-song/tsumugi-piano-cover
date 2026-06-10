import torch
from einops import rearrange
from torch import nn

from src_v2.config import SegmentAutoencoderConfig, TargetRollConfig
from src_v2.models.transformer import Transformer


class SegmentEncoder(nn.Module):
    # ピアノロールのセグメントを潜在表現（mu, logvar）に圧縮するVAEエンコーダー
    def __init__(self, config: SegmentAutoencoderConfig, roll_config: TargetRollConfig) -> None:
        super().__init__()
        self.config = config
        self.roll_config = roll_config
        self.frame_projection = nn.Linear(roll_config.feature_dim, config.d_model)
        self.frame_positions = nn.Parameter(torch.zeros(1, roll_config.frames_per_phrase, config.d_model))
        # 潜在表現用のクエリトークン
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
        batch_size, num_segments, num_frames, _ = segments.shape
        # フレーム位置追加
        hidden = self.frame_projection(rearrange(segments, "b s f d -> (b s) f d"))
        hidden = hidden + self.frame_positions[:, :num_frames]
        
        # 情報抽出用のクエリを追加
        queries = self.query_tokens.expand(batch_size * num_segments, -1, -1)
        hidden = torch.cat([queries, hidden], dim=1)
        encoded = self.encoder(hidden)
        
        # クエリトークン部分を取り出してmuとlogvarに変換
        queried = rearrange(encoded[:, : self.config.num_latent_queries], "bs q d -> bs (q d)")
        mu = rearrange(self.to_mu(queried), "(b s) l -> b s l", b=batch_size)
        logvar = rearrange(self.to_logvar(queried), "(b s) l -> b s l", b=batch_size)
        return mu, logvar


class SegmentDecoder(nn.Module):
    # 潜在表現からクロスアテンションを介してピアノロールを再構成するVAEデコーダー
    def __init__(self, config: SegmentAutoencoderConfig, roll_config: TargetRollConfig) -> None:
        super().__init__()
        self.config = config
        self.roll_config = roll_config
        self.num_memory_tokens = config.num_latent_queries
        # 潜在空間からクロスアテンション用のメモリートークンへ射影
        self.latent_to_memory = nn.Sequential(
            nn.Linear(config.latent_dim, config.d_model * 2),
            nn.SiLU(),
            nn.Linear(config.d_model * 2, config.d_model * self.num_memory_tokens),
        )
        # デコード時の各フレーム用クエリベクトル
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
        self.onset_head = nn.Linear(config.d_model, roll_config.pitch_count)
        self.sustain_head = nn.Linear(config.d_model, roll_config.pitch_count)
        self.pedal_head = nn.Linear(config.d_model, 1)
        self.velocity_head = nn.Sequential(nn.Linear(config.d_model, roll_config.pitch_count), nn.Sigmoid())

    def forward(self, latents: torch.Tensor) -> dict[str, torch.Tensor]:
        # latents: [batch_size, num_segments, latent_dim]
        batch_size, num_segments, _ = latents.shape
        
        # 1. 潜在表現をTransformerのコンテキスト（メモリートークン）に変換
        # [batch_size * num_segments, num_memory_tokens, d_model]
        memory = rearrange(
            self.latent_to_memory(latents),
            "b s (m d) -> (b s) m d",
            m=self.num_memory_tokens,
        )
        
        # 2. 各フレームのデコード用クエリを用意
        # [batch_size * num_segments, frames_per_phrase, d_model]
        queries = self.frame_queries.expand(batch_size * num_segments, -1, -1)
        
        # 3. クロスアテンションによるデコード
        # フレームごとのクエリが、潜在表現のメモリートークンを参照して特徴を抽出する
        # [batch_size * num_segments, frames_per_phrase, d_model]
        decoded = self.decoder(
            queries,
            context=memory,
        )
        
        # 4. 各ヘッドで最終的な出力に変換し、元のバッチとセグメントの次元構造に戻す
        # [batch_size, num_segments, frames_per_phrase, pitch_count or 1]
        onset_logits = rearrange(self.onset_head(decoded), "(b s) f p -> b s f p", b=batch_size)
        sustain_logits = rearrange(self.sustain_head(decoded), "(b s) f p -> b s f p", b=batch_size)
        pedal_logits = rearrange(self.pedal_head(decoded), "(b s) f 1 -> b s f 1", b=batch_size)
        velocity = rearrange(self.velocity_head(decoded), "(b s) f p -> b s f p", b=batch_size)
        return {
            "onset_logits": onset_logits,
            "sustain_logits": sustain_logits,
            "pedal_logits": pedal_logits,
            "velocity": velocity,
        }


class SegmentLatentAutoencoder(nn.Module):
    # VAEの全体モデル
    def __init__(self, config: SegmentAutoencoderConfig, roll_config: TargetRollConfig) -> None:
        super().__init__()
        self.config = config
        self.roll_config = roll_config
        self.encoder = SegmentEncoder(config, roll_config)
        self.decoder = SegmentDecoder(config, roll_config)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # 再パラメータ化トリックによるランダムサンプリング
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def encode(
        self,
        segments: torch.Tensor,
        sample_posterior: bool = True,
        variational: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # セグメントを潜在表現にエンコード
        mu, logvar = self.encoder(segments)
        if variational:
            # variational時は再パラメータ化、それ以外はmuを潜在表現とする
            latents = self.reparameterize(mu, logvar) if sample_posterior else mu
        else:
            logvar = torch.zeros_like(mu)
            latents = mu
        return latents, mu, logvar

    def decode(self, latents: torch.Tensor) -> dict[str, torch.Tensor]:
        # 潜在表現からロジットへデコード
        return self.decoder(latents)

    def forward(
        self,
        segments: torch.Tensor,
        sample_posterior: bool = True,
        variational: bool = True,
    ) -> dict[str, torch.Tensor]:
        latents, mu, logvar = self.encode(
            segments,
            sample_posterior=sample_posterior,
            variational=variational)
        recon = self.decode(latents)
        recon["latents"] = latents
        recon["mu"] = mu
        recon["logvar"] = logvar
        recon["variational"] = variational
        return recon


@torch.no_grad()
def decoder_outputs_to_segment_rolls(
    outputs: dict[str, torch.Tensor],
    model_config: SegmentAutoencoderConfig,
    roll_config: TargetRollConfig,
) -> torch.Tensor:
    # ロジットにsigmoidを適用し、ピアノロール特徴量に戻す
    del model_config
    onset = torch.sigmoid(outputs["onset_logits"])
    sustain = torch.sigmoid(outputs["sustain_logits"])
    pedal = torch.sigmoid(outputs["pedal_logits"])
    velocity = outputs["velocity"].clamp(0.0, 1.0)
    return torch.cat([onset, sustain, velocity, pedal], dim=-1)
