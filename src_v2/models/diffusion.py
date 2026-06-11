from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import nn

from src_v2.config import DiffusionConfig, SourceEncoderConfig
from src_v2.models.source_encoder import SongEncoder, SourceNoteChunkEncoder
from src_v2.models.transformer import Transformer


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.projection = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        exponent = -math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=timesteps.device) * exponent)
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        if embedding.shape[1] < self.dim:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
        return self.projection(embedding)


class SegmentLatentDenoiser(nn.Module):
    def __init__(
        self, diffusion_config: DiffusionConfig, source_config: SourceEncoderConfig, performer_vocab_size: int
    ) -> None:
        super().__init__()
        self.diffusion_config = diffusion_config
        self.input_projection = nn.Linear(diffusion_config.latent_dim, diffusion_config.d_model)
        self.output_projection = nn.Linear(diffusion_config.d_model, diffusion_config.latent_dim)
        self.time_embedding = SinusoidalTimeEmbedding(diffusion_config.d_model)
        self.segment_time_projection = nn.Sequential(
            nn.Linear(1, diffusion_config.d_model),
            nn.SiLU(),
            nn.Linear(diffusion_config.d_model, diffusion_config.d_model),
        )
        self.performer_embedding = nn.Embedding(performer_vocab_size, diffusion_config.d_model)
        self.performer_dropout = nn.Dropout(diffusion_config.performer_dropout)
        self.transformer = Transformer(
            input_dim=diffusion_config.d_model,
            head_dim=diffusion_config.head_dim,
            num_heads=diffusion_config.num_heads,
            num_layers=diffusion_config.num_layers,
            ffn_hidden_size_factor=diffusion_config.ff_multiplier,
            dropout=diffusion_config.dropout,
            use_cross_attention=True,
            output_norm=True,
        )

    def forward(
        self,
        noisy_latents: torch.Tensor,
        segment_times: torch.Tensor,
        segment_mask: torch.Tensor,
        timesteps: torch.Tensor,
        performer_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        ノイズが付加された潜在表現に対し、各種条件（時間、ノイズステップ、演奏者、原曲）を加味して
        ノイズ（または元データ）を予測する順伝播処理。

        Args:
            noisy_latents: ノイズが付与されたセグメントの潜在表現 [batch_size, num_segments, latent_dim]
            segment_times: 各セグメントの曲全体における絶対時間（秒） [batch_size, num_segments]
            segment_mask: 有効なセグメントを示すマスク [batch_size, num_segments]
            timesteps: 現在の拡散モデルのステップ数（ノイズレベル） [batch_size]
            performer_ids: 演奏スタイルの条件付けに用いる演奏者ID [batch_size]
            memory: クロスアテンションで参照する原曲（ソース）の特徴量 [batch_size, memory_length, d_model]
            memory_mask: memoryの有効な要素を示すマスク [batch_size, memory_length]

        Returns:
            予測されたノイズ（または復元された潜在表現） [batch_size, num_segments, latent_dim]
        """
        hidden = self.input_projection(noisy_latents)
        # 条件の埋め込みを加算
        hidden = hidden + self.segment_time_projection(segment_times.unsqueeze(-1))
        hidden = hidden + self.time_embedding(timesteps).unsqueeze(1)
        hidden = hidden + self.performer_dropout(self.performer_embedding(performer_ids)).unsqueeze(1)

        hidden = self.transformer(
            hidden,
            context=memory,
            attention_mask=segment_mask,
            context_attention_mask=memory_mask,
        )
        return self.output_projection(hidden)


class ConditionalSegmentDiffusionModel(nn.Module):
    def __init__(
        self,
        source_config: SourceEncoderConfig,
        diffusion_config: DiffusionConfig,
        performer_vocab_size: int,
    ) -> None:
        super().__init__()
        if source_config.d_model != diffusion_config.d_model:
            raise ValueError(
                f"source d_model ({source_config.d_model}) must match diffusion d_model ({diffusion_config.d_model})"
            )
        self.source_config = source_config
        self.diffusion_config = diffusion_config
        self.chunk_encoder = SourceNoteChunkEncoder(source_config)
        self.song_encoder = SongEncoder(source_config)
        self.denoiser = SegmentLatentDenoiser(diffusion_config, source_config, performer_vocab_size)

        # ノイズスケジュールの事前計算
        betas = torch.linspace(
            diffusion_config.beta_start, diffusion_config.beta_end, diffusion_config.num_train_timesteps
        )
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        for module in self.modules():
            if isinstance(module, Transformer):
                module.set_gradient_checkpointing(enabled)

    def encode_source(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # 原曲エンコードとメモリ生成
        chunk_embeddings = self.chunk_encoder(
            source_features=batch["source_features"],
            source_programs=batch["source_programs"],
            source_drums=batch["source_drums"],
            source_track_roles=batch["source_track_roles"],
            source_note_mask=batch["source_note_mask"],
        )
        memory, _ = self.song_encoder(
            chunk_embeddings=chunk_embeddings,
            source_chunk_mask=batch["source_chunk_mask"],
            source_chunk_times=batch["source_chunk_times"],
        )
        return memory, batch["source_chunk_mask"]

    def q_sample(self, latents: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # フォワードパス：ノイズ付与 (q(x_t | x_0))
        # latents, noise: [batch_size, num_segments, latent_dim]
        # timesteps: [batch_size]

        # [batch_size] -> [batch_size, 1, 1]
        sqrt_alpha_bar = rearrange(self.sqrt_alpha_bars[timesteps], "b -> b 1 1")
        sqrt_one_minus_alpha_bar = rearrange(self.sqrt_one_minus_alpha_bars[timesteps], "b -> b 1 1")

        # [batch_size, num_segments, latent_dim]
        return sqrt_alpha_bar * latents + sqrt_one_minus_alpha_bar * noise

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        memory, memory_mask = self.encode_source(batch)
        return self.denoiser(
            noisy_latents=noisy_latents,
            segment_times=batch["segment_times"],
            segment_mask=batch["segment_mask"],
            timesteps=timesteps,
            performer_ids=batch["performer_ids"],
            memory=memory,
            memory_mask=memory_mask,
        )

    @torch.no_grad()
    def sample(
        self,
        batch: dict[str, torch.Tensor],
        latent_shape: tuple[int, int, int],
        sampling_steps: int | None = None,
    ) -> torch.Tensor:
        # === 1. 初期化と準備 ===
        device = batch["source_features"].device
        num_steps = sampling_steps or self.diffusion_config.sampling_steps

        # 純粋なガウスノイズからスタート
        latents = torch.randn(latent_shape, device=device)

        # 原曲のエンコード（全ステップで使い回すコンテキスト）
        memory, memory_mask = self.encode_source(batch)

        # サンプリング用のタイムステップを計算 (T-1 から 0 へ)
        step_indices = torch.linspace(
            self.diffusion_config.num_train_timesteps - 1,
            0,
            steps=num_steps,
            device=device,
        ).long()

        # === 2. 逆拡散ループ ===
        for step_pos, timestep in enumerate(step_indices):
            # 2-1. ノイズの予測
            timestep_batch = torch.full((latent_shape[0],), int(timestep.item()), device=device, dtype=torch.long)
            pred_noise = self.denoiser(
                noisy_latents=latents,
                segment_times=batch["segment_times"],
                segment_mask=batch["segment_mask"],
                timesteps=timestep_batch,
                performer_ids=batch["performer_ids"],
                memory=memory,
                memory_mask=memory_mask,
            )

            # 2-2. ノイズを除去した元の状態 (x0) を推定
            alpha_bar_t = self.alpha_bars[timestep]
            x0 = (latents - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)

            # 最終ステップの場合は、推定した x0 をそのまま最終結果とする
            if step_pos == len(step_indices) - 1:
                latents = x0
                break

            # 2-3. 次のステップのノイズレベルに合わせて状態 (x_t-1) を計算
            next_timestep = step_indices[step_pos + 1]
            alpha_bar_next = self.alpha_bars[next_timestep]
            latents = torch.sqrt(alpha_bar_next) * x0 + torch.sqrt(1.0 - alpha_bar_next) * pred_noise

        return latents
