from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import nn

from tsumugi_piano_cover.config import DiffusionConfig, SourceEncoderConfig, TsumugiConfig
from tsumugi_piano_cover.models.tsumugi_encoder import TsumugiAudioEncoder
from tsumugi_piano_cover.models.transformer import Transformer


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
        self, diffusion_config: DiffusionConfig, performer_vocab_size: int
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
        context_attention_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        ノイズが付加された潜在表現に対し、各種条件（時間、ノイズステップ、演奏者、原曲）を加味して
        Diffusion学習で使うターゲット（epsilon または velocity）を予測する順伝播処理。

        Args:
            noisy_latents: ノイズが付与されたセグメントの潜在表現 [batch_size, num_segments, latent_dim]
            segment_times: 各セグメントの曲全体における絶対時間（秒） [batch_size, num_segments]
            segment_mask: 有効なセグメントを示すマスク [batch_size, num_segments]
            timesteps: 現在の拡散モデルのステップ数（ノイズレベル） [batch_size]
            performer_ids: 演奏スタイルの条件付けに用いる演奏者ID [batch_size]
            memory: クロスアテンションで参照する原曲（ソース）の特徴量 [batch_size, memory_length, d_model]
            memory_mask: memoryの有効な要素を示すマスク [batch_size, memory_length]
            context_attention_bias: target segment から source memory への soft attention bias
                [batch_size, num_segments, memory_length]

        Returns:
            予測されたDiffusionターゲット [batch_size, num_segments, latent_dim]
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
            context_attention_bias=context_attention_bias,
        )
        return self.output_projection(hidden)


class ConditionalSegmentDiffusionModel(nn.Module):
    def __init__(
        self,
        source_config: SourceEncoderConfig,
        diffusion_config: DiffusionConfig,
        performer_vocab_size: int,
        tsumugi_config: TsumugiConfig,
    ) -> None:
        super().__init__()
        if source_config.d_model != diffusion_config.d_model:
            raise ValueError(
                f"source d_model ({source_config.d_model}) must match diffusion d_model ({diffusion_config.d_model})"
            )
        self.source_config = source_config
        self.diffusion_config = diffusion_config
        self.audio_encoder = TsumugiAudioEncoder(tsumugi_config, output_dim=source_config.d_model)
        self.denoiser = SegmentLatentDenoiser(diffusion_config, performer_vocab_size)

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

    def encode_source(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 原曲エンコードとメモリ生成
        source_embeddings, source_mask, source_times = self.audio_encoder(
            audio=batch["source_audio"],
            audio_lengths=batch["source_audio_lengths"],
        )
        return source_embeddings, source_mask, source_times

    def _build_alignment_attention_bias(
        self,
        batch: dict[str, torch.Tensor],
        source_times: torch.Tensor,
    ) -> torch.Tensor | None:
        # alignment は教師ターゲットをワープせず、cross-attention の事前分布としてだけ使う
        strength = float(self.source_config.alignment_bias_strength)
        if strength <= 0.0 or "alignment_source_times" not in batch:
            return None

        segment_times = batch["segment_times"].to(device=source_times.device, dtype=source_times.dtype)
        aligned_source_times = batch["alignment_source_times"].to(device=source_times.device, dtype=source_times.dtype)
        aligned_weight = float(self.source_config.alignment_bias_aligned_time_weight)
        guide_times = (1.0 - aligned_weight) * segment_times + aligned_weight * aligned_source_times

        sigma_seconds = float(self.source_config.alignment_bias_sigma_seconds)
        time_delta = source_times.unsqueeze(1) - guide_times.unsqueeze(-1)
        bias = -0.5 * (time_delta / sigma_seconds).square()

        if "alignment_mask" in batch:
            alignment_mask = batch["alignment_mask"].to(device=source_times.device, dtype=torch.bool)
            if self.training and self.source_config.alignment_bias_dropout > 0.0:
                keep_prob = 1.0 - float(self.source_config.alignment_bias_dropout)
                keep_mask = torch.rand(alignment_mask.shape, device=alignment_mask.device) < keep_prob
                alignment_mask = alignment_mask & keep_mask
            bias = torch.where(alignment_mask.unsqueeze(-1), bias, torch.zeros_like(bias))

        return bias.clamp_min(-20.0) * strength

    def _expand_noise_scales(self, timesteps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # [batch_size] の時刻インデックスを [batch_size, 1, 1] の係数へ展開
        sqrt_alpha_bar = rearrange(self.sqrt_alpha_bars[timesteps], "b -> b 1 1")
        sqrt_one_minus_alpha_bar = rearrange(self.sqrt_one_minus_alpha_bars[timesteps], "b -> b 1 1")
        return sqrt_alpha_bar, sqrt_one_minus_alpha_bar

    def q_sample(self, latents: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # フォワードパス：ノイズ付与 (q(x_t | x_0))
        # latents, noise: [batch_size, num_segments, latent_dim]
        # timesteps: [batch_size]

        # [batch_size] -> [batch_size, 1, 1]
        sqrt_alpha_bar, sqrt_one_minus_alpha_bar = self._expand_noise_scales(timesteps)
        # [batch_size, num_segments, latent_dim]
        return sqrt_alpha_bar * latents + sqrt_one_minus_alpha_bar * noise

    def compute_training_target(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        # 学習ターゲットを prediction_type に応じて組み立てる
        sqrt_alpha_bar, sqrt_one_minus_alpha_bar = self._expand_noise_scales(timesteps)

        # 1. epsilon-pred は加えたノイズそのものを当てる
        if self.diffusion_config.prediction_type == "epsilon":
            return noise

        # 2. v-pred は v = alpha_t * epsilon - sigma_t * x0 を当てる
        if self.diffusion_config.prediction_type == "v":
            return sqrt_alpha_bar * noise - sqrt_one_minus_alpha_bar * latents

        raise ValueError(f"unsupported prediction_type: {self.diffusion_config.prediction_type!r}")

    def predict_x0_and_noise(
        self,
        noisy_latents: torch.Tensor,
        model_output: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # モデル出力から x0 と epsilon を復元する
        sqrt_alpha_bar, sqrt_one_minus_alpha_bar = self._expand_noise_scales(timesteps)

        # 1. epsilon-pred は出力がそのまま epsilon
        if self.diffusion_config.prediction_type == "epsilon":
            pred_noise = model_output
            pred_x0 = (noisy_latents - sqrt_one_minus_alpha_bar * pred_noise) / sqrt_alpha_bar
            return pred_x0, pred_noise

        # 2. v-pred は出力を v とみなし、x0 と epsilon に戻す
        if self.diffusion_config.prediction_type == "v":
            pred_v = model_output
            pred_x0 = sqrt_alpha_bar * noisy_latents - sqrt_one_minus_alpha_bar * pred_v
            pred_noise = sqrt_one_minus_alpha_bar * noisy_latents + sqrt_alpha_bar * pred_v
            return pred_x0, pred_noise

        raise ValueError(f"unsupported prediction_type: {self.diffusion_config.prediction_type!r}")

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        memory, memory_mask, source_times = self.encode_source(batch)
        context_attention_bias = self._build_alignment_attention_bias(batch, source_times)
        return self.denoiser(
            noisy_latents=noisy_latents,
            segment_times=batch["segment_times"],
            segment_mask=batch["segment_mask"],
            timesteps=timesteps,
            performer_ids=batch["performer_ids"],
            memory=memory,
            memory_mask=memory_mask,
            context_attention_bias=context_attention_bias,
        )

    @torch.no_grad()
    def sample(
        self,
        batch: dict[str, torch.Tensor],
        latent_shape: tuple[int, int, int],
        sampling_steps: int | None = None,
    ) -> torch.Tensor:
        # === 1. 初期化と準備 ===
        device = batch["source_audio"].device
        num_steps = sampling_steps or self.diffusion_config.sampling_steps

        # 純粋なガウスノイズからスタート
        latents = torch.randn(latent_shape, device=device)

        # 原曲のエンコード（全ステップで使い回すコンテキスト）
        memory, memory_mask, source_times = self.encode_source(batch)
        context_attention_bias = self._build_alignment_attention_bias(batch, source_times)

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
            model_output = self.denoiser(
                noisy_latents=latents,
                segment_times=batch["segment_times"],
                segment_mask=batch["segment_mask"],
                timesteps=timestep_batch,
                performer_ids=batch["performer_ids"],
                memory=memory,
                memory_mask=memory_mask,
                context_attention_bias=context_attention_bias,
            )
            pred_x0, pred_noise = self.predict_x0_and_noise(latents, model_output, timestep_batch)

            # 2-2. ノイズを除去した元の状態 (x0) を推定
            # 最終ステップの場合は、推定した x0 をそのまま最終結果とする
            if step_pos == len(step_indices) - 1:
                latents = pred_x0
                break

            # 2-3. 次のステップのノイズレベルに合わせて状態 (x_t-1) を計算
            next_timestep = step_indices[step_pos + 1]
            alpha_bar_next = self.alpha_bars[next_timestep]
            latents = torch.sqrt(alpha_bar_next) * pred_x0 + torch.sqrt(1.0 - alpha_bar_next) * pred_noise

        return latents
