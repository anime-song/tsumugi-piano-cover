from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import nn

from tsumugi_piano_cover.config import DiffusionConfig, SourceEncoderConfig, TsumugiConfig
from tsumugi_piano_cover.models.transformer import Transformer
from tsumugi_piano_cover.models.tsumugi_encoder import TsumugiAudioEncoder


def enforce_zero_terminal_snr(alpha_bars: torch.Tensor) -> torch.Tensor:
    """終端の alpha_bar が 0（= SNR 0）になるよう schedule を線形リスケールする。

    linear schedule のままだと sqrt(alpha_bar[-1]) が 0 にならず、学習時の x_T には
    わずかに元データが残る。一方サンプリングは純ガウスノイズから始まるため、
    特に潜在表現の DC 成分に学習/推論のズレが出る。
    Lin et al., "Common Diffusion Noise Schedules and Sample Steps are Flawed" の手順。
    """
    sqrt_alpha_bars = alpha_bars.sqrt()
    first = sqrt_alpha_bars[0].clone()
    last = sqrt_alpha_bars[-1].clone()
    sqrt_alpha_bars = (sqrt_alpha_bars - last) * (first / (first - last))
    return sqrt_alpha_bars.square()


def sinusoidal_features(
    values: torch.Tensor,
    dim: int,
    min_period: float,
    max_period: float,
) -> torch.Tensor:
    """任意形状のスカラー値を [-1, 1] に収まる正弦波特徴へ展開する。

    生の値をそのまま Linear に通すと入力のスケール（例: 秒単位の絶対時刻）が
    そのまま hidden のスケールになってしまうため、条件付けは必ずここを経由する。

    Args:
        values: 埋め込み対象の値 [...]
        dim: 出力次元
        min_period: 最も細かい周期（values と同じ単位）
        max_period: 最も粗い周期（values と同じ単位）

    Returns:
        正弦波特徴 [..., dim]
    """
    half_dim = dim // 2
    exponent = torch.arange(half_dim, device=values.device, dtype=torch.float32) / max(half_dim - 1, 1)
    periods = min_period * (max_period / min_period) ** exponent
    angles = values.float().unsqueeze(-1) * (2.0 * math.pi / periods)
    features = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if features.shape[-1] < dim:
        features = torch.cat([features, torch.zeros_like(features[..., :1])], dim=-1)
    return features


class ScalarConditionEmbedding(nn.Module):
    """スカラー条件（拡散ステップ / セグメント絶対時刻）を d_model の埋め込みへ変換する。

    末尾の LayerNorm で出力スケールを 1 に固定し、潜在表現や演奏者埋め込みと
    同程度の大きさで hidden に加算されるようにしている。
    """

    def __init__(self, dim: int, min_period: float, max_period: float) -> None:
        super().__init__()
        self.dim = dim
        self.min_period = float(min_period)
        self.max_period = float(max_period)
        self.projection = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = sinusoidal_features(values, self.dim, self.min_period, self.max_period)
        return self.projection(features)


class SegmentLatentDenoiser(nn.Module):
    def __init__(self, diffusion_config: DiffusionConfig, performer_vocab_size: int) -> None:
        super().__init__()
        self.diffusion_config = diffusion_config
        self.performer_vocab_size = int(performer_vocab_size)
        # 語彙の末尾を「演奏者を指定しない」ための null ID として予約する
        self.null_performer_id = int(performer_vocab_size)

        self.input_projection = nn.Linear(diffusion_config.latent_dim, diffusion_config.d_model)
        self.output_projection = nn.Linear(diffusion_config.d_model, diffusion_config.latent_dim)

        # 拡散ステップ（0..num_train_timesteps）の埋め込み
        self.time_embedding = ScalarConditionEmbedding(
            diffusion_config.d_model,
            min_period=2.0,
            max_period=2.0 * diffusion_config.num_train_timesteps,
        )
        # セグメントの絶対時刻（秒）の埋め込み。拍レベルから曲構造レベルまでを覆う
        self.segment_time_embedding = ScalarConditionEmbedding(
            diffusion_config.d_model,
            min_period=diffusion_config.segment_time_min_period_seconds,
            max_period=diffusion_config.segment_time_max_period_seconds,
        )

        self.performer_embedding = nn.Embedding(performer_vocab_size + 1, diffusion_config.d_model)
        self.performer_norm = nn.LayerNorm(diffusion_config.d_model)
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

    def _apply_performer_dropout(self, performer_ids: torch.Tensor) -> torch.Tensor:
        # 学習時に一定確率で null ID へ置き換え、演奏者条件なしでも動くようにする
        # （要素ごとの Dropout と違い無条件埋め込みが明示的に学習されるので CFG にも使える）
        probability = float(self.diffusion_config.performer_dropout)
        if not self.training or probability <= 0.0:
            return performer_ids
        drop = torch.rand(performer_ids.shape, device=performer_ids.device) < probability
        return torch.where(drop, torch.full_like(performer_ids, self.null_performer_id), performer_ids)

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
        # 条件の埋め込みを加算（いずれも LayerNorm 済みで std≈1 に揃えてある）
        hidden = hidden + self.segment_time_embedding(segment_times)
        hidden = hidden + self.time_embedding(timesteps).unsqueeze(1)
        performer_ids = self._apply_performer_dropout(performer_ids)
        hidden = hidden + self.performer_norm(self.performer_embedding(performer_ids)).unsqueeze(1)

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
        if diffusion_config.zero_terminal_snr:
            alpha_bars = enforce_zero_terminal_snr(alpha_bars)
            previous_alpha_bars = torch.cat([alpha_bars.new_ones(1), alpha_bars[:-1]])
            alphas = alpha_bars / previous_alpha_bars
            betas = 1.0 - alphas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

    @property
    def null_performer_id(self) -> int:
        return self.denoiser.null_performer_id

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

        use_identity: torch.Tensor | None = None
        if self.training and self.source_config.alignment_bias_identity_probability > 0.0:
            # 推論時は alignment が無く guide=segment_times になる。その条件を学習中にも
            # 一定確率で与えて、bias がズレていても壊れないようにする
            identity_probability = float(self.source_config.alignment_bias_identity_probability)
            use_identity = torch.rand((segment_times.shape[0], 1), device=segment_times.device) < identity_probability
            guide_times = torch.where(use_identity, segment_times, guide_times)

        sigma_seconds = float(self.source_config.alignment_bias_sigma_seconds)
        time_delta = source_times.unsqueeze(1) - guide_times.unsqueeze(-1)
        bias = -0.5 * (time_delta / sigma_seconds).square()

        active = batch.get("alignment_mask")
        if active is None:
            active = torch.ones(segment_times.shape, dtype=torch.bool, device=segment_times.device)
        else:
            active = active.to(device=segment_times.device, dtype=torch.bool)

        if self.training:
            if use_identity is not None:
                # identity guide は推論時と同じく常に有効な参照とみなす
                active = active | use_identity
            segment_dropout = float(self.source_config.alignment_bias_dropout)
            if segment_dropout > 0.0:
                keep_segment = torch.rand(active.shape, device=active.device) >= segment_dropout
                active = active & keep_segment
            song_dropout = float(self.source_config.alignment_bias_song_dropout)
            if song_dropout > 0.0:
                # 曲単位で bias を丸ごと落とし、bias 無しでも成立する状態を保つ
                keep_song = torch.rand((active.shape[0], 1), device=active.device) >= song_dropout
                active = active & keep_song

        bias = torch.where(active.unsqueeze(-1), bias, torch.zeros_like(bias))
        return bias.clamp_min(-20.0) * strength

    def prepare_conditioning(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | None]:
        """原曲エンコードと alignment bias を1回だけ計算し、使い回せる形で返す。

        音声エンコーダが計算時間の大半を占めるため、同じ曲に対して複数のタイムステップを
        流す学習ループやサンプリングループではこの結果を共有する。
        """
        memory, memory_mask, source_times = self.encode_source(batch)
        context_attention_bias = self._build_alignment_attention_bias(batch, source_times)
        return {
            "memory": memory,
            "memory_mask": memory_mask,
            "context_attention_bias": context_attention_bias,
        }

    def denoise(
        self,
        batch: dict[str, torch.Tensor],
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning: dict[str, torch.Tensor | None],
        performer_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.denoiser(
            noisy_latents=noisy_latents,
            segment_times=batch["segment_times"],
            segment_mask=batch["segment_mask"],
            timesteps=timesteps,
            performer_ids=batch["performer_ids"] if performer_ids is None else performer_ids,
            memory=conditioning["memory"],
            memory_mask=conditioning["memory_mask"],
            context_attention_bias=conditioning["context_attention_bias"],
        )

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
            # zero terminal SNR では sqrt_alpha_bar[-1] == 0 になるため割り算を保護する
            pred_x0 = (noisy_latents - sqrt_one_minus_alpha_bar * pred_noise) / sqrt_alpha_bar.clamp_min(1.0e-8)
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
        conditioning: dict[str, torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        if conditioning is None:
            conditioning = self.prepare_conditioning(batch)
        return self.denoise(batch, noisy_latents, timesteps, conditioning)

    @torch.no_grad()
    def sample(
        self,
        batch: dict[str, torch.Tensor],
        latent_shape: tuple[int, int, int],
        sampling_steps: int | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        # === 1. 初期化と準備 ===
        device = batch["source_audio"].device
        num_steps = sampling_steps or self.diffusion_config.sampling_steps

        # 純粋なガウスノイズからスタート
        latents = torch.randn(latent_shape, device=device)

        # 原曲のエンコード（全ステップで使い回すコンテキスト）
        conditioning = self.prepare_conditioning(batch)

        # 演奏者条件の classifier-free guidance 用に null ID を用意
        use_guidance = guidance_scale != 1.0
        null_performer_ids = torch.full_like(batch["performer_ids"], self.denoiser.null_performer_id)

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
            model_output = self.denoise(batch, latents, timestep_batch, conditioning)
            if use_guidance:
                # 無条件予測との差分を増幅して演奏者条件を強調する
                null_output = self.denoise(
                    batch, latents, timestep_batch, conditioning, performer_ids=null_performer_ids
                )
                model_output = null_output + guidance_scale * (model_output - null_output)
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
