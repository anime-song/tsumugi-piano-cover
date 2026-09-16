from __future__ import annotations

import torch
from torch import nn

from tsumugi_piano_cover.config import TsumugiConfig


class TsumugiAudioEncoder(nn.Module):
    """Tsumugi-MRL frame encoder with a task-specific output projection."""

    def __init__(self, config: TsumugiConfig, output_dim: int) -> None:
        super().__init__()
        try:
            from tsumugi_mrl import LoRAConfig, TsumugiMRLModel, apply_lora
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Tsumugi-MRL is required for the audio source encoder. "
                "Install the repository dependency from requirements.txt."
            ) from exc

        load_kwargs = {"revision": config.revision} if config.revision else {}
        self.model = TsumugiMRLModel.from_pretrained(config.model_id, **load_kwargs)
        self.model.audio_encoder.use_gradient_checkpoint = config.gradient_checkpointing

        if config.lora_enabled:
            lora_config = LoRAConfig(
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                target_modules=tuple(config.lora_target_modules),
                layers=tuple(config.lora_layers) if config.lora_layers is not None else None,
                freeze_base=True,
            )
            apply_lora(self.model, lora_config)
        else:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

        hidden_dim = int(self.model.config.d_model)
        if hidden_dim != config.hidden_dim:
            raise ValueError(f"Tsumugi hidden dimension does not match config: {hidden_dim} != {config.hidden_dim}")

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    @property
    def sample_rate(self) -> int:
        return int(self.model.config.sample_rate)

    @property
    def hop_length(self) -> int:
        return int(self.model.config.hop_length)

    @property
    def temporal_fold(self) -> int:
        return int(self.model.config.temporal_fold)

    def frame_metadata(
        self,
        audio_lengths: torch.Tensor,
        max_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return project-style valid mask and seconds for Tsumugi output tokens."""
        if audio_lengths.ndim != 1:
            raise ValueError(f"audio_lengths must have shape [B], got {tuple(audio_lengths.shape)}")
        if max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {max_samples}")

        n_fft = int(self.model.config.n_fft)
        hop_length = self.hop_length
        temporal_fold = self.temporal_fold
        if max_samples < n_fft:
            raise ValueError(f"audio must contain at least {n_fft} samples, got {max_samples}")

        max_mel_frames = (max_samples - n_fft) // hop_length + 1
        max_tokens = max_mel_frames // temporal_fold
        if max_tokens <= 0:
            raise ValueError("audio is too short to form one Tsumugi frame")

        lengths = audio_lengths.to(dtype=torch.long)
        mel_frames = torch.where(
            lengths >= n_fft,
            (lengths - n_fft) // hop_length + 1,
            torch.zeros_like(lengths),
        )
        valid_tokens = (mel_frames // temporal_fold).clamp(max=max_tokens)
        token_indices = torch.arange(max_tokens, device=audio_lengths.device)
        frame_mask = token_indices.unsqueeze(0) < valid_tokens.unsqueeze(1)
        frame_times = token_indices.to(dtype=torch.float32) * (hop_length * temporal_fold / self.sample_rate)
        frame_times = frame_times.unsqueeze(0).expand(audio_lengths.shape[0], -1)
        return frame_mask, frame_times

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frame_mask, frame_times = self.frame_metadata(audio_lengths, max_samples=audio.shape[-1])
        # パディングが無いときに mask を渡すと SDPA がマスクを実体化する経路に入るため、
        # 必要なときだけ渡す（batch_size=1 の学習では常にパディング無し）。実測 10ms/曲 短縮。
        # なお Windows 版 PyTorch は FlashAttention を同梱しておらず（2.7 / 2.13 とも）、
        # 実際に選ばれるのは cuDNN attention か mem-efficient attention。
        tsumugi_padding_mask = None if bool(frame_mask.all()) else ~frame_mask
        hidden = self.model(audio, padding_mask=tsumugi_padding_mask)
        if hidden.shape[:2] != frame_mask.shape:
            raise RuntimeError(
                "Tsumugi output length does not match calculated frame metadata: "
                f"{tuple(hidden.shape[:2])} != {tuple(frame_mask.shape)}"
            )
        hidden = self.output_projection(hidden)
        return hidden, frame_mask, frame_times
