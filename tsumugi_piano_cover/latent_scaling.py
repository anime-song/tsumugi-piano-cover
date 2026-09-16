from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LatentNormalizer:
    """VAE 潜在表現を Diffusion の N(0, I) 前提に合わせて正規化する。

    VAE の潜在は次元ごとに平均・分散が 0/1 からずれているため、スケールを割るだけでは
    サンプリング開始時の純ガウスノイズと分布が合わない。平均を引き、可能なら次元ごとの
    標準偏差で割って白色化する（``latent_std`` が無い場合はスカラー ``scale`` で代用）。
    """

    scale: float
    mean: torch.Tensor | None = None
    std: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            raise ValueError(f"latent scale must be positive, got {self.scale}")
        if self.std is not None and bool((self.std <= 0.0).any()):
            raise ValueError("latent std must be positive in every dimension")

    @property
    def whitened(self) -> bool:
        return self.std is not None

    def to(self, device: torch.device | str) -> LatentNormalizer:
        return LatentNormalizer(
            scale=self.scale,
            mean=None if self.mean is None else self.mean.to(device),
            std=None if self.std is None else self.std.to(device),
        )

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        if self.mean is not None:
            latents = latents - self.mean.to(device=latents.device, dtype=latents.dtype)
        if self.std is not None:
            return latents / self.std.to(device=latents.device, dtype=latents.dtype)
        return latents / self.scale

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        if self.std is not None:
            latents = latents * self.std.to(device=latents.device, dtype=latents.dtype)
        else:
            latents = latents * self.scale
        if self.mean is not None:
            latents = latents + self.mean.to(device=latents.device, dtype=latents.dtype)
        return latents

    def state_dict(self) -> dict[str, object]:
        return {
            "latent_scale": float(self.scale),
            "latent_mean": None if self.mean is None else self.mean.detach().cpu(),
            "latent_std": None if self.std is None else self.std.detach().cpu(),
        }

    def describe(self) -> str:
        if self.std is None:
            centered = "mean-centered" if self.mean is not None else "raw"
            return f"scalar(scale={self.scale:.4f}, {centered})"
        return f"per-dim whitening(scale~{self.scale:.4f}, mean-centered={self.mean is not None})"

    def matches(self, other: LatentNormalizer, tolerance: float = 1.0e-6) -> bool:
        if abs(self.scale - other.scale) > tolerance:
            return False
        for left, right in ((self.mean, other.mean), (self.std, other.std)):
            if (left is None) != (right is None):
                return False
            if left is not None and not torch.allclose(left.cpu(), right.cpu(), atol=tolerance, rtol=tolerance):
                return False
        return True

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, object]) -> LatentNormalizer:
        scale = payload.get("latent_scale")
        if scale is None:
            raise ValueError("latent normalization payload does not contain latent_scale")
        mean = payload.get("latent_mean")
        std = payload.get("latent_std")
        return cls(
            scale=float(scale),  # type: ignore[arg-type]
            mean=None if mean is None else torch.as_tensor(mean).float(),
            std=None if std is None else torch.as_tensor(std).float(),
        )

    @classmethod
    def from_autoencoder_checkpoint(
        cls,
        checkpoint: Mapping[str, object] | None,
        scale_override: float | None = None,
        center: bool = True,
        whiten: bool = True,
    ) -> LatentNormalizer:
        """VAE チェックポイントに保存された潜在統計から正規化器を作る。

        ``compute_latent_scaling.py --write-to-checkpoint`` が latent_scale / latent_mean /
        latent_std を書き込んでいる前提。``scale_override`` を渡した場合は
        スカラースケールのみを使う（白色化は無効）。
        """
        if scale_override is not None:
            return cls(scale=float(scale_override))

        if checkpoint is None:
            raise ValueError(
                "latent statistics are required. Run recipes/compute_latent_scaling.py "
                "with --write-to-checkpoint, or pass --latent-scale."
            )

        scale = checkpoint.get("latent_scale")
        if scale is None:
            raise ValueError(
                "autoencoder checkpoint does not contain latent_scale. "
                "Run recipes/compute_latent_scaling.py with --write-to-checkpoint, or pass --latent-scale."
            )

        mean_value = checkpoint.get("latent_mean") if center else None
        std_value = checkpoint.get("latent_std") if whiten else None
        return cls(
            scale=float(scale),  # type: ignore[arg-type]
            mean=None if mean_value is None else torch.as_tensor(mean_value).float(),
            std=None if std_value is None else torch.as_tensor(std_value).float(),
        )
