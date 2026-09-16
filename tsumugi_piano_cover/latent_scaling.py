from __future__ import annotations

from collections.abc import Mapping

import torch


def get_checkpoint_latent_scale(checkpoint: Mapping[str, object] | None) -> float | None:
    if checkpoint is None:
        return None
    value = checkpoint.get("latent_scale")
    if value is None:
        return None
    return float(value)


def resolve_latent_scale(
    override: float | None,
    checkpoint: Mapping[str, object] | None = None,
    fallback: float | None = None,
    require: bool = False,
    source_name: str = "checkpoint",
) -> float | None:
    if override is not None:
        return float(override)

    checkpoint_scale = get_checkpoint_latent_scale(checkpoint)
    if checkpoint_scale is not None:
        return checkpoint_scale

    if fallback is not None:
        return float(fallback)

    if require:
        raise ValueError(
            "latent_scale is required but was not provided. "
            f"Pass --latent-scale or store latent_scale in the {source_name}."
        )
    return None


def scale_latents_for_diffusion(latents: torch.Tensor, latent_scale: float) -> torch.Tensor:
    if latent_scale <= 0.0:
        raise ValueError(f"latent_scale must be positive, got {latent_scale}")
    return latents / float(latent_scale)


def unscale_latents_from_diffusion(latents: torch.Tensor, latent_scale: float) -> torch.Tensor:
    if latent_scale <= 0.0:
        raise ValueError(f"latent_scale must be positive, got {latent_scale}")
    return latents * float(latent_scale)
