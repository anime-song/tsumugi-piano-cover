from __future__ import annotations

import torch

from tsumugi_piano_cover.config import ExperimentConfig


_SEQUENCE_KEYS = (
    "alignment_source_times",
    "alignment_mask",
    "target_segments",
    "segment_mask",
    "segment_times",
)


def crop_diffusion_batch(
    batch: dict[str, object],
    config: ExperimentConfig,
    start: int,
) -> dict[str, object]:
    """Crop a target window and mark its teacher-forced left overlap."""
    if not config.diffusion_training.short_sequence_enabled:
        return batch

    segment_mask = batch.get("segment_mask")
    segment_times = batch.get("segment_times")
    if (
        not isinstance(segment_mask, torch.Tensor)
        or segment_mask.ndim != 2
        or segment_mask.shape[0] != 1
        or not isinstance(segment_times, torch.Tensor)
    ):
        raise ValueError("short-sequence training currently requires batch_size=1 with segment times")

    valid_segments = int(segment_mask[0].sum().item())
    window_segments = min(int(config.diffusion_training.short_sequence_segments), valid_segments)
    if window_segments < 1:
        raise ValueError("short-sequence batch has no valid target segments")
    max_start = valid_segments - window_segments
    start = max(0, min(int(start), max_start))
    end = start + window_segments

    cropped = dict(batch)
    for key in _SEQUENCE_KEYS:
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            cropped[key] = value[:, start:end]

    overlap_segments = 0
    if config.diffusion_training.teacher_overlap_enabled and start > 0:
        overlap_segments = min(
            int(config.diffusion_training.teacher_overlap_segments),
            max(0, window_segments - 1),
        )
    teacher_overlap_mask = torch.zeros(
        (1, window_segments),
        dtype=torch.bool,
        device=segment_mask.device,
    )
    if overlap_segments > 0:
        teacher_overlap_mask[:, :overlap_segments] = True

    window_times = segment_times[0, start:end]
    frame_seconds = float(config.target_roll.frame_seconds)
    window_start_frame = max(0, int(round(float(window_times[0].item()) / frame_seconds)))
    window_end_time = float(window_times[-1].item()) + float(config.target_roll.phrase_seconds)
    window_end_frame = int(round(window_end_time / frame_seconds))
    target_num_frames = batch.get("target_num_frames")
    if isinstance(target_num_frames, torch.Tensor) and target_num_frames.numel() > 0:
        window_end_frame = min(window_end_frame, int(target_num_frames[0].item()))

    generated_start_index = overlap_segments
    generated_start_frame = max(
        window_start_frame,
        int(round(float(window_times[generated_start_index].item()) / frame_seconds)),
    )

    cropped["teacher_overlap_mask"] = teacher_overlap_mask
    cropped["short_sequence_start"] = start
    cropped["short_sequence_end"] = end
    cropped["window_frame_start"] = window_start_frame
    cropped["window_frame_end"] = max(window_start_frame + 1, window_end_frame)
    cropped["generated_frame_start"] = generated_start_frame
    cropped["generated_frame_end"] = max(generated_start_frame + 1, window_end_frame)
    return cropped


def random_short_sequence_batch(
    batch: dict[str, object],
    config: ExperimentConfig,
) -> dict[str, object]:
    if not config.diffusion_training.short_sequence_enabled:
        return batch

    segment_mask = batch.get("segment_mask")
    if not isinstance(segment_mask, torch.Tensor) or segment_mask.ndim != 2 or segment_mask.shape[0] != 1:
        raise ValueError("short-sequence training currently requires batch_size=1")
    valid_segments = int(segment_mask[0].sum().item())
    window_segments = min(int(config.diffusion_training.short_sequence_segments), valid_segments)
    max_start = valid_segments - window_segments
    start = int(torch.randint(max_start + 1, (1,), device=segment_mask.device).item()) if max_start > 0 else 0
    return crop_diffusion_batch(batch, config, start)


def deterministic_short_sequence_batch(
    batch: dict[str, object],
    config: ExperimentConfig,
) -> dict[str, object]:
    """Select a stable middle window for validation and generation diagnostics."""
    if not config.diffusion_training.short_sequence_enabled:
        return batch

    segment_mask = batch.get("segment_mask")
    if not isinstance(segment_mask, torch.Tensor) or segment_mask.ndim != 2 or segment_mask.shape[0] != 1:
        raise ValueError("short-sequence evaluation currently requires batch_size=1")
    valid_segments = int(segment_mask[0].sum().item())
    window_segments = min(int(config.diffusion_training.short_sequence_segments), valid_segments)
    max_start = valid_segments - window_segments
    start = max_start // 2 if max_start > 0 else 0
    return crop_diffusion_batch(batch, config, start)


def diffusion_loss_mask(batch: dict[str, object]) -> torch.Tensor:
    """Exclude teacher-provided overlap positions from the diffusion loss."""
    segment_mask = batch.get("segment_mask")
    if not isinstance(segment_mask, torch.Tensor):
        raise ValueError("batch lacks segment_mask")
    teacher_mask = batch.get("teacher_overlap_mask")
    if not isinstance(teacher_mask, torch.Tensor):
        return segment_mask
    return segment_mask & ~teacher_mask.to(device=segment_mask.device, dtype=torch.bool)


def apply_teacher_overlap_context(
    noisy_latents: torch.Tensor,
    clean_latents: torch.Tensor,
    batch: dict[str, object],
) -> torch.Tensor:
    """Replace the left overlap with clean teacher latents for window conditioning."""
    teacher_mask = batch.get("teacher_overlap_mask")
    if not isinstance(teacher_mask, torch.Tensor):
        return noisy_latents
    mask = teacher_mask.to(device=noisy_latents.device, dtype=torch.bool).unsqueeze(-1)
    return torch.where(mask, clean_latents, noisy_latents)
