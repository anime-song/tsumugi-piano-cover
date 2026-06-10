from __future__ import annotations

import torch
import torch.nn.functional as F

from src_v2.config import SegmentAutoencoderConfig, TargetRollConfig


def _balanced_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    # 正負の不均衡を補正する Balanced BCE
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have the same shape")
    positive = targets >= 0.5
    negative = ~positive

    if positive.any():
        positive_loss = F.binary_cross_entropy_with_logits(logits[positive], targets[positive], reduction="mean")
    else:
        positive_loss = logits.new_tensor(0.0)
    if negative.any():
        negative_loss = F.binary_cross_entropy_with_logits(logits[negative], targets[negative], reduction="mean")
    else:
        negative_loss = logits.new_tensor(0.0)
    return 0.5 * (positive_loss + negative_loss)


def _weighted_binary_cross_entropy_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    positive_weight: float,
    negative_weight: float,
) -> torch.Tensor:
    # クラス重み付きの Weighted BCE
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have the same shape")
    positive = targets >= 0.5
    negative = ~positive

    if positive.any():
        positive_loss = F.binary_cross_entropy_with_logits(logits[positive], targets[positive], reduction="mean")
    else:
        positive_loss = logits.new_tensor(0.0)
    if negative.any():
        negative_loss = F.binary_cross_entropy_with_logits(logits[negative], targets[negative], reduction="mean")
    else:
        negative_loss = logits.new_tensor(0.0)
    return float(positive_weight) * positive_loss + float(negative_weight) * negative_loss


def autoencoder_reconstruction_loss(
    outputs: dict[str, torch.Tensor],
    target_segments: torch.Tensor,
    segment_mask: torch.Tensor,
    model_config: SegmentAutoencoderConfig,
    roll_config: TargetRollConfig,
    variational: bool,
) -> dict[str, torch.Tensor]:
    # 再構成損失の計算（onset, sustain, pedal, velocity, KL）
    batch_size, num_segments, num_frames, _ = target_segments.shape
    active_mask = segment_mask.reshape(-1)
    active_count = int(active_mask.sum().item())

    if active_count <= 0:
        onset_loss = target_segments.new_tensor(0.0)
        sustain_loss = target_segments.new_tensor(0.0)
        pedal_loss = target_segments.new_tensor(0.0)
        velocity_loss = target_segments.new_tensor(0.0)
    else:
        flat_targets = target_segments.reshape(batch_size * num_segments, num_frames, -1)[active_mask]
        flat_onset_logits = outputs["onset_logits"].reshape(
            batch_size * num_segments,
            num_frames,
            roll_config.pitch_count,
        )[active_mask]
        flat_sustain_logits = outputs["sustain_logits"].reshape(
            batch_size * num_segments,
            num_frames,
            roll_config.pitch_count,
        )[active_mask]
        flat_pedal_logits = outputs["pedal_logits"].reshape(
            batch_size * num_segments,
            num_frames,
            1,
        )[active_mask]
        flat_velocity = outputs["velocity"].reshape(
            batch_size * num_segments,
            num_frames,
            roll_config.pitch_count,
        )[active_mask]

        pitch_count = roll_config.pitch_count
        onset_target = flat_targets[:, :, :pitch_count]
        sustain_target = flat_targets[:, :, pitch_count : pitch_count * 2]
        velocity_target = flat_targets[:, :, pitch_count * 2 : pitch_count * 3]
        pedal_target = flat_targets[:, :, -1:]

        # onset (Weighted BCE)
        onset_loss = _weighted_binary_cross_entropy_with_logits(
            flat_onset_logits,
            onset_target,
            positive_weight=model_config.onset_positive_class_weight,
            negative_weight=model_config.onset_negative_class_weight,
        )
        # sustain & pedal (Balanced BCE)
        sustain_loss = _balanced_bce_with_logits(flat_sustain_logits, sustain_target)
        pedal_loss = _balanced_bce_with_logits(flat_pedal_logits, pedal_target)

        # velocity_lossは目標値で音が鳴っている（onset/sustainがオン）フレームのみを計算
        velocity_active = (onset_target >= 0.5) | (sustain_target >= 0.5)
        if int(velocity_active.sum().item()) == 0:
            velocity_loss = target_segments.new_tensor(0.0)
        else:
            velocity_loss = F.l1_loss(
                flat_velocity[velocity_active],
                velocity_target[velocity_active],
                reduction="mean",
            )

    # KLダイバージェンス損失
    mu = outputs["mu"]
    logvar = outputs["logvar"]
    kl_per_segment = -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(dim=-1)
    if not variational or int(segment_mask.sum().item()) == 0:
        kl_loss = target_segments.new_tensor(0.0)
    else:
        kl_loss = kl_per_segment[segment_mask].mean()
    total = (
        model_config.onset_loss_weight * onset_loss
        + model_config.sustain_loss_weight * sustain_loss
        + model_config.pedal_loss_weight * pedal_loss
        + model_config.velocity_loss_weight * velocity_loss
        + model_config.kl_beta * kl_loss
    )
    return {
        "total": total,
        "onset": onset_loss,
        "sustain": sustain_loss,
        "pedal": pedal_loss,
        "velocity": velocity_loss,
        "kl": kl_loss,
    }


def diffusion_mse_loss(
    predicted_noise: torch.Tensor,
    target_noise: torch.Tensor,
    segment_mask: torch.Tensor,
) -> torch.Tensor:
    # マスク付きのMSE損失（ノイズ予測誤差）の計算
    expanded_mask = segment_mask.unsqueeze(-1).expand_as(predicted_noise)
    if int(expanded_mask.sum().item()) == 0:
        return predicted_noise.new_tensor(0.0)
    return F.mse_loss(predicted_noise[expanded_mask], target_noise[expanded_mask], reduction="mean")
