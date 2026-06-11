from __future__ import annotations

import torch
import torch.nn.functional as F
from external.amt_model.models.interval_boundaries import PitchIntervalTargets, gather_boundary_targets
from external.amt_model.models.semi_crf import compute_pitch_interval_loss

from src_v2.config import SegmentAutoencoderConfig, TargetRollConfig
from src_v2.models.segment_semi_crf import predict_interval_boundary_logits


def _balanced_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    # 正負の不均衡を補正する Balanced BCE
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have the same shape")
    # boolean indexingや .any() を使うとGPU-CPU同期が発生して学習が遅くなるため、
    # マスク乗算と総和を使って計算します
    positive = (targets >= 0.5).float()
    negative = 1.0 - positive

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    pos_count = positive.sum().clamp_min(1.0)
    neg_count = negative.sum().clamp_min(1.0)

    positive_loss = (bce * positive).sum() / pos_count
    negative_loss = (bce * negative).sum() / neg_count

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
    # boolean indexingや .any() を使うとGPU-CPU同期が発生して学習が遅くなるため、
    # マスク乗算と総和を使って計算します
    positive = (targets >= 0.5).float()
    negative = 1.0 - positive

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    pos_count = positive.sum().clamp_min(1.0)
    neg_count = negative.sum().clamp_min(1.0)

    positive_loss = (bce * positive).sum() / pos_count
    negative_loss = (bce * negative).sum() / neg_count

    return float(positive_weight) * positive_loss + float(negative_weight) * negative_loss


def _resolve_interval_targets(
    interval_targets: list[PitchIntervalTargets],
    active_mask: torch.Tensor,
    batch_size: int,
    num_segments: int,
) -> list[PitchIntervalTargets]:
    # flatten 後の active segment と interval target の対応を揃える
    expected_count = batch_size * num_segments
    if len(interval_targets) != expected_count:
        raise ValueError(
            "interval_targets must align with flattened segments, "
            f"got {len(interval_targets)} targets for {expected_count} segments"
        )
    return [interval_targets[index] for index, is_active in enumerate(active_mask.tolist()) if is_active]


def autoencoder_reconstruction_loss(
    outputs: dict[str, torch.Tensor],
    target_segments: torch.Tensor,
    segment_mask: torch.Tensor,
    model_config: SegmentAutoencoderConfig,
    roll_config: TargetRollConfig,
    variational: bool,
    interval_targets: list[PitchIntervalTargets],
    segment_valid_lengths: torch.Tensor,
    boundary_predictor: torch.nn.Module | None = None,
) -> dict[str, torch.Tensor]:
    # semi-CRF 再構成損失: interval + boundary + pedal + velocity + KL
    batch_size, num_segments, num_frames, _ = target_segments.shape
    active_mask = segment_mask.reshape(-1)
    active_count = int(active_mask.sum().item())

    if active_count <= 0:
        pedal_loss = target_segments.new_tensor(0.0)
        velocity_loss = target_segments.new_tensor(0.0)
        interval_loss = target_segments.new_tensor(0.0)
        boundary_loss = target_segments.new_tensor(0.0)
        onset_loss = target_segments.new_tensor(0.0)
        sustain_loss = target_segments.new_tensor(0.0)
    else:
        flat_targets = target_segments.reshape(batch_size * num_segments, num_frames, -1)[active_mask]
        pitch_count = roll_config.pitch_count
        onset_target = flat_targets[:, :, :pitch_count]
        sustain_target = flat_targets[:, :, pitch_count : pitch_count * 2]
        velocity_target = flat_targets[:, :, pitch_count * 2 : pitch_count * 3]
        pedal_target = flat_targets[:, :, -1:]

        if model_config.decoder_mode == "frame":
            flat_logits = outputs["frame_logits"].reshape(batch_size * num_segments, num_frames, -1)[active_mask]
            onset_logits = flat_logits[:, :, :pitch_count]
            sustain_logits = flat_logits[:, :, pitch_count : pitch_count * 2]
            velocity_logits = flat_logits[:, :, pitch_count * 2 : pitch_count * 3]
            pedal_logits = flat_logits[:, :, -1:]

            onset_loss = _weighted_binary_cross_entropy_with_logits(
                onset_logits,
                onset_target,
                positive_weight=model_config.onset_positive_class_weight,
                negative_weight=model_config.onset_negative_class_weight,
            )
            sustain_loss = _weighted_binary_cross_entropy_with_logits(
                sustain_logits,
                sustain_target,
                positive_weight=model_config.sustain_positive_class_weight,
                negative_weight=model_config.sustain_negative_class_weight,
            )
            pedal_loss = _balanced_bce_with_logits(pedal_logits, pedal_target)

            velocity_active = ((onset_target >= 0.5) | (sustain_target >= 0.5)).float()
            active_count = velocity_active.sum().clamp_min(1.0)
            velocity_diff = F.l1_loss(torch.sigmoid(velocity_logits), velocity_target, reduction="none")
            velocity_loss = (velocity_diff * velocity_active).sum() / active_count
            
            interval_loss = target_segments.new_tensor(0.0)
            boundary_loss = target_segments.new_tensor(0.0)
        else:
            flat_pedal_logits = outputs["pedal_logits"].reshape(batch_size * num_segments, num_frames, 1)[active_mask]
            flat_velocity = outputs["velocity"].reshape(
                batch_size * num_segments,
                num_frames,
                roll_config.pitch_count,
            )[active_mask]

            pedal_loss = _balanced_bce_with_logits(flat_pedal_logits, pedal_target)

            velocity_active = ((onset_target >= 0.5) | (sustain_target >= 0.5)).float()
            active_count = velocity_active.sum().clamp_min(1.0)
            velocity_diff = F.l1_loss(flat_velocity, velocity_target, reduction="none")
            velocity_loss = (velocity_diff * velocity_active).sum() / active_count

            active_interval_targets = _resolve_interval_targets(interval_targets, active_mask, batch_size, num_segments)
            flat_valid_lengths = segment_valid_lengths.reshape(-1)[active_mask]
            flat_interval_query = outputs["interval_query"].reshape(
                batch_size * num_segments,
                num_frames,
                roll_config.pitch_count,
                model_config.semi_crf_head_dim,
            )[active_mask]
            flat_interval_key = outputs["interval_key"].reshape(
                batch_size * num_segments,
                num_frames,
                roll_config.pitch_count,
                model_config.semi_crf_head_dim,
            )[active_mask]
            flat_interval_diag = outputs["interval_diag"].reshape(
                batch_size * num_segments,
                num_frames,
                roll_config.pitch_count,
            )[active_mask]

            # semi-CRF 本体の区間損失
            interval_loss, _, _ = compute_pitch_interval_loss(
                flat_interval_query,
                flat_interval_key,
                flat_interval_diag,
                [target.intervals for target in active_interval_targets],
                flat_valid_lengths,
                length_scaling=model_config.semi_crf_length_scaling,
                length_penalty=model_config.semi_crf_length_penalty,
                track_batch_size=model_config.semi_crf_track_batch_size,
                false_negative_cost=model_config.semi_crf_false_negative_cost,
                false_positive_cost=model_config.semi_crf_false_positive_cost
            )

            # 必要なら区間の onset/offset の存在も別ヘッドで学習する
            boundary_loss = target_segments.new_tensor(0.0)
            if boundary_predictor is not None:
                flat_pitch_features = outputs["pitch_features"].reshape(
                    batch_size * num_segments,
                    num_frames,
                    roll_config.pitch_count,
                    model_config.semi_crf_pitch_feature_dim,
                )[active_mask]
                boundary_logits, entries = predict_interval_boundary_logits(
                    boundary_predictor,
                    flat_pitch_features,
                    [target.intervals for target in active_interval_targets],
                )
                if boundary_logits is not None and entries:
                    has_onset, has_offset, _, _ = gather_boundary_targets(
                        active_interval_targets,
                        entries,
                        device=boundary_logits.device,
                    )
                    boundary_targets = torch.stack([has_onset, has_offset], dim=-1)
                    boundary_loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary_targets)
            
            onset_loss = target_segments.new_tensor(0.0)
            sustain_loss = target_segments.new_tensor(0.0)

    mu = outputs["mu"]
    logvar = outputs["logvar"]
    kl_per_segment = -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(dim=-1)
    if not variational or int(segment_mask.sum().item()) == 0:
        kl_loss = target_segments.new_tensor(0.0)
    else:
        kl_loss = kl_per_segment[segment_mask].mean()

    if model_config.decoder_mode == "frame":
        total = (
            model_config.onset_loss_weight * onset_loss
            + model_config.sustain_loss_weight * sustain_loss
            + model_config.pedal_loss_weight * pedal_loss
            + model_config.velocity_loss_weight * velocity_loss
            + model_config.kl_beta * kl_loss
        )
    else:
        total = (
            interval_loss
            + model_config.interval_presence_loss_weight * boundary_loss
            + model_config.pedal_loss_weight * pedal_loss
            + model_config.velocity_loss_weight * velocity_loss
            + model_config.kl_beta * kl_loss
        )

    return {
        "total": total,
        "interval": interval_loss,
        "boundary": boundary_loss,
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
    # 有効な segment だけでノイズ予測誤差を計算する
    expanded_mask = segment_mask.unsqueeze(-1).expand_as(predicted_noise)
    if int(expanded_mask.sum().item()) == 0:
        return predicted_noise.new_tensor(0.0)
    return F.mse_loss(predicted_noise[expanded_mask], target_noise[expanded_mask], reduction="mean")
