"""学習中に実際にサンプリングして生成品質を測る。

拡散モデルの MSE は「ノイズをどれだけ当てられるか」しか見ておらず、生成結果が
音楽として成立しているかとはほとんど相関しない。無音に潰れる / 音符を敷き詰める
といった典型的な破綻は loss が下がっていても起こるため、定期的に逆拡散を最後まで
回して、正解のピアノカバーと比べた指標を出す。

指標は3層に分かれている。
1. 破綻検知  : 音符密度・発音フレーム比・潜在の統計。ここが壊れていれば以降は見る必要がない
2. 正解一致  : onset / sustain / pedal の F1 と velocity MAE。VAE 再構成を天井として併記する
3. 分布一致  : ピッチクラス分布の重なりや密度比。カバーの正解は一意でないため、
               フレーム単位の一致より頑健な指標として見る
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tsumugi_piano_cover.config import ExperimentConfig, TargetRollConfig
from tsumugi_piano_cover.data.segment import segments_to_roll, write_roll_midi
from tsumugi_piano_cover.latent_scaling import LatentNormalizer


@dataclass
class RollParts:
    onset: torch.Tensor
    sustain: torch.Tensor
    velocity: torch.Tensor
    pedal: torch.Tensor


def split_roll(roll: torch.Tensor, config: TargetRollConfig) -> RollParts:
    pitch_count = config.pitch_count
    return RollParts(
        onset=roll[:, :pitch_count],
        sustain=roll[:, pitch_count : pitch_count * 2],
        velocity=roll[:, pitch_count * 2 : pitch_count * 3],
        pedal=roll[:, -1],
    )


def _binary_scores(pred: torch.Tensor, target: torch.Tensor, threshold: float) -> tuple[float, float, float]:
    pred_active = pred >= threshold
    target_active = target >= threshold
    true_positive = float((pred_active & target_active).sum().item())
    false_positive = float((pred_active & ~target_active).sum().item())
    false_negative = float((~pred_active & target_active).sum().item())
    precision = true_positive / max(1.0, true_positive + false_positive)
    recall = true_positive / max(1.0, true_positive + false_negative)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def _pitch_class_histogram(onset: torch.Tensor, config: TargetRollConfig) -> torch.Tensor:
    # ピッチごとの発音回数を12音に畳む
    counts = (onset >= config.onset_threshold).float().sum(dim=0)
    pitch_classes = (torch.arange(config.pitch_count, device=counts.device) + config.pitch_min) % 12
    histogram = torch.zeros(12, device=counts.device)
    histogram.index_add_(0, pitch_classes, counts)
    total = histogram.sum()
    return histogram / total if total > 0 else histogram


def describe_roll(roll: torch.Tensor, config: TargetRollConfig, prefix: str) -> dict[str, float]:
    """1本のロールだけから計算できる統計（正解を必要としない破綻検知用）。"""
    parts = split_roll(roll, config)
    num_frames = max(1, int(roll.shape[0]))
    duration = num_frames * config.frame_seconds

    onset_active = parts.onset >= config.onset_threshold
    sustain_active = parts.sustain >= config.sustain_threshold
    note_count = float(onset_active.sum().item())
    frames_with_note = sustain_active.any(dim=1)
    active_frame_count = float(frames_with_note.sum().item())
    polyphony = sustain_active.float().sum(dim=1)
    used_pitches = sustain_active.any(dim=0).nonzero(as_tuple=False).flatten()

    return {
        f"{prefix}/note_density_per_sec": note_count / duration,
        f"{prefix}/active_frame_ratio": active_frame_count / num_frames,
        f"{prefix}/polyphony_mean": float(polyphony[frames_with_note].mean().item()) if active_frame_count else 0.0,
        f"{prefix}/pitch_range": float(used_pitches.max() - used_pitches.min()) if used_pitches.numel() else 0.0,
        f"{prefix}/pedal_ratio": float((parts.pedal >= config.pedal_threshold).float().mean().item()),
    }


def compare_rolls(
    predicted: torch.Tensor,
    target: torch.Tensor,
    config: TargetRollConfig,
    prefix: str,
) -> dict[str, float]:
    """正解ロールと比べた一致度。predicted と target は同じ長さに揃えてから渡すこと。"""
    pred = split_roll(predicted, config)
    gold = split_roll(target, config)

    metrics: dict[str, float] = {}
    for name, pred_part, gold_part, threshold in (
        ("onset", pred.onset, gold.onset, config.onset_threshold),
        ("sustain", pred.sustain, gold.sustain, config.sustain_threshold),
        ("pedal", pred.pedal, gold.pedal, config.pedal_threshold),
    ):
        precision, recall, f1 = _binary_scores(pred_part, gold_part, threshold)
        metrics[f"{prefix}/{name}_f1"] = f1
        metrics[f"{prefix}/{name}_precision"] = precision
        metrics[f"{prefix}/{name}_recall"] = recall

    active = (gold.onset >= config.onset_threshold) | (gold.sustain >= config.sustain_threshold)
    if int(active.sum().item()) > 0:
        metrics[f"{prefix}/velocity_mae"] = float((pred.velocity[active] - gold.velocity[active]).abs().mean().item())
    else:
        metrics[f"{prefix}/velocity_mae"] = 0.0

    # 分布ベースの指標。カバーの正解は一意でないのでフレーム一致より頑健
    pred_histogram = _pitch_class_histogram(pred.onset, config)
    gold_histogram = _pitch_class_histogram(gold.onset, config)
    metrics[f"{prefix}/pitch_class_overlap"] = float(torch.minimum(pred_histogram, gold_histogram).sum().item())

    gold_notes = float((gold.onset >= config.onset_threshold).sum().item())
    pred_notes = float((pred.onset >= config.onset_threshold).sum().item())
    metrics[f"{prefix}/note_density_ratio"] = pred_notes / gold_notes if gold_notes > 0 else 0.0
    return metrics


def render_roll_image(rolls: list[tuple[str, torch.Tensor]], config: TargetRollConfig, width: int = 900) -> np.ndarray:
    """生成と正解のピアノロールを縦に並べた RGB 画像を作る（matplotlib 不要）。

    グレーが sustain、明るい色が onset。一目で「無音」「敷き詰め」「それらしい構造」が判別できる。
    """
    colors = ((90, 200, 255), (255, 170, 80), (160, 160, 160))
    row_scale = 2
    panels: list[np.ndarray] = []

    for index, (_, roll) in enumerate(rolls):
        parts = split_roll(roll, config)
        onset = (parts.onset >= config.onset_threshold).float()
        sustain = (parts.sustain >= config.sustain_threshold).float()
        frames = max(1, int(sustain.shape[0]))
        # 時間軸を max pooling で width 列に圧縮する（短い音が消えないように max を使う）
        bucket = max(1, frames // width)
        usable = (frames // bucket) * bucket

        def compress(x: torch.Tensor) -> np.ndarray:
            if usable == 0:
                return np.zeros((config.pitch_count, 1), dtype=np.float32)
            v = x[:usable].reshape(usable // bucket, bucket, -1).amax(dim=1)
            return v.transpose(0, 1).flip(0).cpu().numpy()

        sustain_image = compress(sustain)
        onset_image = compress(onset)
        color = np.array(colors[index % len(colors)], dtype=np.float32)
        panel = sustain_image[..., None] * (color * 0.45) + onset_image[..., None] * color
        panel = np.clip(panel, 0, 255).astype(np.uint8)
        panel = np.repeat(panel, row_scale, axis=0)
        panels.append(panel)
        panels.append(np.full((4, panel.shape[1], 3), 40, dtype=np.uint8))

    panels.pop()
    target_width = max(p.shape[1] for p in panels)
    padded = [np.pad(p, ((0, 0), (0, target_width - p.shape[1]), (0, 0))) for p in panels]
    return np.concatenate(padded, axis=0)


@torch.no_grad()
def evaluate_generation(
    model,
    autoencoder,
    batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    config: ExperimentConfig,
    normalizer: LatentNormalizer,
    epoch: int,
    sample_dir: Path | None = None,
    sampling_steps: int | None = None,
    metric_prefix: str = "gen",
    target_metric_prefix: str = "target",
    recon_metric_prefix: str = "recon",
    sample_prefix: str = "",
) -> tuple[dict[str, float], list[tuple[str, np.ndarray]]]:
    """固定の数曲を最後までサンプリングし、指標と可視化画像を返す。

    初期ノイズは曲ごとに固定シードで引くので、エポック間の比較ができる。
    """
    model.eval()
    roll_config = config.target_roll
    steps = sampling_steps or config.diffusion_training.generation_eval_sampling_steps
    accumulated: dict[str, list[float]] = {}
    images: list[tuple[str, np.ndarray]] = []
    generator = torch.Generator(device=device)

    for index, batch in enumerate(batches):
        name = batch["metadata"][0][1] if isinstance(batch.get("metadata"), list) else f"song{index}"
        batch = {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in batch.items()}
        num_segments = int(batch["segment_mask"].shape[1])
        num_frames = int(batch["target_num_frames"][0].item())
        segment_times = batch["segment_times"][0]

        target_latents, _, _ = autoencoder.encode(batch["target_segments"], sample_posterior=False)
        target_latents = target_latents.float()
        teacher_mask = batch.get("teacher_overlap_mask")
        use_teacher_overlap = isinstance(teacher_mask, torch.Tensor) and bool(teacher_mask.any().item())

        # 1. 逆拡散を最後まで回す
        generator.manual_seed(config.seed * 7919 + index)
        latents = model.sample(
            batch,
            latent_shape=(1, num_segments, config.diffusion_model.latent_dim),
            sampling_steps=steps,
            generator=generator,
            known_latents=normalizer.normalize(target_latents) if use_teacher_overlap else None,
            known_mask=teacher_mask if use_teacher_overlap else None,
        ).float()
        # 正規化空間での統計。学習時の潜在は mean 0 / std 1 なので、ここがずれていれば
        # デコーダにとって未知の入力を渡していることになる
        accumulated.setdefault(f"{metric_prefix}/latent_mean_abs", []).append(float(latents.mean().abs().item()))
        accumulated.setdefault(f"{metric_prefix}/latent_std", []).append(float(latents.std().item()))

        # 2. 潜在をロールへ戻す
        generated_roll = autoencoder.reconstruct_roll(
            autoencoder.decode(normalizer.denormalize(latents)), segment_times, None, num_frames
        )
        # 3. 正解ロールと、VAE 再構成（達成可能な上限）
        target_roll = segments_to_roll(batch["target_segments"][0], segment_times, num_frames, roll_config)
        # VAE を素通りさせた結果。正規化は往復すると恒等なのでここでは使わない
        recon_roll = autoencoder.reconstruct_roll(autoencoder.decode(target_latents), segment_times, None, num_frames)

        length = min(generated_roll.shape[0], target_roll.shape[0], recon_roll.shape[0])
        generated_roll = generated_roll[:length]
        target_roll = target_roll[:length].to(generated_roll.device)
        recon_roll = recon_roll[:length]

        window_start = max(0, min(int(batch.get("window_frame_start", 0)), length))
        window_end = max(window_start + 1, min(int(batch.get("window_frame_end", length)), length))
        metric_start = max(0, min(int(batch.get("generated_frame_start", window_start)), window_end - 1))
        metric_end = max(metric_start + 1, min(int(batch.get("generated_frame_end", window_end)), window_end))
        metric_generated_roll = generated_roll[metric_start:metric_end]
        metric_target_roll = target_roll[metric_start:metric_end]
        metric_recon_roll = recon_roll[metric_start:metric_end]
        image_generated_roll = generated_roll[window_start:window_end]
        image_target_roll = target_roll[window_start:window_end]

        for key, value in describe_roll(metric_generated_roll, roll_config, metric_prefix).items():
            accumulated.setdefault(key, []).append(value)
        for key, value in describe_roll(metric_target_roll, roll_config, target_metric_prefix).items():
            accumulated.setdefault(key, []).append(value)
        for key, value in compare_rolls(metric_generated_roll, metric_target_roll, roll_config, metric_prefix).items():
            accumulated.setdefault(key, []).append(value)
        # VAE 再構成を天井として併記する。gen がこれに近ければ拡散側はやることをやっている
        for key, value in compare_rolls(
            metric_recon_roll,
            metric_target_roll,
            roll_config,
            recon_metric_prefix,
        ).items():
            accumulated.setdefault(key, []).append(value)

        if index < 2:
            images.append(
                (
                    name,
                    render_roll_image(
                        [("generated", image_generated_roll), ("target", image_target_roll)], roll_config
                    ),
                )
            )
        if sample_dir is not None:
            sample_dir.mkdir(parents=True, exist_ok=True)
            write_roll_midi(
                image_generated_roll.cpu(),
                roll_config,
                sample_dir / f"epoch{epoch:04d}_{sample_prefix}{name}.mid",
            )

    metrics = {key: float(np.mean(values)) for key, values in accumulated.items()}
    if f"{metric_prefix}/onset_f1" in metrics and metrics.get(f"{recon_metric_prefix}/onset_f1", 0.0) > 0.0:
        # 達成可能な上限に対してどこまで来ているか
        metrics[f"{metric_prefix}/onset_f1_vs_recon"] = (
            metrics[f"{metric_prefix}/onset_f1"] / metrics[f"{recon_metric_prefix}/onset_f1"]
        )
    return metrics, images
