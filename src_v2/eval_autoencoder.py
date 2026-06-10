from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src_v2.config import load_experiment_config
from src_v2.data.segment import (
    midi_to_target_roll,
    roll_to_score,
    segment_grid_from_roll,
    segments_to_roll,
)
from src_v2.data.midi import load_trimmed_target_events
from src_v2.models.segment_autoencoder import (
    SegmentLatentAutoencoder,
    decoder_outputs_to_segment_rolls,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate segment autoencoder reconstruction on one piano MIDI.")
    parser.add_argument("--config", default="configs/piano_cover_v2.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--stage", choices=("auto", "ae", "vae"), default="auto")
    parser.add_argument("--midi-path", required=True)
    parser.add_argument("--output-midi", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-posterior", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def binary_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float) -> dict[str, float]:
    # 適合率、再現率、F1値を算出
    pred_active = pred >= threshold
    target_active = target >= threshold
    tp = int((pred_active & target_active).sum().item())
    fp = int((pred_active & ~target_active).sum().item())
    fn = int((~pred_active & target_active).sum().item())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def velocity_mae_on_active_frames(
    pred_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    target_onset: torch.Tensor,
    target_sustain: torch.Tensor,
) -> float:
    # 発音開始または持続中のフレームを対象にベロシティのMAEを算出
    active = (target_onset >= 0.5) | (target_sustain >= 0.5)
    if int(active.sum().item()) == 0:
        return 0.0
    return float((pred_velocity[active] - target_velocity[active]).abs().mean().item())


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    device = resolve_device(args.device)
    default_checkpoint = Path(config.runtime.autoencoder_work_dir) / "vae" / "best.pt"
    checkpoint_path = Path(args.checkpoint or default_checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"autoencoder checkpoint not found: {checkpoint_path}")

    # MIDIファイルの読み込みとセグメント表現の作成
    notes, pedals = load_trimmed_target_events(
        args.midi_path, min_duration_seconds=config.target_roll.min_duration_seconds
    )
    target_roll = midi_to_target_roll(notes, pedals, config.target_roll)
    segment_song = segment_grid_from_roll(target_roll, config.target_roll)

    model = SegmentLatentAutoencoder(config.autoencoder_model, config.target_roll).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    stage = payload.get("autoencoder_stage", "vae") if args.stage == "auto" else args.stage
    use_variational = stage == "vae"

    # セグメントデータのエンコードとデコード
    segments = segment_song.segments.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(segments, sample_posterior=args.sample_posterior, variational=use_variational)
        recon_segment_rolls = decoder_outputs_to_segment_rolls(outputs, config.autoencoder_model, config.target_roll)[
            0
        ].cpu()

    # セグメントを結合してピアノロールを再構成
    recon_roll = segments_to_roll(
        recon_segment_rolls,
        segment_song.segment_times,
        segment_song.num_frames,
        config.target_roll,
    )

    # MIDIファイルへ出力
    output_midi = Path(args.output_midi)
    output_midi.parent.mkdir(parents=True, exist_ok=True)
    roll_to_score(recon_roll, config.target_roll).dump_midi(str(output_midi))

    # 各特徴量ごとの評価用テンソルを切り出す
    pitch_count = config.target_roll.pitch_count
    target_onset = target_roll[:, :pitch_count]
    target_sustain = target_roll[:, pitch_count : pitch_count * 2]
    target_velocity = target_roll[:, pitch_count * 2 : pitch_count * 3]
    target_pedal = target_roll[:, -1:]

    recon_onset = recon_roll[:, :pitch_count]
    recon_sustain = recon_roll[:, pitch_count : pitch_count * 2]
    recon_velocity = recon_roll[:, pitch_count * 2 : pitch_count * 3]
    recon_pedal = recon_roll[:, -1:]

    metrics = {
        "frame_mse": float(torch.mean((recon_roll - target_roll).square()).item()),
        "onset": binary_metrics(recon_onset, target_onset, config.target_roll.onset_threshold),
        "sustain": binary_metrics(recon_sustain, target_sustain, config.target_roll.sustain_threshold),
        "pedal": binary_metrics(recon_pedal, target_pedal, config.target_roll.pedal_threshold),
        "velocity_mae_on_active_frames": velocity_mae_on_active_frames(
            recon_velocity, target_velocity, target_onset, target_sustain
        ),
        "original_frames": int(target_roll.shape[0]),
        "reconstructed_frames": int(recon_roll.shape[0]),
    }

    report = {
        "midi_path": args.midi_path,
        "checkpoint": str(checkpoint_path),
        "stage": stage,
        "sample_posterior": bool(args.sample_posterior),
        "num_segments": int(segment_song.segments.shape[0]),
        "segment_shape": list(segment_song.segments.shape),
        "latent_shape": list(outputs["latents"].shape),
        "output_midi": str(output_midi),
        "metrics": metrics,
    }

    # 評価結果レポートをJSONとして保存
    if args.output_json is not None:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
