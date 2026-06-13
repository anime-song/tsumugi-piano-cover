from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src_v2.config import load_experiment_config
from src_v2.data.segment import segment_grid_from_duration, roll_to_score
from src_v2.data.midi import chunk_source_events, load_trimmed_source_events
from src_v2.models.diffusion import ConditionalSegmentDiffusionModel
from src_v2.models.segment_autoencoder import SegmentLatentAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate piano-cover MIDI with segment-latent diffusion.")
    parser.add_argument("--config", default="configs/piano_cover_v2.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--autoencoder-checkpoint", default=None)
    parser.add_argument("--source-midi", required=True)
    parser.add_argument("--output-midi", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--performer-id", type=int)
    parser.add_argument("--piano-id")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sampling-steps", type=int)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_performer_id(args: argparse.Namespace, config) -> int:
    if args.performer_id is not None:
        return int(args.performer_id)
    if args.piano_id is None:
        raise ValueError("either --performer-id or --piano-id is required")
    performer_map = json.loads(Path(config.dataset.piano_to_performer_json).read_text(encoding="utf-8"))
    if args.piano_id not in performer_map:
        raise KeyError(f"piano_id not found in performer map: {args.piano_id}")
    return int(performer_map[args.piano_id])


def build_source_batch(source_midi_path: str, config, performer_id: int, device: torch.device):
    # モデル入力用バッチの構築
    source_events = load_trimmed_source_events(source_midi_path, config.source_chunks)
    chunked = chunk_source_events(source_events, config.source_chunks)

    # セグメント分割用グリッドの作成
    source_duration = max((event.end for event in source_events), default=config.target_roll.frame_seconds)
    segment_grid = segment_grid_from_duration(source_duration, config.target_roll)
    num_chunks = int(chunked["source_features"].shape[0])
    num_segments = int(segment_grid.segments.shape[0])
    batch = {
        "source_features": chunked["source_features"].unsqueeze(0).to(device),
        "source_programs": chunked["source_programs"].unsqueeze(0).to(device),
        "source_drums": chunked["source_drums"].unsqueeze(0).to(device),
        "source_track_roles": chunked["source_track_roles"].unsqueeze(0).to(device),
        "source_note_mask": chunked["source_note_mask"].unsqueeze(0).to(device),
        "source_chunk_times": chunked["source_chunk_times"].unsqueeze(0).to(device),
        "source_chunk_mask": torch.ones((1, num_chunks), dtype=torch.bool, device=device),
        "segment_times": segment_grid.segment_times.unsqueeze(0).to(device),
        "segment_mask": torch.ones((1, num_segments), dtype=torch.bool, device=device),
        "performer_ids": torch.tensor([performer_id], dtype=torch.long, device=device),
    }
    return batch, segment_grid


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    device = resolve_device(args.device)

    # 1. Diffusionモデルのロード
    diffusion_checkpoint = Path(args.checkpoint or (Path(config.runtime.work_dir) / "best.pt"))
    if not diffusion_checkpoint.is_file():
        raise FileNotFoundError(f"diffusion checkpoint not found: {diffusion_checkpoint}")
    diffusion_payload = torch.load(diffusion_checkpoint, map_location=device)

    # 1-1. 生成時はcheckpoint保存時の prediction_type を優先する
    checkpoint_config = diffusion_payload.get("experiment_config")
    if isinstance(checkpoint_config, dict):
        checkpoint_diffusion_config = checkpoint_config.get("diffusion_model")
        if isinstance(checkpoint_diffusion_config, dict):
            checkpoint_prediction_type = checkpoint_diffusion_config.get("prediction_type", "epsilon")
            config.diffusion_model.prediction_type = str(checkpoint_prediction_type)

    # 2. オートエンコーダー（VAE）のチェックポイントの特定
    autoencoder_checkpoint = args.autoencoder_checkpoint
    if autoencoder_checkpoint is None:
        autoencoder_checkpoint = diffusion_payload.get("autoencoder_checkpoint")
    if autoencoder_checkpoint is None:
        fallback = Path(config.runtime.autoencoder_work_dir) / "vae" / "best.pt"
        autoencoder_checkpoint = config.runtime.autoencoder_checkpoint or str(fallback)
    autoencoder_checkpoint = Path(autoencoder_checkpoint)
    if not autoencoder_checkpoint.is_file():
        raise FileNotFoundError(f"autoencoder checkpoint not found: {autoencoder_checkpoint}")

    # 3. 演奏者情報の解決
    performer_id = resolve_performer_id(args, config)
    performer_map = json.loads(Path(config.dataset.piano_to_performer_json).read_text(encoding="utf-8"))
    performer_vocab_size = diffusion_payload.get("performer_vocab_size")
    if performer_vocab_size is None:
        # 重み形状から演奏者数を推測
        performer_vocab_size = diffusion_payload["model_state"]["denoiser.performer_embedding.weight"].shape[0]

    # 4. オートエンコーダーモデルのロード
    autoencoder = SegmentLatentAutoencoder(config.autoencoder_model, config.target_roll).to(device)
    autoencoder_payload = torch.load(autoencoder_checkpoint, map_location=device)
    autoencoder.load_state_dict(autoencoder_payload["model_state"])
    autoencoder.eval()

    # 5. Diffusionモデルのロード
    model = ConditionalSegmentDiffusionModel(config.source_model, config.diffusion_model, performer_vocab_size).to(
        device
    )
    model.load_state_dict(diffusion_payload["model_state"])
    model.eval()

    # 6. 入力バッチの構築と推論（サンプリング＆デコード）の実行
    batch, segment_grid = build_source_batch(args.source_midi, config, performer_id, device)
    latent_shape = (
        1,
        int(batch["segment_mask"].shape[1]),
        config.diffusion_model.latent_dim,
    )

    with torch.no_grad():
        # Diffusionモデルから潜在変数をサンプリング
        latents = model.sample(batch, latent_shape=latent_shape, sampling_steps=args.sampling_steps)
        # 潜在変数をオートエンコーダーでデコード
        decoded = autoencoder.decode(latents)
        recon_roll = autoencoder.reconstruct_roll(
            decoded,
            segment_grid.segment_times,
            segment_grid.segment_valid_lengths,
            segment_grid.num_frames,
        )

    # 7. 再構成されたピアノロールをMIDIスコアに変換して書き出し
    output_midi = Path(args.output_midi)
    output_midi.parent.mkdir(parents=True, exist_ok=True)
    roll_to_score(recon_roll, config.target_roll).dump_midi(str(output_midi))

    # 8. 実行結果のレポート作成と保存
    report = {
        "source_midi": args.source_midi,
        "diffusion_checkpoint": str(diffusion_checkpoint),
        "autoencoder_checkpoint": str(autoencoder_checkpoint),
        "performer_id": performer_id,
        "prediction_type": config.diffusion_model.prediction_type,
        "num_source_chunks": int(batch["source_chunk_mask"].sum().item()),
        "num_segments": int(batch["segment_mask"].sum().item()),
        "num_frames": segment_grid.num_frames,
        "sampling_steps": args.sampling_steps or config.diffusion_model.sampling_steps,
        "output_midi": str(output_midi),
    }
    if args.output_json is not None:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
