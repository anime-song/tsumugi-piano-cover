from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from recipes.data.dataset import SegmentAutoencoderDataset, collate_autoencoder_samples
from recipes.train_common import build_splits, move_batch_to_device, resolve_device
from tsumugi_piano_cover.config import load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute latent scaling statistics from a trained VAE checkpoint.")
    parser.add_argument("--config", default="configs/piano_cover_v2.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-train-pairs", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--output-json")
    parser.add_argument("--write-to-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"autoencoder checkpoint not found: {checkpoint_path}")

    train_pairs, _, _ = build_splits(config, limit_train_pairs=args.limit_train_pairs)
    autoencoder_num_workers = (
        config.runtime.autoencoder_num_workers
        if config.runtime.autoencoder_num_workers is not None
        else config.runtime.num_workers
    )
    dataset = SegmentAutoencoderDataset(
        train_pairs,
        config.target_roll,
        max_cached_songs=config.runtime.autoencoder_max_cached_songs,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=autoencoder_num_workers,
        persistent_workers=autoencoder_num_workers > 0,
        collate_fn=collate_autoencoder_samples,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    stage = str(checkpoint.get("autoencoder_stage", "vae"))
    if stage != "vae":
        raise ValueError(f"latent scaling is only defined for VAE checkpoints, got stage={stage!r}")

    from tsumugi_piano_cover.models.segment_autoencoder import SegmentLatentAutoencoder

    model = SegmentLatentAutoencoder(config.autoencoder_model, config.target_roll).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    latent_sum: torch.Tensor | None = None
    latent_square_sum: torch.Tensor | None = None
    latent_abs_max = 0.0
    total_count = 0

    with torch.no_grad():
        progress = tqdm(loader, desc="latent stats", dynamic_ncols=True)
        for batch_index, batch in enumerate(progress):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            batch = move_batch_to_device(batch, device)
            latents, _, _ = model.encode(
                batch["target_segments"],
                sample_posterior=False,
                variational=True,
            )
            active_latents = latents[batch["segment_mask"]]
            if int(active_latents.shape[0]) == 0:
                continue

            active_latents = active_latents.float()
            batch_sum = active_latents.sum(dim=0)
            batch_square_sum = active_latents.square().sum(dim=0)
            latent_sum = batch_sum if latent_sum is None else latent_sum + batch_sum
            latent_square_sum = batch_square_sum if latent_square_sum is None else latent_square_sum + batch_square_sum
            total_count += int(active_latents.shape[0])
            latent_abs_max = max(latent_abs_max, float(active_latents.abs().max().item()))
            progress.set_postfix(segments=total_count)

    if total_count == 0 or latent_sum is None or latent_square_sum is None:
        raise ValueError("no active latent vectors were collected")

    latent_mean = latent_sum / total_count
    latent_var = latent_square_sum / total_count - latent_mean.square()
    latent_std = latent_var.clamp_min(1.0e-12).sqrt()
    latent_scale = float(latent_std.mean().item())

    report = {
        "checkpoint": str(checkpoint_path),
        "stage": stage,
        "train_pairs": len(train_pairs),
        "segments": total_count,
        "latent_dim": int(latent_mean.shape[0]),
        "latent_scale": latent_scale,
        "mean_abs_mean": float(latent_mean.abs().mean().item()),
        "std_mean": float(latent_std.mean().item()),
        "std_min": float(latent_std.min().item()),
        "std_max": float(latent_std.max().item()),
        "abs_max": latent_abs_max,
    }

    if args.write_to_checkpoint:
        checkpoint["latent_scale"] = latent_scale
        checkpoint["latent_mean"] = latent_mean.cpu()
        checkpoint["latent_std"] = latent_std.cpu()
        torch.save(checkpoint, checkpoint_path)

    if args.output_json is not None:
        output_json = Path(args.output_json).expanduser()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
