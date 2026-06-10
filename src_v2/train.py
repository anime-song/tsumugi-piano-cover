from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src_v2.config import ExperimentConfig, load_experiment_config
from src_v2.data.dataset import SegmentDiffusionDataset, collate_diffusion_samples
from src_v2.models.diffusion import ConditionalSegmentDiffusionModel
from src_v2.models.losses import diffusion_mse_loss
from src_v2.models.segment_autoencoder import SegmentLatentAutoencoder
from src_v2.train_common import (
    ResumeState,
    build_splits,
    clear_cuda_cache,
    get_autocast_context,
    load_training_checkpoint,
    move_batch_to_device,
    resolve_device,
    save_checkpoint,
    skip_to_batch,
)
from src_v2.wandb_utils import finish_wandb_run, init_wandb_run, log_wandb_metrics, update_wandb_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train source-conditioned segment-latent diffusion.")
    parser.add_argument("--config", default="configs/piano_cover_v2.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-train-pairs", type=int)
    parser.add_argument("--limit-val-pairs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume-from")
    parser.add_argument("--autoencoder-checkpoint")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-name")
    return parser.parse_args()


def resolve_autoencoder_checkpoint(config: ExperimentConfig, override: str | None) -> Path:
    if override is not None:
        return Path(override).expanduser()
    if config.runtime.autoencoder_checkpoint is not None:
        return Path(config.runtime.autoencoder_checkpoint).expanduser()
    return Path(config.runtime.autoencoder_work_dir) / "vae" / "best.pt"


def load_frozen_autoencoder(
    config: ExperimentConfig, checkpoint_path: Path, device: torch.device
) -> SegmentLatentAutoencoder:
    # オートエンコーダーの重みをフリーズ
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"autoencoder checkpoint not found: {checkpoint_path}")
    model = SegmentLatentAutoencoder(config.autoencoder_model, config.target_roll).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def run_validation(
    model: ConditionalSegmentDiffusionModel,
    autoencoder: SegmentLatentAutoencoder,
    loader: DataLoader,
    device: torch.device,
    config: ExperimentConfig,
    epoch: int,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        val_progress = tqdm(loader, desc=f"Epoch {epoch} val", dynamic_ncols=True, leave=False)
        for batch in val_progress:
            batch = move_batch_to_device(batch, device)
            # ピアノカバーを潜在表現にエンコード
            latents, _, _ = autoencoder.encode(batch["target_segments"], sample_posterior=False)
            
            timesteps = torch.randint(
                0,
                config.diffusion_model.num_train_timesteps,
                (latents.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noise = torch.randn_like(latents)
            
            noisy_latents = model.q_sample(latents, timesteps, noise)
            predicted_noise = model(batch, noisy_latents, timesteps)
            loss = diffusion_mse_loss(predicted_noise, noise, batch["segment_mask"])
            losses.append(loss.detach().cpu())
            val_progress.set_postfix(val_total=f"{loss.item():.4f}")
    if not losses:
        return float("nan")
    return torch.stack(losses).mean().item()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    if args.wandb:
        config.wandb.enabled = True
    if args.wandb_name is not None:
        config.wandb.name = args.wandb_name
    device = resolve_device(args.device)
    wandb_run = init_wandb_run(config, job_type="diffusion")

    try:
        train_pairs, val_pairs, test_pairs = build_splits(
            config,
            limit_train_pairs=args.limit_train_pairs,
            limit_val_pairs=args.limit_val_pairs,
        )
        # 演奏者IDの語彙サイズ
        performer_vocab_size = max(pair.performer_id for pair in (train_pairs + val_pairs + test_pairs)) + 1
        work_dir = Path(config.runtime.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        # フリーズされたオートエンコーダーのロード
        autoencoder_checkpoint = resolve_autoencoder_checkpoint(config, args.autoencoder_checkpoint)
        autoencoder = load_frozen_autoencoder(config, autoencoder_checkpoint, device)
        
        model = ConditionalSegmentDiffusionModel(config.source_model, config.diffusion_model, performer_vocab_size).to(
            device
        )
        model.set_gradient_checkpointing(True)
        optimizer = AdamW(
            model.parameters(),
            lr=config.diffusion_training.learning_rate,
            weight_decay=config.diffusion_training.weight_decay,
        )
        
        # GradScalerの初期化
        scaler = torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda" and config.diffusion_training.mixed_precision == "fp16"
        )
        resume_state = ResumeState()
        if args.resume_from is not None:
            resume_state, _ = load_training_checkpoint(
                Path(args.resume_from).expanduser(),
                model,
                optimizer,
                scaler,
                device,
            )

        print(f"experiment={config.experiment_name}")
        print(f"device={device}")
        print(f"train_pairs={len(train_pairs)}")
        print(f"val_pairs={len(val_pairs)}")
        print(f"test_pairs={len(test_pairs)}")
        print(f"performer_vocab_size={performer_vocab_size}")
        print(f"autoencoder_checkpoint={autoencoder_checkpoint}")
        print(f"diffusion_parameters={sum(parameter.numel() for parameter in model.parameters())}")
        print("gradient_checkpointing=True")

        update_wandb_summary(
            wandb_run,
            train_pairs=len(train_pairs),
            val_pairs=len(val_pairs),
            test_pairs=len(test_pairs),
            performer_vocab_size=performer_vocab_size,
            autoencoder_checkpoint=str(autoencoder_checkpoint),
            diffusion_parameters=sum(parameter.numel() for parameter in model.parameters()),
            gradient_checkpointing=True,
            device=str(device),
        )

        if args.dry_run:
            return

        train_dataset = SegmentDiffusionDataset(train_pairs, config.source_chunks, config.target_roll)
        val_dataset = SegmentDiffusionDataset(val_pairs, config.source_chunks, config.target_roll)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.diffusion_training.batch_size,
            shuffle=True,
            num_workers=config.runtime.num_workers,
            collate_fn=collate_diffusion_samples,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.diffusion_training.batch_size,
            shuffle=False,
            num_workers=config.runtime.num_workers,
            collate_fn=collate_diffusion_samples,
        )

        # 再開時のバッチ位置調整
        if resume_state.start_batch_index == len(train_loader):
            resume_state.start_epoch += 1
            resume_state.start_batch_index = 0
        if resume_state.start_batch_index > len(train_loader):
            raise ValueError(
                f"resume batch index {resume_state.start_batch_index} exceeds train loader length {len(train_loader)}"
            )

        global_step = resume_state.global_step
        best_val_loss = resume_state.best_val_loss
        optimizer.zero_grad(set_to_none=True)

        for epoch in range(resume_state.start_epoch, config.diffusion_training.max_epochs):
            model.train()
            running_total = 0.0
            batch_count = 0
            epoch_start_batch = resume_state.start_batch_index if epoch == resume_state.start_epoch else 0
            epoch_step_limit = config.diffusion_training.max_steps_per_epoch
            epoch_step_count = epoch_start_batch // config.diffusion_training.grad_accum_steps
            stopped_by_epoch_step_limit = False
            train_progress = tqdm(
                skip_to_batch(train_loader, epoch_start_batch),
                total=len(train_loader),
                initial=epoch_start_batch,
                desc=f"Epoch {epoch} train",
                dynamic_ncols=True,
            )

            for batch_index, batch in train_progress:
                if epoch_step_limit is not None and epoch_step_count >= epoch_step_limit:
                    stopped_by_epoch_step_limit = True
                    break
                batch = move_batch_to_device(batch, device)
                
                # ピアノロールを潜在表現にエンコード
                with torch.no_grad():
                    latents, _, _ = autoencoder.encode(batch["target_segments"], sample_posterior=False)

                timesteps = torch.randint(
                    0,
                    config.diffusion_model.num_train_timesteps,
                    (latents.shape[0],),
                    device=device,
                    dtype=torch.long,
                )
                noise = torch.randn_like(latents)

                with get_autocast_context(device, config.diffusion_training.mixed_precision):
                    # 潜在変数にノイズを付与
                    noisy_latents = model.q_sample(latents, timesteps, noise)
                    # 予測ノイズを計算
                    predicted_noise = model(batch, noisy_latents, timesteps)
                    loss = diffusion_mse_loss(predicted_noise, noise, batch["segment_mask"])
                    step_loss = loss / config.diffusion_training.grad_accum_steps

                if scaler.is_enabled():
                    scaler.scale(step_loss).backward()
                else:
                    step_loss.backward()

                running_total += loss.detach().item()
                batch_count += 1
                train_progress.set_postfix(total=f"{running_total / batch_count:.4f}", step=global_step)

                should_step = (batch_index + 1) % config.diffusion_training.grad_accum_steps == 0
                if should_step:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.diffusion_training.grad_clip_norm)
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    epoch_step_count += 1

                    if global_step % config.runtime.log_every_steps == 0:
                        tqdm.write(f"epoch={epoch} step={global_step} train_total={running_total / batch_count:.4f}")
                        log_wandb_metrics(
                            wandb_run,
                            {
                                "train/total": running_total / batch_count,
                                "train/lr": optimizer.param_groups[0]["lr"],
                                "epoch": epoch,
                            },
                            step=global_step,
                        )

                    # チェックポイント保存
                    if global_step % config.runtime.save_every_steps == 0:
                        next_epoch = epoch
                        next_batch_index = batch_index + 1
                        if next_batch_index >= len(train_loader):
                            next_epoch += 1
                            next_batch_index = 0
                        save_checkpoint(
                            work_dir / "last.pt",
                            model,
                            optimizer,
                            scaler,
                            config,
                            epoch,
                            global_step,
                            best_val_loss,
                            next_epoch,
                            next_batch_index,
                            "last",
                            extra_state={
                                "autoencoder_checkpoint": str(autoencoder_checkpoint),
                                "performer_vocab_size": performer_vocab_size,
                            },
                        )

                    if args.max_steps is not None and global_step >= args.max_steps:
                        break
                    if epoch_step_limit is not None and epoch_step_count >= epoch_step_limit:
                        stopped_by_epoch_step_limit = True
                        break

            if (
                not stopped_by_epoch_step_limit
                and batch_count > 0
                and batch_count % config.diffusion_training.grad_accum_steps != 0
            ):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.diffusion_training.grad_clip_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            # 検証とチェックポイント更新
            val_loss = run_validation(model, autoencoder, val_loader, device, config, epoch)
            is_best = val_loss <= best_val_loss
            best_val_loss = min(best_val_loss, val_loss)
            log_wandb_metrics(wandb_run, {"val/total": val_loss, "epoch": epoch}, step=global_step)
            update_wandb_summary(wandb_run, best_val_loss=best_val_loss)

            save_checkpoint(
                work_dir / "last.pt",
                model,
                optimizer,
                scaler,
                config,
                epoch,
                global_step,
                best_val_loss,
                epoch + 1,
                0,
                "last",
                extra_state={
                    "autoencoder_checkpoint": str(autoencoder_checkpoint),
                    "performer_vocab_size": performer_vocab_size,
                },
            )
            if is_best:
                save_checkpoint(
                    work_dir / "best.pt",
                    model,
                    optimizer,
                    scaler,
                    config,
                    epoch,
                    global_step,
                    best_val_loss,
                    epoch + 1,
                    0,
                    "best",
                    extra_state={
                        "autoencoder_checkpoint": str(autoencoder_checkpoint),
                        "performer_vocab_size": performer_vocab_size,
                    },
                )

            # CUDAキャッシュのクリア
            clear_cuda_cache(device, epoch)

            if args.max_steps is not None and global_step >= args.max_steps:
                break
            if config.diffusion_training.max_steps is not None and global_step >= config.diffusion_training.max_steps:
                break

    finally:
        finish_wandb_run(wandb_run)


if __name__ == "__main__":
    main()
