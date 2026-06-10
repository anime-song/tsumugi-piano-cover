from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src_v2.config import ExperimentConfig, load_experiment_config
from src_v2.data.dataset import SegmentAutoencoderDataset, SegmentSongBatchSampler, collate_autoencoder_samples
from src_v2.models.losses import autoencoder_reconstruction_loss
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
    parser = argparse.ArgumentParser(description="Train segment-latent autoencoder for piano MIDI.")
    parser.add_argument("--config", default="configs/piano_cover_v2.yaml")
    # ae（AE事前学習）または vae（VAE微調整）
    parser.add_argument("--stage", choices=("ae", "vae"), default="ae")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-train-pairs", type=int)
    parser.add_argument("--limit-val-pairs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume-from")
    parser.add_argument("--init-from")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-name")
    return parser.parse_args()


def resolve_stage_work_dir(config: ExperimentConfig, stage: str) -> Path:
    return Path(config.runtime.autoencoder_work_dir) / stage


def resolve_stage_init_checkpoint(config: ExperimentConfig, override: str | None, stage: str) -> Path | None:
    if override is not None:
        return Path(override).expanduser()
    if stage != "vae":
        return None
    return resolve_stage_work_dir(config, "ae") / "best.pt"


def load_model_weights(model: SegmentLatentAutoencoder, checkpoint_path: Path, device: torch.device) -> None:
    # 重みのみをロード
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state"], strict=True)


def run_validation(
    model: SegmentLatentAutoencoder,
    loader: DataLoader,
    device: torch.device,
    config: ExperimentConfig,
    epoch: int,
    stage: str,
) -> float:
    # 検証データでの再構成損失の計算
    model.eval()
    losses = []
    use_variational = stage == "vae"
    with torch.no_grad():
        val_progress = tqdm(loader, desc=f"Epoch {epoch} val", dynamic_ncols=True, leave=False)
        for batch in val_progress:
            batch = move_batch_to_device(batch, device)
            # 評価時はランダムサンプリングを無効化（sample_posterior=False）して決定論的にデコード
            outputs = model(
                batch["target_segments"],
                sample_posterior=False,
                variational=use_variational,
            )
            loss_dict = autoencoder_reconstruction_loss(
                outputs=outputs,
                target_segments=batch["target_segments"],
                segment_mask=batch["segment_mask"],
                model_config=config.autoencoder_model,
                roll_config=config.target_roll,
                variational=use_variational,
            )
            loss_value = loss_dict["total"].detach().cpu()
            losses.append(loss_value)
            val_progress.set_postfix(
                total=f"{loss_value.item():.4f}",
                onset=f"{loss_dict['onset'].item():.4f}",
                sustain=f"{loss_dict['sustain'].item():.4f}",
                pedal=f"{loss_dict['pedal'].item():.4f}",
                velocity=f"{loss_dict['velocity'].item():.4f}",
                kl=f"{loss_dict['kl'].item():.4f}",
            )
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
    wandb_run = init_wandb_run(config, job_type="autoencoder")
    use_variational = args.stage == "vae"

    try:
        # データスプリットの作成
        train_pairs, val_pairs, test_pairs = build_splits(
            config,
            limit_train_pairs=args.limit_train_pairs,
            limit_val_pairs=args.limit_val_pairs,
        )
        work_dir = resolve_stage_work_dir(config, args.stage)
        work_dir.mkdir(parents=True, exist_ok=True)

        model = SegmentLatentAutoencoder(config.autoencoder_model, config.target_roll).to(device)
        
        # 初期重みのロード（aeの事前学習モデル、または指定チェックポイント）
        init_checkpoint = resolve_stage_init_checkpoint(config, args.init_from, args.stage)
        if args.resume_from is None and init_checkpoint is not None:
            load_model_weights(model, init_checkpoint, device)
            
        optimizer = AdamW(
            model.parameters(),
            lr=config.autoencoder_training.learning_rate,
            weight_decay=config.autoencoder_training.weight_decay,
        )
        
        # 混合精度用のGradScalerの初期化
        scaler = torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda" and config.autoencoder_training.mixed_precision == "fp16"
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
        print(f"stage={args.stage}")
        print(f"device={device}")
        print(f"train_pairs={len(train_pairs)}")
        print(f"val_pairs={len(val_pairs)}")
        print(f"test_pairs={len(test_pairs)}")
        print(f"autoencoder_parameters={sum(parameter.numel() for parameter in model.parameters())}")
        if init_checkpoint is not None and args.resume_from is None:
            print(f"initialized_from={init_checkpoint}")

        update_wandb_summary(
            wandb_run,
            autoencoder_stage=args.stage,
            train_pairs=len(train_pairs),
            val_pairs=len(val_pairs),
            test_pairs=len(test_pairs),
            autoencoder_parameters=sum(parameter.numel() for parameter in model.parameters()),
            device=str(device),
        )

        if args.dry_run:
            return

        # 訓練用（データ拡張有効）および検証用のDataset
        train_dataset = SegmentAutoencoderDataset(
            train_pairs,
            config.target_roll,
            augment_pitch_shift=True,
            pitch_shift_min_semitones=config.autoencoder_training.pitch_shift_min_semitones,
            pitch_shift_max_semitones=config.autoencoder_training.pitch_shift_max_semitones,
        )
        val_dataset = SegmentAutoencoderDataset(val_pairs, config.target_roll)
        
        # 曲単位でバッチをグループ化するためのバッチサンプラー
        train_batch_sampler = SegmentSongBatchSampler(
            train_dataset,
            batch_size=config.autoencoder_training.batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_batch_sampler = SegmentSongBatchSampler(
            val_dataset,
            batch_size=config.autoencoder_training.batch_size,
            shuffle=False,
            drop_last=False,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=config.runtime.num_workers,
            persistent_workers=config.runtime.num_workers > 0,
            collate_fn=collate_autoencoder_samples,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=config.runtime.num_workers,
            persistent_workers=config.runtime.num_workers > 0,
            collate_fn=collate_autoencoder_samples,
        )

        # 再開バッチ位置の調整
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

        for epoch in range(resume_state.start_epoch, config.autoencoder_training.max_epochs):
            model.train()
            running_total = 0.0
            running_onset = 0.0
            running_sustain = 0.0
            running_pedal = 0.0
            running_velocity = 0.0
            running_kl = 0.0
            batch_count = 0
            
            # 再開時のスキップ等の制御
            epoch_start_batch = resume_state.start_batch_index if epoch == resume_state.start_epoch else 0
            epoch_step_limit = config.autoencoder_training.max_steps_per_epoch
            epoch_step_count = epoch_start_batch // config.autoencoder_training.grad_accum_steps
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

                with get_autocast_context(device, config.autoencoder_training.mixed_precision):
                    outputs = model(
                        batch["target_segments"],
                        sample_posterior=use_variational,
                        variational=use_variational,
                    )
                    loss_dict = autoencoder_reconstruction_loss(
                        outputs=outputs,
                        target_segments=batch["target_segments"],
                        segment_mask=batch["segment_mask"],
                        model_config=config.autoencoder_model,
                        roll_config=config.target_roll,
                        variational=use_variational,
                    )
                    step_loss = loss_dict["total"] / config.autoencoder_training.grad_accum_steps

                # 誤差逆伝播の実行
                if scaler.is_enabled():
                    scaler.scale(step_loss).backward()
                else:
                    step_loss.backward()

                running_total += loss_dict["total"].detach().item()
                running_onset += loss_dict["onset"].detach().item()
                running_sustain += loss_dict["sustain"].detach().item()
                running_pedal += loss_dict["pedal"].detach().item()
                running_velocity += loss_dict["velocity"].detach().item()
                running_kl += loss_dict["kl"].detach().item()
                batch_count += 1

                train_progress.set_postfix(
                    total=f"{running_total / batch_count:.4f}",
                    onset=f"{running_onset / batch_count:.4f}",
                    sustain=f"{running_sustain / batch_count:.4f}",
                    pedal=f"{running_pedal / batch_count:.4f}",
                    velocity=f"{running_velocity / batch_count:.4f}",
                    step=global_step,
                )

                # 勾配累積のステップ数が満たされたらオプティマイザを更新
                should_step = (batch_index + 1) % config.autoencoder_training.grad_accum_steps == 0
                if should_step:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.autoencoder_training.grad_clip_norm)
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    epoch_step_count += 1

                    # 進捗ロギング
                    if global_step % config.runtime.log_every_steps == 0:
                        tqdm.write(
                            f"epoch={epoch} step={global_step} "
                            f"train_total={running_total / batch_count:.4f} "
                            f"train_onset={running_onset / batch_count:.4f} "
                            f"train_sustain={running_sustain / batch_count:.4f} "
                            f"train_pedal={running_pedal / batch_count:.4f} "
                            f"train_velocity={running_velocity / batch_count:.4f} "
                            f"train_kl={running_kl / batch_count:.4f}"
                        )
                        log_wandb_metrics(
                            wandb_run,
                            {
                                "train/total": running_total / batch_count,
                                "train/onset": running_onset / batch_count,
                                "train/sustain": running_sustain / batch_count,
                                "train/pedal": running_pedal / batch_count,
                                "train/velocity": running_velocity / batch_count,
                                "train/kl": running_kl / batch_count,
                                "train/lr": optimizer.param_groups[0]["lr"],
                                "epoch": epoch,
                            },
                            step=global_step,
                        )

                    # 定期的なチェックポイント保存
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
                            extra_state={"autoencoder_stage": args.stage},
                        )

                    if args.max_steps is not None and global_step >= args.max_steps:
                        break
                    if epoch_step_limit is not None and epoch_step_count >= epoch_step_limit:
                        stopped_by_epoch_step_limit = True
                        break

            # 勾配累積の残り（端数分）の更新
            if (
                not stopped_by_epoch_step_limit
                and batch_count > 0
                and batch_count % config.autoencoder_training.grad_accum_steps != 0
            ):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.autoencoder_training.grad_clip_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            # 検証の実行とチェックポイントの更新
            val_loss = run_validation(model, val_loader, device, config, epoch, args.stage)
            is_best = val_loss <= best_val_loss
            best_val_loss = min(best_val_loss, val_loss)
            log_wandb_metrics(wandb_run, {"val/total": val_loss, "epoch": epoch}, step=global_step)
            update_wandb_summary(wandb_run, best_val_loss=best_val_loss)

            # エポック終了時の保存
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
                extra_state={"autoencoder_stage": args.stage},
            )
            # ベストロスの更新による保存
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
                    extra_state={"autoencoder_stage": args.stage},
                )

            # CUDAキャッシュをクリアしてメモリの断片化を防止
            clear_cuda_cache(device, epoch)

            if args.max_steps is not None and global_step >= args.max_steps:
                break
            if (
                config.autoencoder_training.max_steps is not None
                and global_step >= config.autoencoder_training.max_steps
            ):
                break

    finally:
        finish_wandb_run(wandb_run)


if __name__ == "__main__":
    main()
