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
    build_lr_scheduler,
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
    parser.add_argument("--init-encoder-only", action="store_true", help="事前学習済みのエンコーダーの重みのみをロードする")
    parser.add_argument("--freeze-encoder-epochs", type=int, default=0, help="エンコーダーをフリーズするエポック数")
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
    default_checkpoint = resolve_stage_work_dir(config, "ae") / "best.pt"
    if not default_checkpoint.is_file():
        print(f"Warning: AE checkpoint not found at {default_checkpoint}. Starting VAE from scratch.")
        return None
    return default_checkpoint


def load_model_weights(
    model: SegmentLatentAutoencoder,
    checkpoint_path: Path,
    device: torch.device,
    encoder_only: bool = False,
) -> None:
    # 重みのみをロード
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload["model_state"]
    if encoder_only:
        # エンコーダーの重みのみを抽出してロード（デコーダー等の他モジュールは現在の初期化状態を維持）
        state_dict = {key: value for key, value in state_dict.items() if key.startswith("encoder.")}
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded ONLY encoder weights from {checkpoint_path}")
    else:
        model.load_state_dict(state_dict, strict=True)


def run_validation(
    model: SegmentLatentAutoencoder,
    loader: DataLoader,
    device: torch.device,
    config: ExperimentConfig,
    epoch: int,
    stage: str,
) -> dict[str, float]:
    # 検証データでの再構成損失の計算
    model.eval()
    losses = []
    tp_onset, fp_onset, fn_onset = 0, 0, 0
    tp_sustain, fp_sustain, fn_sustain = 0, 0, 0
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
                interval_targets=batch["interval_targets"],
                segment_valid_lengths=batch["segment_valid_lengths"],
                boundary_predictor=getattr(model.decoder, "interval_boundary_predictor", None),
            )
            loss_value = loss_dict["total"].detach().cpu()
            losses.append(loss_value)
            postfix = {
                "total": f"{loss_value.item():.4f}",
                "pedal": f"{loss_dict['pedal'].item():.4f}",
                "velocity": f"{loss_dict['velocity'].item():.4f}",
                "kl": f"{loss_dict['kl'].item():.4f}",
            }
            if config.autoencoder_model.decoder_mode == "frame":
                mask = batch["segment_mask"]
                if mask.any():
                    valid_targets = batch["target_segments"][mask]
                    valid_logits = outputs["frame_logits"][mask]
                    pitch_count = config.target_roll.pitch_count
                    
                    onset_target = valid_targets[:, :, :pitch_count] >= config.target_roll.onset_threshold
                    onset_pred = valid_logits[:, :, :pitch_count] >= 0.0
                    tp_onset += int((onset_pred & onset_target).sum().item())
                    fp_onset += int((onset_pred & ~onset_target).sum().item())
                    fn_onset += int((~onset_pred & onset_target).sum().item())

                    sustain_target = valid_targets[:, :, pitch_count : pitch_count * 2] >= config.target_roll.sustain_threshold
                    sustain_pred = valid_logits[:, :, pitch_count : pitch_count * 2] >= 0.0
                    tp_sustain += int((sustain_pred & sustain_target).sum().item())
                    fp_sustain += int((sustain_pred & ~sustain_target).sum().item())
                    fn_sustain += int((~sustain_pred & sustain_target).sum().item())

                postfix["onset"] = f"{loss_dict['onset'].item():.4f}"
                postfix["sustain"] = f"{loss_dict['sustain'].item():.4f}"
                if tp_onset + fp_onset + fn_onset > 0:
                    precision_onset = tp_onset / max(1, tp_onset + fp_onset)
                    recall_onset = tp_onset / max(1, tp_onset + fn_onset)
                    f1_onset = 0.0 if precision_onset + recall_onset == 0.0 else 2.0 * precision_onset * recall_onset / (precision_onset + recall_onset)
                    postfix["on_f1"] = f"{f1_onset:.4f}"
                    postfix["on_p"] = f"{precision_onset:.4f}"
                    postfix["on_r"] = f"{recall_onset:.4f}"
                if tp_sustain + fp_sustain + fn_sustain > 0:
                    precision_sustain = tp_sustain / max(1, tp_sustain + fp_sustain)
                    recall_sustain = tp_sustain / max(1, tp_sustain + fn_sustain)
                    f1_sustain = 0.0 if precision_sustain + recall_sustain == 0.0 else 2.0 * precision_sustain * recall_sustain / (precision_sustain + recall_sustain)
                    postfix["su_f1"] = f"{f1_sustain:.4f}"
                    postfix["su_p"] = f"{precision_sustain:.4f}"
                    postfix["su_r"] = f"{recall_sustain:.4f}"
            else:
                postfix["interval"] = f"{loss_dict['interval'].item():.4f}"
                postfix["boundary"] = f"{loss_dict['boundary'].item():.4f}"
            val_progress.set_postfix(**postfix)

    if not losses:
        return {"total": float("nan")}
    
    metrics = {"total": torch.stack(losses).mean().item()}
    if config.autoencoder_model.decoder_mode == "frame":
        if tp_onset + fp_onset + fn_onset > 0:
            precision_onset = tp_onset / max(1, tp_onset + fp_onset)
            recall_onset = tp_onset / max(1, tp_onset + fn_onset)
            metrics["onset_precision"] = precision_onset
            metrics["onset_recall"] = recall_onset
            metrics["onset_f1"] = 0.0 if precision_onset + recall_onset == 0.0 else 2.0 * precision_onset * recall_onset / (precision_onset + recall_onset)
        if tp_sustain + fp_sustain + fn_sustain > 0:
            precision_sustain = tp_sustain / max(1, tp_sustain + fp_sustain)
            recall_sustain = tp_sustain / max(1, tp_sustain + fn_sustain)
            metrics["sustain_precision"] = precision_sustain
            metrics["sustain_recall"] = recall_sustain
            metrics["sustain_f1"] = 0.0 if precision_sustain + recall_sustain == 0.0 else 2.0 * precision_sustain * recall_sustain / (precision_sustain + recall_sustain)
    return metrics


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
        if init_checkpoint is not None and args.resume_from is None:
            load_model_weights(model, init_checkpoint, device, encoder_only=args.init_encoder_only)

        autoencoder_num_workers = (
            config.runtime.autoencoder_num_workers
            if config.runtime.autoencoder_num_workers is not None
            else config.runtime.num_workers
        )

        # 訓練用（データ拡張有効）および検証用のDataset
        train_dataset = SegmentAutoencoderDataset(
            train_pairs,
            config.target_roll,
            decoder_mode=config.autoencoder_model.decoder_mode,
            augment_pitch_shift=True,
            pitch_shift_min_semitones=config.autoencoder_training.pitch_shift_min_semitones,
            pitch_shift_max_semitones=config.autoencoder_training.pitch_shift_max_semitones,
            max_cached_songs=config.runtime.autoencoder_max_cached_songs,
        )
        val_dataset = SegmentAutoencoderDataset(
            val_pairs,
            config.target_roll,
            decoder_mode=config.autoencoder_model.decoder_mode,
            max_cached_songs=config.runtime.autoencoder_max_cached_songs,
        )

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
            num_workers=autoencoder_num_workers,
            persistent_workers=autoencoder_num_workers > 0,
            collate_fn=collate_autoencoder_samples,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=autoencoder_num_workers,
            persistent_workers=autoencoder_num_workers > 0,
            collate_fn=collate_autoencoder_samples,
        )

        optimizer = AdamW(
            model.parameters(),
            lr=config.autoencoder_training.learning_rate,
            weight_decay=config.autoencoder_training.weight_decay,
        )

        # 混合精度用のGradScalerの初期化
        scaler = torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda" and config.autoencoder_training.mixed_precision == "fp16"
        )

        import math
        total_steps = config.autoencoder_training.max_steps
        if total_steps is None:
            steps_per_epoch = math.ceil(len(train_loader) / config.autoencoder_training.grad_accum_steps)
            total_steps = steps_per_epoch * config.autoencoder_training.max_epochs

        scheduler = build_lr_scheduler(optimizer, config.autoencoder_training, total_steps)

        resume_state = ResumeState()
        if args.resume_from is not None:
            resume_state, _ = load_training_checkpoint(
                Path(args.resume_from).expanduser(),
                model,
                optimizer,
                scaler,
                device,
                scheduler=scheduler,
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
        encoder_frozen_state = None

        for epoch in range(resume_state.start_epoch, config.autoencoder_training.max_epochs):
            should_freeze_encoder = epoch < args.freeze_encoder_epochs
            if encoder_frozen_state != should_freeze_encoder:
                for param in model.encoder.parameters():
                    param.requires_grad = not should_freeze_encoder
                encoder_frozen_state = should_freeze_encoder
                if should_freeze_encoder:
                    print(f"[Epoch {epoch}] Encoder is frozen.")
                else:
                    print(f"[Epoch {epoch}] Encoder is unfrozen.")

            model.train()
            running_total = 0.0
            running_interval = 0.0
            running_boundary = 0.0
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
                        interval_targets=batch["interval_targets"],
                        segment_valid_lengths=batch["segment_valid_lengths"],
                        boundary_predictor=getattr(model.decoder, "interval_boundary_predictor", None),
                    )
                    step_loss = loss_dict["total"] / config.autoencoder_training.grad_accum_steps

                # 誤差逆伝播の実行
                if scaler.is_enabled():
                    scaler.scale(step_loss).backward()
                else:
                    step_loss.backward()

                running_total += loss_dict["total"].detach().item()
                running_interval += loss_dict["interval"].detach().item()
                running_boundary += loss_dict["boundary"].detach().item()
                running_onset += loss_dict["onset"].detach().item()
                running_sustain += loss_dict["sustain"].detach().item()
                running_pedal += loss_dict["pedal"].detach().item()
                running_velocity += loss_dict["velocity"].detach().item()
                running_kl += loss_dict["kl"].detach().item()
                batch_count += 1

                postfix = {
                    "total": f"{running_total / batch_count:.4f}",
                    "pedal": f"{running_pedal / batch_count:.4f}",
                    "velocity": f"{running_velocity / batch_count:.4f}",
                    "step": global_step,
                }
                if config.autoencoder_model.decoder_mode == "frame":
                    postfix["onset"] = f"{running_onset / batch_count:.4f}"
                    postfix["sustain"] = f"{running_sustain / batch_count:.4f}"
                else:
                    postfix["interval"] = f"{running_interval / batch_count:.4f}"
                    postfix["boundary"] = f"{running_boundary / batch_count:.4f}"
                train_progress.set_postfix(**postfix)

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
                    
                    if scheduler is not None:
                        scheduler.step()

                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    epoch_step_count += 1

                    # 進捗ロギング
                    if global_step % config.runtime.log_every_steps == 0:
                        log_str = (
                            f"epoch={epoch} step={global_step} "
                            f"train_total={running_total / batch_count:.4f} "
                        )
                        if config.autoencoder_model.decoder_mode == "frame":
                            log_str += (
                                f"train_onset={running_onset / batch_count:.4f} "
                                f"train_sustain={running_sustain / batch_count:.4f} "
                            )
                        else:
                            log_str += (
                                f"train_interval={running_interval / batch_count:.4f} "
                                f"train_boundary={running_boundary / batch_count:.4f} "
                            )
                        log_str += (
                            f"train_pedal={running_pedal / batch_count:.4f} "
                            f"train_velocity={running_velocity / batch_count:.4f} "
                            f"train_kl={running_kl / batch_count:.4f}"
                        )
                        tqdm.write(log_str)

                        metrics = {
                            "train/total": running_total / batch_count,
                            "train/pedal": running_pedal / batch_count,
                            "train/velocity": running_velocity / batch_count,
                            "train/kl": running_kl / batch_count,
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "epoch": epoch,
                        }
                        if config.autoencoder_model.decoder_mode == "frame":
                            metrics["train/onset"] = running_onset / batch_count
                            metrics["train/sustain"] = running_sustain / batch_count
                        else:
                            metrics["train/interval"] = running_interval / batch_count
                            metrics["train/boundary"] = running_boundary / batch_count

                        log_wandb_metrics(wandb_run, metrics, step=global_step)

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
                            scheduler=scheduler,
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
                
                if scheduler is not None:
                    scheduler.step()

                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            # 検証の実行とチェックポイントの更新
            val_metrics = run_validation(model, val_loader, device, config, epoch, args.stage)
            val_loss = val_metrics["total"]
            is_best = val_loss <= best_val_loss
            best_val_loss = min(best_val_loss, val_loss)
            
            wandb_val_metrics = {"epoch": epoch}
            for k, v in val_metrics.items():
                wandb_val_metrics[f"val/{k}"] = v
            log_wandb_metrics(wandb_run, wandb_val_metrics, step=global_step)
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
                scheduler=scheduler,
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
                    scheduler=scheduler,
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
