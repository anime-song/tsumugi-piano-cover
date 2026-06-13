from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from tqdm.auto import tqdm
import math

from src_v2.config import ExperimentConfig, TrainingConfig
from src_v2.data.index import PairEntry, build_pair_index, split_pairs_by_song


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def get_autocast_context(device: torch.device, mixed_precision: str):
    # 混合精度（bf16/fp16/none）のautocastコンテキストを取得
    if device.type != "cuda":
        return nullcontext()
    if mixed_precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if mixed_precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    # バッチ内のTensorをデバイスへ転送
    moved: dict[str, object] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def clear_cuda_cache(device: torch.device, epoch: int) -> None:
    # CUDAキャッシュのクリア
    if device.type != "cuda":
        return
    torch.cuda.empty_cache()
    tqdm.write(f"cleared cuda cache after epoch={epoch}")


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    total_steps: int,
) -> LRScheduler | None:
    # 学習率スケジューラーの構築
    if config.lr_scheduler_type == "none":
        return None
        
    if config.lr_scheduler_type == "cosine_with_warmup":
        warmup_steps = config.lr_warmup_steps
        lr_min_ratio = config.lr_min / config.learning_rate if config.learning_rate > 0 else 0.0

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(1.0, max(0.0, progress))
            return lr_min_ratio + 0.5 * (1.0 - lr_min_ratio) * (1.0 + math.cos(math.pi * progress))
            
        return LambdaLR(optimizer, lr_lambda)
        
    raise ValueError(f"unsupported lr_scheduler_type: {config.lr_scheduler_type}")


@dataclass
class ResumeState:
    # 学習再開用の状態を保持するクラス
    start_epoch: int = 0
    start_batch_index: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    checkpoint_path: str | None = None


def _coerce_rng_state(state: object) -> torch.Tensor:
    # 乱数（RNG）状態を適切なCPUテンソルに変換
    if isinstance(state, torch.Tensor):
        tensor = state.detach().cpu()
    else:
        tensor = torch.as_tensor(state)
    if tensor.dtype != torch.uint8:
        tensor = tensor.to(dtype=torch.uint8)
    return tensor.contiguous()


def _coerce_cuda_rng_state_all(states: object) -> list[torch.Tensor]:
    # すべてのGPUの乱数状態をテンソルリストに変換
    if isinstance(states, torch.Tensor):
        return [_coerce_rng_state(states)]
    if isinstance(states, (list, tuple)):
        return [_coerce_rng_state(state) for state in states]
    raise TypeError(f"unsupported cuda rng state type: {type(states)!r}")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: ExperimentConfig,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    resume_epoch: int,
    resume_batch_index: int,
    checkpoint_name: str,
    extra_state: dict[str, object] | None = None,
    scheduler: LRScheduler | None = None,
) -> None:
    # チェックポイントの保存（重み、最適化状態、乱数シードなど）
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
        "experiment_config": asdict(config),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "resume_epoch": resume_epoch,
        "resume_batch_index": resume_batch_index,
        "cpu_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "lr_scheduler_state": scheduler.state_dict() if scheduler is not None else None,
    }
    if extra_state:
        payload.update(extra_state)
    torch.save(payload, path)
    best_text = "inf" if best_val_loss == float("inf") else f"{best_val_loss:.4f}"
    tqdm.write(
        f"saved checkpoint [{checkpoint_name}] path={path} epoch={epoch} step={global_step} "
        f"resume_epoch={resume_epoch} resume_batch={resume_batch_index} best_val_loss={best_text}"
    )


def load_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    scheduler: LRScheduler | None = None,
) -> tuple[ResumeState, dict[str, object]]:
    # チェックポイントの読み込み
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    scaler_state = checkpoint.get("scaler_state")
    if scaler.is_enabled() and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    lr_scheduler_state = checkpoint.get("lr_scheduler_state")
    if scheduler is not None and lr_scheduler_state is not None:
        scheduler.load_state_dict(lr_scheduler_state)

    # 乱数状態の復元
    cpu_rng_state = checkpoint.get("cpu_rng_state")
    if cpu_rng_state is not None:
        torch.random.set_rng_state(_coerce_rng_state(cpu_rng_state))
    cuda_rng_state_all = checkpoint.get("cuda_rng_state_all")
    if device.type == "cuda" and cuda_rng_state_all is not None:
        torch.cuda.set_rng_state_all(_coerce_cuda_rng_state_all(cuda_rng_state_all))

    saved_epoch = int(checkpoint.get("epoch", 0))
    if "resume_epoch" in checkpoint and "resume_batch_index" in checkpoint:
        start_epoch = int(checkpoint["resume_epoch"])
        start_batch_index = int(checkpoint["resume_batch_index"])
    else:
        start_epoch = saved_epoch
        start_batch_index = 0
        tqdm.write("checkpoint does not include resume position metadata; restarting saved epoch from batch 0.")

    global_step = int(checkpoint.get("global_step", 0))
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    best_text = "inf" if best_val_loss == float("inf") else f"{best_val_loss:.4f}"
    tqdm.write(
        f"loaded checkpoint path={path} saved_epoch={saved_epoch} resume_epoch={start_epoch} "
        f"resume_batch={start_batch_index} step={global_step} best_val_loss={best_text}"
    )

    # 予約キー以外の追加状態を取得
    reserved = {
        "model_state",
        "optimizer_state",
        "scaler_state",
        "epoch",
        "global_step",
        "best_val_loss",
        "resume_epoch",
        "resume_batch_index",
        "cpu_rng_state",
        "cuda_rng_state_all",
        "lr_scheduler_state",
    }
    extra_state = {key: value for key, value in checkpoint.items() if key not in reserved}
    return (
        ResumeState(
            start_epoch=start_epoch,
            start_batch_index=start_batch_index,
            global_step=global_step,
            best_val_loss=best_val_loss,
            checkpoint_path=str(path),
        ),
        extra_state,
    )


def build_splits(
    config: ExperimentConfig,
    limit_train_pairs: int | None = None,
    limit_val_pairs: int | None = None,
) -> tuple[list[PairEntry], list[PairEntry], list[PairEntry]]:
    # データセットを構築し、曲単位で train/val/test に分割
    pairs = build_pair_index(
        dataset_json_path=config.dataset.dataset_json,
        original_midi_dir=config.dataset.original_midi_dir,
        piano_midi_dir=config.dataset.piano_midi_dir,
        piano_to_performer_json=config.dataset.piano_to_performer_json,
    )
    splits = split_pairs_by_song(
        entries=pairs,
        train_fraction=config.dataset.train_fraction,
        val_fraction=config.dataset.val_fraction,
        test_fraction=config.dataset.test_fraction,
        seed=config.seed,
    )
    train_pairs = splits["train"]
    val_pairs = splits["val"]
    test_pairs = splits["test"]

    if limit_train_pairs is not None:
        train_pairs = train_pairs[:limit_train_pairs]
    if limit_val_pairs is not None:
        val_pairs = val_pairs[:limit_val_pairs]
    return train_pairs, val_pairs, test_pairs


def skip_to_batch(loader, start_batch_index: int):
    # 指定したバッチインデックスまでスキップ
    return enumerate(islice(loader, start_batch_index, None), start=start_batch_index)
