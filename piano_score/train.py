"""楽譜の事前学習 (楽譜の言語モデル)。piano_ar のモデルを、1 小節 = 1 パッチの楽譜トークンで学習する。

    python -m piano_score.train --wandb-project piano-score

途中から再開する場合は --resume checkpoints/piano_score/latest.pt (wandb も同じ run に続けて記録する)。
検証の損失がそれまでで最も低くなったら out-dir/best.pt にも保存する。
この段階では演奏がないので、MTIME はテンポ記号どおりの時刻に倍率と揺れをかけたものを使う (data.py)。
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from piano_ar.config import ModelConfig
from piano_ar.data import collate
from piano_ar.model import PianoARModel
from piano_ar.train import limit_gpu_memory, lr_at, make_optimizer, to_device

from .data import ScoreAugmentConfig, ScoreCache, ScoreWindowDataset, make_sampler
from .tokenizer import TOKEN_GROUPS, ScoreTokenizer


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", default="data/piano_score/pretraining")
    parser.add_argument("--out-dir", default="checkpoints/piano_score")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--window-measures", type=int, default=32, help="1 つの窓の小節数 (Global のパッチ数)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100_000, help="optimizer の更新回数")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--song-start-prob", type=float, default=0.1, help="窓を曲の冒頭から切り出す確率")
    parser.add_argument("--no-augment", action="store_true", help="移調とテンポの変化をかけない")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument(
        "--no-grad-checkpoint", action="store_true", help="gradient checkpointing を切る (速いがメモリを大きく使う)"
    )
    parser.add_argument("--length-buckets", type=int, default=8, help="パッチを長さ順に何個の塊に分けて処理するか")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Transformer を torch.compile する (triton-windows と、日本語版 Windows では PYTHONUTF8=1 が必要)",
    )
    parser.add_argument(
        "--gpu-memory-limit",
        type=float,
        default=0.9,
        help="起動時の空き VRAM のうち使ってよい割合。超えたらメモリ不足エラーにする (0 で無制限)",
    )
    parser.add_argument("--device", default="auto", help="auto / cuda / cpu (動作確認を CPU で回すときなど)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    # モデルの大きさ (ModelConfig の各項目を --dim などで上書きできる)。楽譜ではチャンネルとスタイル参照は使わない
    for field in fields(ModelConfig):
        if field.name not in ("num_channels", "style_tokens"):
            parser.add_argument(f"--{field.name.replace('_', '-')}", type=type(field.default), default=field.default)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: PianoARModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(to_device(batch, device))
        tokens = int(output["tokens"])
        for key in ("loss", *[f"loss_{g}" for g in TOKEN_GROUPS]):
            sums[key] = sums.get(key, 0.0) + float(output[key]) * tokens
        count += tokens
    model.train()
    return {key: value / max(count, 1) for key, value in sums.items()}


def format_losses(values: dict[str, float], prefix: str = "") -> str:
    return " ".join(f"{g} {values[f'{prefix}loss_{g}']:.3f}" for g in TOKEN_GROUPS)


def main() -> None:
    args = build_args()
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        limit_gpu_memory(args.gpu_memory_limit)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = ScoreCache(args.cache_dir)
    tokenizer = ScoreTokenizer(cache.tokenizer_config)
    model_config = ModelConfig(
        **{
            f.name: getattr(args, f.name) for f in fields(ModelConfig) if f.name not in ("num_channels", "style_tokens")
        },
        num_channels=1,
        style_tokens=0,
    )
    train_set = ScoreWindowDataset(
        cache,
        split="train",
        window_measures=args.window_measures,
        augment=None if args.no_augment else ScoreAugmentConfig(),
        song_start_prob=args.song_start_prob,
    )
    val_set = ScoreWindowDataset(cache, split="val", window_measures=args.window_measures)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=make_sampler(train_set, args.steps * args.grad_accum * args.batch_size),
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate)
    print(
        f"学習 {len(train_set)} 曲 / 検証 {len(val_set)} 曲 / 窓 {args.window_measures} 小節 / 語彙 {tokenizer.vocab_size}"
    )

    model = PianoARModel(model_config, tokenizer).to(device)
    model.set_gradient_checkpointing(not args.no_grad_checkpoint)
    print(f"パラメータ数 {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    optimizer = make_optimizer(model, args)
    step = 0
    best_val_loss = float("inf")
    checkpoint: dict = {}
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        print(f"{args.resume} の step {step} から再開 (これまでの最良の検証損失 {best_val_loss:.4f})")
    if args.compile:
        # inductor は内部のテンプレートを既定の文字コードで読むため、日本語版 Windows (cp932) では失敗する
        if not sys.flags.utf8_mode:
            raise SystemExit(
                "--compile には Python の UTF-8 モードが必要です。PYTHONUTF8=1 を設定するか python -X utf8 で起動してください"
            )
        model.compile_blocks()

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_id = checkpoint.get("wandb_id")
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=wandb_id,
            resume="allow" if wandb_id else None,
            config={**vars(args), "model": asdict(model_config), "tokenizer": asdict(cache.tokenizer_config)},
        )
        wandb_run.summary["parameters_M"] = sum(p.numel() for p in model.parameters()) / 1e6

    def save(path: Path) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "model_config": asdict(model_config),
                "tokenizer_config": asdict(cache.tokenizer_config),
                "args": vars(args),
                "wandb_id": wandb_run.id if wandb_run else None,
                "best_val_loss": best_val_loss,
            },
            path,
        )

    loader_iter = iter(train_loader)
    model.train()
    running: dict[str, float] = {}
    running_tokens = 0
    last_log = time.time()
    while step < args.steps:
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, args)
        for _ in range(args.grad_accum):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                batch = next(loader_iter)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(to_device(batch, device), args.length_buckets)
            (output["loss"] / args.grad_accum).backward()
            tokens = int(output["tokens"])
            running_tokens += tokens
            for key in ("loss", *[f"loss_{g}" for g in TOKEN_GROUPS]):
                running[key] = running.get(key, 0.0) + float(output[key].detach()) * tokens
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_every == 0:
            elapsed = time.time() - last_log
            logs = {f"train/{k}": v / running_tokens for k, v in running.items()}
            logs.update({"train/lr": lr_at(step, args), "train/grad_norm": float(grad_norm)})
            logs["train/tokens_per_sec"] = running_tokens / elapsed
            print(
                f"step {step} loss {logs['train/loss']:.4f} {format_losses(logs, 'train/')}"
                f" lr {logs['train/lr']:.2e} {logs['train/tokens_per_sec']:.0f} tok/s"
            )
            if wandb_run:
                wandb_run.log(logs, step=step)
            running, running_tokens, last_log = {}, 0, time.time()

        if args.val_every and step % args.val_every == 0 and len(val_set):
            val = evaluate(model, val_loader, device)
            print(f"[val] step {step} loss {val['loss']:.4f} {format_losses(val)}")
            if wandb_run:
                wandb_run.log({f"val/{k}": v for k, v in val.items()}, step=step)
            if val["loss"] < best_val_loss:
                best_val_loss = val["loss"]
                save(out_dir / "best.pt")
                print(f"[val] 最良を更新したので best.pt に保存 (step {step})")
                if wandb_run:
                    wandb_run.summary.update({"best/val_loss": best_val_loss, "best/step": step})

        if step % args.save_every == 0 or step == args.steps:
            save(out_dir / "latest.pt")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
