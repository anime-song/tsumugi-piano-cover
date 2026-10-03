"""演奏 -> 楽譜モデルの 2 段目: 楽譜から合成した演奏との組で学習する。デコーダは楽譜の事前学習モデルから始める。

    python -m piano_perf2score.train --wandb-project piano-perf2score

楽譜のキャッシュは piano_score.prepare --visible-only で作ったもの (見えない音符と cue サイズの音符を除いたもの)。
合成演奏は DataLoader のワーカーの中で窓ごとに描き出す (piano_perf2score.render)。
途中から再開する場合は --resume checkpoints/piano_perf2score/latest.pt。検証の損失が最良になったら best.pt にも保存する。
モデルに部品を足したときは --init-from で重みだけを読み (足した部品は初期値のまま)、step と wandb の run を引き継ぐ。
optimizer の状態は作り直すので、--rewarmup-steps の間は学習率を 0 から上げ直す。
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
from piano_ar.train import limit_gpu_memory, lr_at, to_device
from piano_score.data import ScoreAugmentConfig, ScoreCache, make_sampler
from piano_score.tokenizer import TOKEN_GROUPS, ScoreTokenizer

from .data import SynthWindowDataset, collate
from .model import Perf2ScoreConfig, Perf2ScoreModel


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", default="data/piano_score/synth")
    parser.add_argument("--pretrained", default="checkpoints/piano_score_pretrain/step98000.pt")
    parser.add_argument("--out-dir", default="checkpoints/piano_perf2score")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-from", default=None, help="重みだけを読んで続ける (足した部品は初期値。optimizer は作り直す)")
    parser.add_argument("--rewarmup-steps", type=int, default=500, help="--init-from のあとに学習率を上げ直すステップ数")
    parser.add_argument("--window-measures", type=int, default=32)
    parser.add_argument("--context-measures", type=int, default=2, help="窓の前に演奏を描き出す小節数")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--steps", type=int, default=60_000)
    parser.add_argument("--lr", type=float, default=3e-4, help="新しく足した部分の学習率")
    parser.add_argument("--decoder-lr-scale", type=float, default=0.3, help="事前学習済みのデコーダの学習率の倍率")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--song-start-prob", type=float, default=0.2)
    parser.add_argument("--rare-meter-weight", type=float, default=2.5)
    parser.add_argument("--no-augment", action="store_true", help="移調をかけない")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=2000)
    parser.add_argument("--val-songs", type=int, default=400, help="検証に使う曲数 (検証曲の先頭から)")
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--no-grad-checkpoint", action="store_true")
    parser.add_argument("--length-buckets", type=int, default=8)
    parser.add_argument(
        "--compile", action="store_true", help="ブロックごとに torch.compile する (PYTHONUTF8=1 が必要)"
    )
    parser.add_argument("--gpu-memory-limit", type=float, default=0.9)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    for field in fields(Perf2ScoreConfig):
        parser.add_argument(f"--{field.name.replace('_', '-')}", type=type(field.default), default=field.default)
    return parser.parse_args()


def make_optimizer(model: Perf2ScoreModel, args: argparse.Namespace) -> torch.optim.Optimizer:
    new = {id(p) for p in model.new_parameters()}
    groups: dict[tuple[bool, bool], list[torch.nn.Parameter]] = {}
    for name, param in model.named_parameters():
        decay = param.dim() >= 2 and "norm" not in name
        groups.setdefault((id(param) in new, decay), []).append(param)
    return torch.optim.AdamW(
        [
            {
                "params": params,
                "weight_decay": args.weight_decay if decay else 0.0,
                "lr_scale": 1.0 if is_new else args.decoder_lr_scale,
            }
            for (is_new, decay), params in groups.items()
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        fused=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate(model: Perf2ScoreModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
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


def gate_values(model: Perf2ScoreModel) -> dict[str, float]:
    """cross-attention のゲート (tanh) の平均の絶対値。0 なら演奏をまったく使っていない"""
    logs = {}
    for name, blocks in (("global", model.global_cross), ("local", model.local_cross)):
        attn = [abs(float(torch.tanh(b.attn_gate.detach()))) for b in blocks]
        logs[f"gate/{name}_attn"] = sum(attn) / max(len(attn), 1)
    if model.onset_head is not None:
        # OnsetHead の query の大きさ (0 なら打鍵の量をまったく足していない)
        logs["gate/onset_query"] = float(model.onset_head.query.weight.detach().norm())
    return logs


def format_losses(values: dict[str, float], prefix: str = "") -> str:
    return " ".join(f"{g} {values[f'{prefix}loss_{g}']:.3f}" for g in TOKEN_GROUPS)


def main() -> None:
    args = build_args()
    torch.manual_seed(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        limit_gpu_memory(args.gpu_memory_limit)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = ScoreCache(args.cache_dir)
    if not cache.meta.get("visible_only"):
        print("注意: --visible-only で作っていないキャッシュ (見えない音符を含む楽譜) を使う")
    tokenizer = ScoreTokenizer(cache.tokenizer_config)
    pretrained = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    model_config = ModelConfig.from_dict(pretrained["model_config"])
    config = Perf2ScoreConfig(**{f.name: getattr(args, f.name) for f in fields(Perf2ScoreConfig)})
    model = Perf2ScoreModel(model_config, config, tokenizer)
    model.decoder.load_state_dict(pretrained["model"])
    print(f"デコーダを {args.pretrained} (step {pretrained['step']}) から読んだ")
    model.to(device)
    model.set_gradient_checkpointing(not args.no_grad_checkpoint)
    new_count = sum(p.numel() for p in model.new_parameters())
    print(
        f"パラメータ数 {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M (新しく足した分 {new_count / 1e6:.1f}M)"
    )

    train_set = SynthWindowDataset(
        cache,
        split="train",
        window_measures=args.window_measures,
        context_measures=args.context_measures,
        augment=None if args.no_augment else ScoreAugmentConfig(),
        song_start_prob=args.song_start_prob,
    )
    val_set = SynthWindowDataset(
        cache,
        split="val",
        window_measures=args.window_measures,
        context_measures=args.context_measures,
        songs=None,
    )
    val_set.songs = val_set.songs[: args.val_songs]
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=make_sampler(train_set, args.steps * args.grad_accum * args.batch_size, args.rare_meter_weight),
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=min(args.num_workers, 4),
        collate_fn=collate,
        persistent_workers=args.num_workers > 0,
    )
    print(f"学習 {len(train_set)} 曲 / 検証 {len(val_set)} 曲 / 窓 {args.window_measures} 小節")

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
        print(f"{args.resume} の step {step} から再開")
    restart_step = None
    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        if unexpected:
            raise SystemExit(f"チェックポイントにあってモデルにない重み: {unexpected}")
        step = restart_step = checkpoint["step"]
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        added = sorted({".".join(name.split(".")[:2]) for name in missing})
        print(f"{args.init_from} の step {step} の重みから続ける (初期値のままの部品: {added})")
    if args.compile:
        if not sys.flags.utf8_mode:
            raise SystemExit("--compile には PYTHONUTF8=1 か python -X utf8 が必要です")
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
            config={**vars(args), "model": asdict(model_config), "perf2score": asdict(config)},
        )

    def save(path: Path) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "model_config": asdict(model_config),
                "perf2score_config": asdict(config),
                "tokenizer_config": asdict(cache.tokenizer_config),
                "token_counts": pretrained.get("token_counts"),
                "args": vars(args),
                "wandb_id": wandb_run.id if wandb_run else None,
                "best_val_loss": best_val_loss,
            },
            path,
        )

    if step == 0 and len(val_set):
        # 演奏を入れる前 (ゲート 0) の損失。ここからどれだけ下がるかが、演奏をどれだけ使えているか
        val = evaluate(model, val_loader, device)
        print(f"[val] step 0 loss {val['loss']:.4f} {format_losses(val)}")
        if wandb_run:
            wandb_run.log({f"val/{k}": v for k, v in val.items()}, step=0)

    loader_iter = iter(train_loader)
    model.train()
    running: dict[str, float] = {}
    running_tokens = 0
    last_log = time.time()
    while step < args.steps:
        scale = lr_at(step, args) / args.lr
        if restart_step is not None:
            scale *= min(1.0, (step - restart_step + 1) / max(args.rewarmup_steps, 1))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * scale * group["lr_scale"]
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
            logs.update({"train/lr": args.lr * scale, "train/grad_norm": float(grad_norm)})
            logs["train/tokens_per_sec"] = running_tokens / elapsed
            logs.update(gate_values(model))
            print(
                f"step {step} loss {logs['train/loss']:.4f} {format_losses(logs, 'train/')}"
                f" gate {logs['gate/global_attn']:.3f}/{logs['gate/local_attn']:.3f}"
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

        if step % args.save_every == 0 or step == args.steps:
            save(out_dir / "latest.pt")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
