"""演奏 -> 楽譜モデルの 2 段目: 楽譜から合成した演奏との組で学習する。デコーダは楽譜の事前学習モデルから始める。

    python -m piano_perf2score.train --wandb-project piano-perf2score

楽譜のキャッシュは piano_score.prepare --visible-only で作ったもの (見えない音符と cue サイズの音符を除いたもの)。
合成演奏は DataLoader のワーカーの中で窓ごとに描き出す (piano_perf2score.render)。
途中から再開する場合は --resume checkpoints/piano_perf2score/latest.pt。検証の損失が最良になったら best.pt にも保存する。
モデルに部品を足したときは --init-from で重みだけを読み (足した部品は初期値のまま)、step と wandb の run を引き継ぐ。
optimizer の状態は作り直すので、--rewarmup-steps の間は学習率を 0 から上げ直す。

3 段目 (実演奏) は、2 段目の重みから --init-from と --new-run (step 0・新しい wandb の run) で始める。

    python -m piano_perf2score.train --init-from checkpoints/piano_perf2score/best.pt --new-run \
        --real-cache-dir data/piano_score/asap --out-dir checkpoints/piano_perf2score_asap ...

--real-cache-dir の組 (asap.py) と合成演奏を --real-ratio の割合で混ぜる (実演奏の楽譜は 200 曲ほどしかないので、
それだけで学習すると楽譜を覚えてしまう)。検証は両方で測り、best.pt は実演奏の検証の損失で選ぶ。
--ref-noise-prob で小節の基準のずれ (data.RefNoise) を入れる。
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from piano_ar.config import ModelConfig
from piano_ar.train import limit_gpu_memory, lr_at, to_device
from piano_score.data import ScoreAugmentConfig, ScoreCache, sampler_weights
from piano_score.tokenizer import TOKEN_GROUPS, ScoreTokenizer

from .data import PairCache, RealWindowDataset, RefNoise, SynthWindowDataset, collate
from .model import Perf2ScoreConfig, Perf2ScoreModel


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", default="data/piano_score/synth")
    parser.add_argument("--pretrained", default="checkpoints/piano_score_pretrain/step98000.pt")
    parser.add_argument("--out-dir", default="checkpoints/piano_perf2score")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-from", default=None, help="重みだけを読んで続ける (足した部品は初期値。optimizer は作り直す)")
    parser.add_argument("--rewarmup-steps", type=int, default=500, help="--init-from のあとに学習率を上げ直すステップ数")
    parser.add_argument(
        "--new-run", action="store_true", help="--init-from の重みから step 0・新しい wandb の run として始める (3 段目)"
    )
    parser.add_argument("--real-cache-dir", default=None, help="実演奏と楽譜の組のキャッシュ (asap.py)")
    parser.add_argument("--real-ratio", type=float, default=0.5, help="学習の窓のうち実演奏から取る割合")
    parser.add_argument("--real-val-songs", type=int, default=0, help="実演奏の検証に使う組の数 (0 なら全部)")
    parser.add_argument("--ref-noise-prob", type=float, default=0.0, help="小節の基準をずらす小節の割合 (data.RefNoise)")
    parser.add_argument("--ref-noise-sigma", type=float, default=0.15, help="基準のずれ (前の小節の長さに対する標準偏差)")
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
    group_counts: dict[str, float] = {}
    count = 0
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(to_device(batch, device))
        tokens = int(output["tokens"])
        for key in ("loss", "acc", *[f"loss_{g}" for g in TOKEN_GROUPS]):
            sums[key] = sums.get(key, 0.0) + float(output[key]) * tokens
        # 種類ごとの正解率はその種類のトークンの数で重みづける (その種類がないバッチを 0 として数えない)
        for g in TOKEN_GROUPS:
            n = float(output[f"count_{g}"])
            sums[f"acc_{g}"] = sums.get(f"acc_{g}", 0.0) + float(output[f"acc_{g}"]) * n
            group_counts[g] = group_counts.get(g, 0.0) + n
        count += tokens
    model.train()
    result = {key: value / max(count, 1) for key, value in sums.items()}
    result.update({f"acc_{g}": sums[f"acc_{g}"] / max(group_counts[g], 1) for g in TOKEN_GROUPS})
    return result


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


def format_accuracy(values: dict[str, float]) -> str:
    return " ".join(f"{g} {values[f'acc_{g}']:.3f}" for g in TOKEN_GROUPS)


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

    noise = RefNoise(prob=args.ref_noise_prob, sigma=args.ref_noise_sigma) if args.ref_noise_prob > 0 else None
    common = {"window_measures": args.window_measures, "context_measures": args.context_measures, "ref_noise": noise}
    augment = None if args.no_augment else ScoreAugmentConfig()
    train_set = SynthWindowDataset(cache, split="train", augment=augment, song_start_prob=args.song_start_prob, **common)
    val_set = SynthWindowDataset(cache, split="val", **common)
    val_set.songs = val_set.songs[: args.val_songs]
    weights = sampler_weights(train_set, args.rare_meter_weight)
    datasets: list = [train_set]
    real_val_set = None
    if args.real_cache_dir:
        real_cache = PairCache(args.real_cache_dir)
        real_train = RealWindowDataset(
            real_cache, split="train", augment=augment, song_start_prob=args.song_start_prob, **common
        )
        real_val_set = RealWindowDataset(real_cache, split="val", **common)
        if args.real_val_songs:
            real_val_set.songs = real_val_set.songs[: args.real_val_songs]
        # 合成演奏と実演奏の重みの合計を 1 - real_ratio : real_ratio にする (それぞれの中は小節数に比例)
        real_weights = sampler_weights(real_train)
        weights = np.concatenate(
            [weights / weights.sum() * (1 - args.real_ratio), real_weights / real_weights.sum() * args.real_ratio]
        )
        datasets.append(real_train)
        print(f"実演奏 {len(real_train)} 組 (検証 {len(real_val_set)}) を {args.real_ratio:.0%} の割合で混ぜる")
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights), num_samples=args.steps * args.grad_accum * args.batch_size, replacement=True
    )
    train_loader = DataLoader(
        ConcatDataset(datasets),
        batch_size=args.batch_size,
        sampler=sampler,
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
    real_val_loader = None
    if real_val_set is not None:
        real_val_loader = DataLoader(
            real_val_set,
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
        added = sorted({".".join(name.split(".")[:2]) for name in missing})
        print(f"{args.init_from} の step {checkpoint['step']} の重みから続ける (初期値のままの部品: {added})")
        if args.new_run:
            checkpoint = {}  # step・最良の損失・wandb の run は引き継がない
        else:
            step = restart_step = checkpoint["step"]
            best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
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

    def validate() -> float:
        """合成演奏と実演奏の検証の損失を出して、最良の判定に使う損失 (実演奏があればそちら) を返す"""
        val = evaluate(model, val_loader, device)
        print(f"[val] step {step} loss {val['loss']:.4f} {format_losses(val)}")
        print(f"[val] step {step} acc {val['acc']:.4f} {format_accuracy(val)}")
        logs = {f"val/{k}": v for k, v in val.items()}
        main = val["loss"]
        if real_val_loader is not None:
            real = evaluate(model, real_val_loader, device)
            print(f"[val_real] step {step} loss {real['loss']:.4f} {format_losses(real)}")
            print(f"[val_real] step {step} acc {real['acc']:.4f} {format_accuracy(real)}")
            logs.update({f"val_real/{k}": v for k, v in real.items()})
            main = real["loss"]
        if wandb_run:
            wandb_run.log(logs, step=step)
        return main

    if step == 0 and len(val_set):
        # 2 段目の最初は演奏を入れる前 (ゲート 0) の損失。ここからどれだけ下がるかが、演奏をどれだけ使えているか
        validate()

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
            val_loss = validate()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save(out_dir / "best.pt")
                print(f"[val] 最良を更新したので best.pt に保存 (step {step})")

        if step % args.save_every == 0 or step == args.steps:
            save(out_dir / "latest.pt")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
