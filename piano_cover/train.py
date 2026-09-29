"""ピアノカバーの学習。事前学習したピアノ生成モデルに原曲エンコーダと cross-attention を足して学習する。

    python -m piano_cover.train --wandb-project piano-cover

--pretrained の重みでデコーダを初期化し、新しく足した部分は --lr、デコーダは --lr x --decoder-lr-scale で学習する。
--freeze-decoder ならデコーダは固定して、新しく足した部分だけを学習する (Flamingo と同じ)。
--init-cover なら学習済みのカバーモデルから始めて、あとから足した部分 (--onset-head-dim の OnsetHead など) を足す。
--freeze-loaded ならそのとき読んだ重みは固定して、足した部分だけを学習する。
--pretrain-mix の割合で事前学習の曲を原曲なしで混ぜ、ピアノの生成の力を忘れないようにする。
検証では原曲ありとなしの両方で損失を測り、その差 (val/source_gain) で原曲がどれだけ効いているかを見る。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from piano_ar.config import ModelConfig
from piano_ar.data import AugmentConfig, PianoWindowDataset, PretrainingCache
from piano_ar.evaluation import piano_roll_image, sample_stats, synthesize
from piano_ar.tokenizer import TOKEN_GROUPS, PianoTokenizer
from piano_ar.train import limit_gpu_memory, lr_at, to_device

from .config import CoverConfig
from .data import CoverCache, CoverWindowDataset, collate_cover, make_cover_sampler, source_tensors
from .model import CoverModel, SourceCondition
from .source import SourceVocab, source_features


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", default="data/piano_cover")
    parser.add_argument("--pretrained", default="checkpoints/piano_ar/best.pt")
    parser.add_argument("--out-dir", default="checkpoints/piano_cover")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--init-cover",
        default=None,
        help="学習済みのカバーモデルで初期化する (--pretrained の代わり)。設定はそのチェックポイントのものを使い、"
        "--onset-head-dim だけ引数で変えられる",
    )
    parser.add_argument(
        "--freeze-loaded", action="store_true", help="--init-cover で読んだ重みを固定して、足した部分だけを学習する"
    )
    parser.add_argument("--pretrain-cache", default="data/piano_ar/pretraining")
    parser.add_argument("--pretrain-mix", type=float, default=0.2, help="事前学習の曲を混ぜる割合 (0 で混ぜない)")
    parser.add_argument("--window-seconds", type=float, default=64.0)
    # 大きいデコーダ (v3, 384M) では VRAM 9.3GB に収まるのは batch 4 x 累積 4 まで (累積中は勾配も残るため)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20_000, help="optimizer の更新回数")
    parser.add_argument("--lr", type=float, default=3e-4, help="新しく足した部分の学習率")
    parser.add_argument("--decoder-lr-scale", type=float, default=0.3, help="事前学習済みのデコーダの学習率の倍率")
    parser.add_argument(
        "--freeze-decoder",
        action="store_true",
        help="事前学習済みのデコーダを固定する。カバーのデータへの過学習と、ピアノの生成の力の崩れを防ぐ。"
        "デコーダの勾配と optimizer の状態が要らない分、メモリも減る",
    )
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--song-start-prob", type=float, default=0.1)
    parser.add_argument("--channel-dropout", type=float, default=0.15)
    parser.add_argument("--source-dropout", type=float, default=0.1, help="原曲を外す確率 (CFG 用)")
    parser.add_argument("--structure-dropout", type=float, default=0.2, help="拍・コード・キーをそれぞれ外す確率")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--no-grad-checkpoint", action="store_true")
    parser.add_argument("--length-buckets", type=int, default=8)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Transformer のブロックを torch.compile する (triton-windows と、日本語版 Windows では PYTHONUTF8=1 が必要)",
    )
    parser.add_argument("--gpu-memory-limit", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--sample-every", type=int, default=2000, help="0 で生成評価をしない")
    parser.add_argument("--sample-count", type=int, default=2)
    parser.add_argument("--sample-seconds", type=float, default=30.0)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--sample-top-p", type=float, default=0.95)
    for field in fields(CoverConfig):
        name = f"--{field.name.replace('_', '-')}"
        if isinstance(field.default, bool):
            parser.add_argument(name, action="store_true")
        else:
            parser.add_argument(name, type=type(field.default), default=field.default)
    return parser.parse_args()


def make_optimizer(model: CoverModel, args: argparse.Namespace) -> torch.optim.Optimizer:
    new = {id(p) for p in model.new_parameters()}
    groups: dict[tuple[bool, bool], list[torch.nn.Parameter]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
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
def evaluate(model: CoverModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
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


def sample_evaluation(
    model: CoverModel,
    tokenizer: PianoTokenizer,
    val_set: CoverWindowDataset,
    args: argparse.Namespace,
    step: int,
    out_dir: Path,
    wandb_run,
) -> None:
    """検証の原曲 (毎回同じ曲) の冒頭からカバーを生成する。比べられるよう、同じ原曲の本物のカバーの冒頭も残す"""
    cache = val_set.cache
    c = tokenizer.config
    num_patches = math.ceil(args.sample_seconds / c.patch_seconds)
    covers, seen = [], set()
    for cover in val_set.covers.tolist():
        if int(cache.source_index[cover]) not in seen:
            seen.add(int(cache.source_index[cover]))
            covers.append(cover)
        if len(covers) == args.sample_count:
            break

    sample_dir = out_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    logs: dict[str, object] = {}
    stats = []
    if wandb_run:
        import wandb
    model.eval()
    started = time.time()
    device = next(model.parameters()).device
    for i, cover in enumerate(covers):
        source = int(cache.source_index[cover])
        features = source_features(
            cache.source_rows(source),
            int(cache.source_end_frames[source]),
            tokenizer,
            val_set.vocab,
            model.config.max_source_rows,
        )
        source_items = {key: value.to(device) for key, value in source_tensors(features).items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            condition = SourceCondition(model, source_items)
            patches = model.decoder.generate(
                tokenizer,
                num_patches=num_patches,
                channel=int(cache.channels[cover]),
                temperature=args.sample_temperature,
                top_p=args.sample_top_p,
                context_patches=round(args.window_seconds / c.patch_seconds),
                condition=condition,
            )[0]
        events = tokenizer.patches_to_events(patches)
        stats.append(sample_stats(events, c.frame_rate))
        # wandb に動画 ID が残らないよう、キャッシュ内の番号で名前を付ける
        name = f"cover{i}_source{source}"
        path = sample_dir / f"step{step:07d}_{name}.mid"
        tokenizer.events_to_midi(events, path)
        reference_path = sample_dir / f"reference_{name}.mid"
        reference = cache.cover_events(cover)
        reference = reference[reference[:, 0] < num_patches * tokenizer.patch_frames]
        if not reference_path.exists():
            tokenizer.events_to_midi(reference, reference_path)
        if wandb_run:
            wandb_run.save(str(path), base_path=str(out_dir), policy="now")
            logs[f"samples/{name}_audio"] = wandb.Audio(synthesize(events, c.frame_rate), sample_rate=16000)
            logs[f"samples/{name}_roll"] = wandb.Image(piano_roll_image(events, tokenizer))
            if step == args.sample_every:
                logs[f"samples/{name}_reference_audio"] = wandb.Audio(
                    synthesize(reference, c.frame_rate), sample_rate=16000
                )
    model.train()

    mean_stats = {key: float(np.mean([s.get(key, 0.0) for s in stats])) for key in stats[0]} if stats else {}
    logs.update({f"sample_stats/{k}": v for k, v in mean_stats.items()})
    print(
        f"[sample] step {step} {time.time() - started:.0f}s "
        + " ".join(f"{k} {v:.2f}" for k, v in mean_stats.items())
        + f" -> {sample_dir}"
    )
    if wandb_run:
        wandb_run.log(logs, step=step)


def main() -> None:
    args = build_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    limit_gpu_memory(args.gpu_memory_limit)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = CoverCache(args.cache_dir)
    tokenizer = PianoTokenizer(cache.tokenizer_config)
    vocab = SourceVocab(cache.tokenizer_config)
    cover_config = CoverConfig(**{f.name: getattr(args, f.name) for f in fields(CoverConfig)})
    if args.init_cover:
        pretrained = torch.load(args.init_cover, map_location="cpu", weights_only=False, mmap=True)
        cover_config = replace(CoverConfig(**pretrained["cover_config"]), onset_head_dim=args.onset_head_dim)
    else:
        pretrained = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    if pretrained["tokenizer_config"] != asdict(cache.tokenizer_config):
        raise SystemExit("事前学習とカバーのキャッシュでトークナイザーの設定が違います")
    model_config = ModelConfig.from_dict(pretrained["model_config"])
    window_patches = round(args.window_seconds / cache.tokenizer_config.patch_seconds)

    pretraining = None
    if args.pretrain_mix > 0:
        pretraining = PianoWindowDataset(
            PretrainingCache(args.pretrain_cache),
            split="train",
            window_patches=window_patches,
            augment=None if args.no_augment else AugmentConfig(),
            song_start_prob=args.song_start_prob,
            channel_dropout=args.channel_dropout,
        )
    common = {"window_patches": window_patches, "cover_config": cover_config}
    train_set = CoverWindowDataset(
        cache,
        split="train",
        augment=None if args.no_augment else AugmentConfig(),
        song_start_prob=args.song_start_prob,
        channel_dropout=args.channel_dropout,
        source_dropout=args.source_dropout,
        structure_dropout=args.structure_dropout,
        pretraining=pretraining,
        **common,
    )
    val_set = CoverWindowDataset(cache, split="val", source_dropout=0.0, **common)
    val_nosource = CoverWindowDataset(cache, split="val", source_dropout=1.0, **common)
    micro_steps = args.steps * args.grad_accum
    loader_options = {"num_workers": args.num_workers, "collate_fn": collate_cover}
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=make_cover_sampler(train_set, micro_steps * args.batch_size, args.pretrain_mix),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
        **loader_options,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, **loader_options)
    val_nosource_loader = DataLoader(val_nosource, batch_size=args.batch_size, **loader_options)
    print(
        f"学習 カバー {len(train_set.covers)} 本 (+ 事前学習 {len(pretraining) if pretraining else 0} 曲を"
        f" {args.pretrain_mix:.0%}) / 検証 {len(val_set)} 本 / 窓 {window_patches} パッチ"
    )

    model = CoverModel(model_config, cover_config, tokenizer, vocab.size).to(device)
    model.set_gradient_checkpointing(not args.no_grad_checkpoint)
    if args.freeze_decoder:
        model.decoder.requires_grad_(False)
    if args.init_cover:
        missing, unexpected = model.load_state_dict(pretrained["model"], strict=False)
        if unexpected:
            raise SystemExit(f"{args.init_cover} にこのモデルにない重みがあります: {unexpected[:5]}")
        if args.freeze_loaded:
            loaded = set(pretrained["model"])
            for name, param in model.named_parameters():
                if name in loaded:
                    param.requires_grad_(False)
        if not args.resume:
            print(f"{args.init_cover} (step {pretrained['step']}) で初期化 (足した重み {len(missing)} 個)")
    optimizer = make_optimizer(model, args)
    step = 0
    best_val_loss = float("inf")
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        print(f"{args.resume} の step {step} から再開 (これまでの最良の検証損失 {best_val_loss:.4f})")
    elif not args.init_cover:
        model.decoder.load_state_dict(pretrained["model"])
        print(f"{args.pretrained} (step {pretrained['step']}) でデコーダを初期化")
    if args.compile:
        # inductor は内部のテンプレートを既定の文字コードで読むため、日本語版 Windows (cp932) では失敗する
        if not sys.flags.utf8_mode:
            raise SystemExit(
                "--compile には Python の UTF-8 モードが必要です。PYTHONUTF8=1 を設定するか python -X utf8 で起動してください"
            )
        model.compile_blocks()
    new_count = sum(p.numel() for p in model.new_parameters())
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(
        f"パラメータ数 {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M (新規 {new_count / 1e6:.1f}M"
        f" / 学習する {sum(p.numel() for p in trainable) / 1e6:.1f}M)"
    )

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_id = checkpoint.get("wandb_id") if checkpoint else None
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=wandb_id,
            resume="allow" if wandb_id else None,
            config={**vars(args), "model": asdict(model_config), "cover": asdict(cover_config)},
        )

    def save(path: Path) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "model_config": asdict(model_config),
                "cover_config": asdict(cover_config),
                "tokenizer_config": asdict(cache.tokenizer_config),
                "source_vocab_size": vocab.size,
                "channel_index": pretrained.get("channel_index"),
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
        scale = lr_at(step, args) / args.lr
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
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_every == 0:
            elapsed = time.time() - last_log
            logs = {f"train/{k}": v / running_tokens for k, v in running.items()}
            logs.update({"train/lr": args.lr * scale, "train/grad_norm": float(grad_norm)})
            logs["train/tokens_per_sec"] = running_tokens / elapsed
            # cross-attention のゲートがどれだけ開いたか (0 なら原曲をまったく使っていない)
            gates = [torch.tanh(b.attn_gate).abs().item() for b in (*model.global_cross, *model.local_cross)]
            logs["train/cross_gate_mean"] = float(np.mean(gates))
            if device.type == "cuda":
                logs["train/max_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
            print(
                f"step {step} loss {logs['train/loss']:.4f} "
                + " ".join(f"{g} {logs[f'train/loss_{g}']:.3f}" for g in TOKEN_GROUPS)
                + f" gate {logs['train/cross_gate_mean']:.3f} lr {logs['train/lr']:.2e}"
                + f" mem {logs.get('train/max_memory_gb', 0):.1f}GB"
                + f" {logs['train/tokens_per_sec']:.0f} tok/s"
            )
            if wandb_run:
                wandb_run.log(logs, step=step)
            running, running_tokens, last_log = {}, 0, time.time()

        if args.val_every and step % args.val_every == 0 and len(val_set):
            val = evaluate(model, val_loader, device)
            val_nosource_loss = evaluate(model, val_nosource_loader, device)["loss"]
            print(
                f"[val] step {step} loss {val['loss']:.4f} (原曲なし {val_nosource_loss:.4f}) "
                + " ".join(f"{g} {val[f'loss_{g}']:.3f}" for g in TOKEN_GROUPS)
            )
            if wandb_run:
                logs = {f"val/{k}": v for k, v in val.items()}
                logs.update(
                    {"val/loss_nosource": val_nosource_loss, "val/source_gain": val_nosource_loss - val["loss"]}
                )
                wandb_run.log(logs, step=step)
            if val["loss"] < best_val_loss:
                best_val_loss = val["loss"]
                save(out_dir / "best.pt")
                print(f"[val] 最良を更新したので best.pt に保存 (step {step})")
                if wandb_run:
                    wandb_run.summary.update({"best/val_loss": best_val_loss, "best/step": step})

        if args.sample_every and step % args.sample_every == 0 and len(val_set):
            sample_evaluation(model, tokenizer, val_set, args, step, out_dir, wandb_run)

        if step % args.save_every == 0 or step == args.steps:
            save(out_dir / "latest.pt")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(line_buffering=True)
    main()
