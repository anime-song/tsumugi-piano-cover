"""楽譜の事前学習 (楽譜の言語モデル)。piano_ar のモデルを、1 小節 = 1 パッチの楽譜トークンで学習する。

    python -m piano_score.train --wandb-project piano-score

途中から再開する場合は --resume checkpoints/piano_score/latest.pt (wandb も同じ run に続けて記録する)。
検証の損失がそれまでで最も低くなったら out-dir/best.pt にも保存する。
この段階では演奏がないので、MTIME はテンポ記号どおりの時刻に倍率と揺れをかけたものを使う (data.py)。
--sample-every ごとに楽譜を生成して out-dir/samples に MusicXML で保存する。MuseScore があれば PNG にもして wandb に送る。
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from piano_ar.config import ModelConfig
from piano_ar.data import collate
from piano_ar.model import PianoARModel
from piano_ar.train import limit_gpu_memory, lr_at, make_optimizer, to_device

from .data import ScoreAugmentConfig, ScoreCache, ScoreWindowDataset, make_sampler, rare_meter_songs
from .generate import find_musescore, generate, render_png, score_stats, unseen_tokens
from .musicxml import write_musicxml
from .tokenizer import PAD, TOKEN_GROUPS, ScoreTokenizer


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
    parser.add_argument(
        "--rare-meter-weight", type=float, default=2.5, help="珍しい拍子か変拍子を含む曲の選ばれやすさの倍率"
    )
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
    # 学習中の生成評価
    parser.add_argument("--sample-every", type=int, default=5000, help="0 で生成評価をしない")
    parser.add_argument("--sample-count", type=int, default=2, help="冒頭からの生成と続きの生成、それぞれの本数")
    parser.add_argument("--sample-measures", type=int, default=16)
    parser.add_argument("--prompt-measures", type=int, default=4, help="続きの生成で検証曲の冒頭から与える小節数")
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--sample-top-p", type=float, default=0.95)
    parser.add_argument(
        "--musescore", default="auto", help="PNG にする MuseScore の実行ファイル (auto で探す、none で使わない)"
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    # モデルの大きさ (ModelConfig の各項目を --dim などで上書きできる)。楽譜ではチャンネルは使わない
    for field in fields(ModelConfig):
        if field.name != "num_channels":
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


# 各小節の MTIME の後に必ずこの順で並ぶ小節ヘッダのトークン
HEADER_FIELDS = ("ts", "key", "clef1", "clef2")


@torch.no_grad()
def header_change_stats(model: PianoARModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """小節ヘッダ (拍子・調・音部記号) を、前の小節から変わる所と同じ所に分けて測る。

    平均の損失では、ほとんどを占める「前の小節と同じ」所に隠れて、変わる所を学べているかが見えないため。
    change_ratio は「変わる」確率 (1 - 前の小節と同じトークンの確率) の合計 / 実際に変わった回数。
    1 を大きく下回っていくなら、前の小節を写す方へ偏っている。
    ヘッダの予測に要るのは Local の先頭の数トークンだけなので、Local は小節全体ではなくそこまで通す。
    compile のグラフを増やさないよう、通常の実行で通す (forward の記憶と同じ理由)。
    """
    model.eval()
    H = len(HEADER_FIELDS)
    # 変わる所の損失・正解数・回数・「変わる」確率の合計、同じ所の損失・回数
    sums = torch.zeros(6, H, dtype=torch.float64, device=device)
    for batch in loader:
        batch = to_device(batch, device)
        tokens, valid = batch["tokens"], batch["patch_valid"]
        patches = tokens[valid]
        patches = patches[:, : int((patches != PAD).sum(-1).max())]
        with (
            torch.compiler.set_stance("force_eager"),
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"),
        ):
            summarized = model.summarize_patches(patches)
            summaries = summarized.new_zeros(*valid.shape, summarized.shape[-1])
            summaries[valid] = summarized
            context = model.global_forward(summaries, batch["song_start"], batch["pedal_state"], batch["channel"])
            # local_forward の出力の k 番目がトークン k の予測。ヘッダはトークン 1..H
            logits = model.local_forward(patches[:, : 1 + H], context[valid])[:, 1 : 1 + H]
        log_probs = logits.new_zeros(*valid.shape, H, logits.shape[-1], dtype=torch.float32)
        log_probs[valid] = logits.float().log_softmax(-1)

        target = tokens[:, :, 1 : 1 + H]
        current, previous, log_probs = target[:, 1:], target[:, :-1], log_probs[:, 1:]
        pair = (valid[:, 1:] & valid[:, :-1])[..., None]
        changed = (current != previous) & pair
        same = (current == previous) & pair
        loss = -log_probs.gather(-1, current[..., None])[..., 0]
        hit = log_probs.argmax(-1) == current
        p_change = 1 - log_probs.gather(-1, previous[..., None])[..., 0].exp()
        for row, (value, mask) in enumerate(
            [(loss, changed), (hit, changed), (changed, changed), (p_change, pair), (loss, same), (same, same)]
        ):
            sums[row] += (value.double() * mask).sum((0, 1))
    model.train()

    stats: dict[str, float] = {}
    for h, field in enumerate(HEADER_FIELDS):
        change_loss, change_hit, changes, expected, same_loss, sames = sums[:, h].tolist()
        stats[f"header_{field}/loss_change"] = change_loss / max(changes, 1)
        stats[f"header_{field}/acc_change"] = change_hit / max(changes, 1)
        stats[f"header_{field}/change_ratio"] = expected / max(changes, 1)
        stats[f"header_{field}/loss_same"] = same_loss / max(sames, 1)
    return stats


def sample_evaluation(
    model: PianoARModel,
    tokenizer: ScoreTokenizer,
    val_set: ScoreWindowDataset,
    args: argparse.Namespace,
    step: int,
    out_dir: Path,
    wandb_run,
) -> None:
    """冒頭から生成した楽譜と、検証曲の冒頭をプロンプトにして続きを生成した楽譜を残す。
    学習データに一度も出てこないトークンは出さない (val_set.banned)。

    続きの生成のプロンプトは毎回同じ検証曲を使うので、学習が進むにつれて同じ入力への応答がどう変わるかを比べられる。
    """
    cache = val_set.cache
    indices = [i for i, s in enumerate(val_set.songs.tolist()) if cache.num_measures(s) >= args.sample_measures]
    references = []
    for index in indices[: args.sample_count]:
        item = val_set[index]
        rows = item["tokens"][item["patch_valid"]].tolist()
        references.append([[t for t in row if t != PAD] for row in rows][: args.sample_measures])
    prompts = [reference[: args.prompt_measures] for reference in references]

    model.eval()
    started = time.time()
    common = {
        "num_measures": args.sample_measures,
        "temperature": args.sample_temperature,
        "top_p": args.sample_top_p,
        "context_measures": args.window_measures,
    }
    device = next(model.parameters()).device
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        common["banned"] = val_set.banned
        free = generate(model, tokenizer, num_samples=args.sample_count, **common)
        continued = generate(model, tokenizer, num_samples=len(prompts), prompts=prompts, **common) if prompts else []
    model.train()

    sample_dir = out_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    musescore = find_musescore() if args.musescore == "auto" else (None if args.musescore == "none" else args.musescore)
    logs: dict[str, object] = {}
    if wandb_run:
        import wandb

    def add(name: str, patches: list[list[int]], log_name: str | None = None) -> None:
        path = sample_dir / f"{name}.musicxml"
        write_musicxml(tokenizer.decode(patches)[0], path)
        pages = render_png(path, musescore) if musescore else []
        if wandb_run:
            wandb_run.save(str(path), base_path=str(out_dir), policy="now")
            if pages:
                logs[f"samples/{log_name or name}"] = wandb.Image(str(pages[0]))

    stats = []
    for i, patches in enumerate(free):
        add(f"step{step:07d}_free{i}", patches, f"free{i}")
        stats.append(score_stats(patches, tokenizer))
    for i, patches in enumerate(continued):
        add(f"step{step:07d}_continuation{i}", patches, f"continuation{i}")
        if not (sample_dir / f"reference{i}.musicxml").exists():  # 比べる用の元の曲 (最初の評価のときだけ)
            add(f"reference{i}", references[i])

    mean_stats = {key: float(np.mean([s.get(key, 0.0) for s in stats])) for key in stats[0]} if stats else {}
    mean_stats["ended_rate"] = float(np.mean([len(p) < args.sample_measures for p in free])) if free else 0.0
    logs.update({f"sample_stats/{k}": v for k, v in mean_stats.items()})
    print(
        f"[sample] step {step} {time.time() - started:.0f}s "
        + " ".join(f"{k} {v:.2f}" for k, v in mean_stats.items())
        + f" -> {sample_dir}"
    )
    if wandb_run:
        wandb_run.log(logs, step=step)


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
        **{f.name: getattr(args, f.name) for f in fields(ModelConfig) if f.name != "num_channels"}, num_channels=1
    )
    train_set = ScoreWindowDataset(
        cache,
        split="train",
        window_measures=args.window_measures,
        augment=None if args.no_augment else ScoreAugmentConfig(),
        song_start_prob=args.song_start_prob,
    )
    val_set = ScoreWindowDataset(cache, split="val", window_measures=args.window_measures)
    token_counts = cache.token_counts(tokenizer.vocab_size)
    val_set.banned = unseen_tokens(token_counts, tokenizer)
    # 珍しい拍子か変拍子を含む検証曲だけの損失も測る (弱点が改善しているかを見る)
    rare = rare_meter_songs(cache)
    val_rare = ScoreWindowDataset(
        cache, split="val", window_measures=args.window_measures, songs=val_set.songs[rare[val_set.songs]]
    )
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
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate)
    val_rare_loader = DataLoader(val_rare, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate)
    print(
        f"学習 {len(train_set)} 曲 / 検証 {len(val_set)} 曲 (珍しい拍子・変拍子 {len(val_rare)} 曲)"
        f" / 窓 {args.window_measures} 小節 / 語彙 {tokenizer.vocab_size}"
        f" / 学習データに出てこないトークン {int(val_set.banned.sum())}"
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
                # 生成で学習データに出てこないトークンを出さないようにするため
                "token_counts": token_counts,
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
            if len(val_rare):
                val["loss_rare_meter"] = evaluate(model, val_rare_loader, device)["loss"]
            val.update(header_change_stats(model, val_loader, device))
            print(
                f"[val] step {step} loss {val['loss']:.4f} {format_losses(val)}"
                + (f" (珍しい拍子・変拍子 {val['loss_rare_meter']:.4f})" if "loss_rare_meter" in val else "")
            )
            print(
                "[val] ヘッダが変わる所 (損失 / 正解率 / 見積もり÷実際) "
                + " ".join(
                    f"{f} {val[f'header_{f}/loss_change']:.2f}/{val[f'header_{f}/acc_change']:.0%}"
                    f"/{val[f'header_{f}/change_ratio']:.2f}"
                    for f in HEADER_FIELDS
                )
            )
            if wandb_run:
                wandb_run.log({f"val/{k}": v for k, v in val.items()}, step=step)
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
    main()
