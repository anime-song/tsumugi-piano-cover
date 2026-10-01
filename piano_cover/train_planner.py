"""学習済みのカバーモデルの原曲エンコーダを固定し、Planner だけを学習し直す。

    python -m piano_cover.train_planner --checkpoint checkpoints/piano_cover_v3r/best.pt --cache-dir data/piano_cover_v3 \
        --out checkpoints/piano_cover_v3r_planner/planner.pt

本体の学習では、窓を 1 つ選ぶたびにそのカバーの曲全体の曲線を正解にするので、Planner は 1 本の曲線を何十回も見て
カバーごとの癖を覚え、検証の損失が 1000〜2000 step で底を打って上がっていく (Planner の勾配は原曲エンコーダにも流れる)。
正解の大半は原曲から予測できない成分 (同じ曲の他のカバーの平均で当てても 0.76、local/probe_planner.py) なので、
ここでは原曲エンコーダの出力 (SongEncoder、Planner の入力) を曲ごとに 1 回だけ計算してキャッシュし、Planner だけを
ドロップアウト付きで学習して、検証の損失が一番よい時点の重みを残す。
--smooth N なら正解を N パッチの移動平均にならしてから学習する (2 秒ごとの細かい揺れではなく、区間ごとの曲線を当てる)。
--views で、原曲を移調したエンコーダの出力も作り、学習で曲ごとにどれかを選ぶ (本体の学習の移調の拡張の代わり)。

出力は Planner の重みだけ。生成では元のチェックポイントと一緒に渡す (piano_cover.generate の --planner)。
検証では、2 秒ごとの損失と、正解と予測の両方を 4 パッチ (8 秒) の移動平均にした損失を測る。
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from piano_ar.config import ModelConfig
from piano_ar.model import PianoARModel
from piano_ar.tokenizer import KIND, KIND_NOTE, ONSET, PianoTokenizer

from .arrangement import ARRANGEMENT_NAMES, arrangement_values, map_to_source
from .config import CoverConfig
from .data import SPLITS, CoverCache, plan_target, source_tensors
from .model import CoverModel, Planner
from .source import SourceVocab, source_features, transpose_rows

COLUMNS = ("loudness", "density", *ARRANGEMENT_NAMES)
EVAL_SMOOTH = 4


def smooth_curve(curve: np.ndarray, width: int) -> np.ndarray:
    """[S, C] の曲線を width パッチの移動平均にする (NaN は除いて平均し、元が NaN の所は NaN のまま)"""
    if width <= 1:
        return curve
    known = ~np.isnan(curve)
    kernel = np.ones(width)
    out = np.full_like(curve, np.nan)
    for j in range(curve.shape[1]):
        total = np.convolve(np.where(known[:, j], curve[:, j], 0.0), kernel, mode="same")
        count = np.convolve(known[:, j].astype(np.float64), kernel, mode="same")
        out[:, j] = np.where(known[:, j] & (count > 0), total / np.maximum(count, 1), np.nan)
    return out


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="原曲エンコーダを使うカバーモデル")
    parser.add_argument("--cache-dir", default="data/piano_cover_v3")
    parser.add_argument("--out", required=True, help="Planner の重みの保存先")
    parser.add_argument("--features", default=None, help="エンコーダの出力のキャッシュ (省略で --out の隣)")
    parser.add_argument("--views", type=int, default=4, help="原曲 1 曲あたりのエンコーダの出力の数 (1 つ目は移調なし)")
    parser.add_argument("--max-transpose", type=int, default=5)
    parser.add_argument("--smooth", type=int, default=4, help="学習の正解を何パッチの移動平均にするか (1 でならさない)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Planner の Transformer のドロップアウト")
    parser.add_argument(
        "--input-dropout", type=float, default=0.3, help="Planner の入力 (エンコーダの出力) のドロップアウト"
    )
    parser.add_argument("--channel-dropout", type=float, default=0.15, help="演奏者を「指定なし」にする確率")
    parser.add_argument("--init", choices=("fresh", "checkpoint"), default="fresh", help="Planner の初期値")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def encode_sources(model: CoverModel, cache: CoverCache, sources: list[int], views: int, max_transpose: int) -> dict:
    """原曲ごとの Planner の入力 (SongEncoder の出力をパッチごとに K 本つないだもの) [views, S, K*D] (fp16, CPU)"""
    tokenizer = model.tokenizer
    vocab = SourceVocab(tokenizer.config)
    device = next(model.parameters()).device
    rng = random.Random(0)
    out = {}
    for n, source in enumerate(sources):
        rows = cache.source_rows(source)
        end = int(cache.source_end_frames[source])
        shifts = [0] + [
            rng.choice([s for s in range(-max_transpose, max_transpose + 1) if s]) for _ in range(views - 1)
        ]
        encoded = []
        for shift in shifts:
            shifted = transpose_rows(rows, shift) if shift else rows
            features = source_features(shifted, end, tokenizer, vocab, model.config.max_source_rows)
            batch = {key: value[None].to(device) for key, value in source_tensors(features).items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                memory = model.encode_source(batch, torch.zeros_like(batch["src_patch_valid"]))
            K = model.config.source_latents
            encoded.append(memory.song.reshape(1, -1, K * memory.song.shape[-1])[0].half().cpu())
        out[source] = torch.stack(encoded)
        if (n + 1) % 100 == 0:
            print(f"原曲 {n + 1}/{len(sources)} をエンコードした", flush=True)
    return out


def cover_targets(cache: CoverCache, tokenizer: PianoTokenizer, cover: int, num_patches: int) -> np.ndarray:
    """カバーの Planner の正解 [S, 5] (CoverWindowDataset と同じ。伸縮なし)"""
    source = int(cache.source_index[cover])
    events = cache.cover_events(cover)
    align = cache.cover_align(cover)
    rows = cache.source_rows(source)
    F_ = tokenizer.patch_frames
    target = plan_target(events, align, cache.align_step, 1.0, num_patches, F_)
    notes = events[events[:, KIND] == KIND_NOTE]
    note_times = map_to_source(notes[:, ONSET].astype(np.float64), align, cache.align_step)
    values = arrangement_values(events, note_times, rows, np.arange(num_patches + 1) * F_, tokenizer.config.frame_rate)
    return np.concatenate([target, values.astype(np.float32)], axis=1)


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """列ごとの二乗誤差の和と数 (NaN の正解は除く)"""
    known = ~target.isnan()
    err = ((pred - target.nan_to_num()) ** 2) * known
    return err.sum(dim=(0, 1)), known.sum(dim=(0, 1))


def main() -> None:
    args = build_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features_path = Path(args.features) if args.features else out_path.with_name("planner_features.pt")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    cache = CoverCache(args.cache_dir)
    tokenizer = PianoTokenizer(cache.tokenizer_config)
    model_config = ModelConfig.from_dict(checkpoint["model_config"])
    cover_config = CoverConfig.from_dict(checkpoint["cover_config"])
    if not cover_config.planner_dim:
        raise SystemExit(f"{args.checkpoint} には Planner がない")
    model = CoverModel(model_config, cover_config, tokenizer, checkpoint["source_vocab_size"]).to(device).eval()
    model.load_state_dict(checkpoint["model"])

    covers = {name: np.flatnonzero(cache.split == SPLITS[name]) for name in ("train", "val")}
    sources = sorted({int(cache.source_index[c]) for c in np.concatenate(list(covers.values()))})
    if features_path.exists():
        stored = torch.load(features_path, weights_only=False)
        if stored["checkpoint"] != args.checkpoint or stored["views"] < args.views:
            raise SystemExit(f"{features_path} は別の設定で作ったキャッシュ。消すか --features で別の場所を指定する")
        encoded = stored["encoded"]
        print(f"{features_path} からエンコーダの出力を読んだ")
    else:
        began = time.time()
        encoded = encode_sources(model, cache, sources, args.views, args.max_transpose)
        torch.save({"checkpoint": args.checkpoint, "views": args.views, "encoded": encoded}, features_path)
        print(
            f"原曲 {len(sources)} 曲 x {args.views} をエンコードして {features_path} に保存 ({time.time() - began:.0f} 秒)"
        )

    # 演奏者の埋め込みはデコーダのものをそのまま使う (固定)
    channel_weight = model.decoder.channel_embedding.weight.detach().float().to(device)
    K = cover_config.source_latents
    song_dim = K * (cover_config.source_dim or model_config.dim)
    head_dim = model_config.dim // model_config.heads
    planner_config = replace(
        model_config, dim=cover_config.planner_dim, heads=cover_config.planner_dim // head_dim, dropout=args.dropout
    )
    planner = Planner(
        song_dim, model_config.dim, planner_config, cover_config.planner_layers, model_config.dynamics_columns
    ).to(device)
    if args.init == "checkpoint":
        planner.load_state_dict(model.planner.state_dict())
    else:
        planner.apply(PianoARModel._init_weights)
    del model
    torch.cuda.empty_cache()

    def load_split(name: str) -> list[dict]:
        items = []
        for cover in covers[name]:
            source = int(cache.source_index[cover])
            raw = cover_targets(cache, tokenizer, int(cover), encoded[source].shape[1])
            if np.isnan(raw).all():
                continue
            items.append(
                {"source": source, "channel": int(cache.channels[cover]), "raw": raw,
                 "train": smooth_curve(raw, args.smooth), "eval": smooth_curve(raw, EVAL_SMOOTH)}
            )  # fmt: skip
        return items

    train_items, val_items = load_split("train"), load_split("val")
    print(f"学習 カバー {len(train_items)} 本 / 検証 {len(val_items)} 本 / 正解は {args.smooth} パッチの移動平均")

    def make_batch(items: list[dict], key: str, train: bool) -> tuple[torch.Tensor, ...]:
        S = max(encoded[item["source"]].shape[1] for item in items)
        song = torch.zeros(len(items), S, song_dim, dtype=torch.float16)
        target = torch.full((len(items), S, len(COLUMNS)), float("nan"))
        valid = torch.zeros(len(items), S, dtype=torch.bool)
        channel = torch.zeros(len(items), dtype=torch.long)
        for i, item in enumerate(items):
            views = encoded[item["source"]]
            view = random.randrange(args.views) if train else 0
            length = views.shape[1]
            song[i, :length] = views[view]
            target[i, :length] = torch.from_numpy(item[key])
            valid[i, :length] = True
            dropped = train and random.random() < args.channel_dropout
            channel[i] = 0 if dropped else item["channel"]
        return song.to(device).float(), target.to(device), valid.to(device), channel_weight[channel.to(device)]

    def predict(song: torch.Tensor, valid: torch.Tensor, channel: torch.Tensor) -> torch.Tensor:
        song = F.dropout(song, args.input_dropout, planner.training)
        return planner(song, channel, valid)

    @torch.no_grad()
    def evaluate() -> dict[str, float]:
        planner.eval()
        sums = {key: torch.zeros(len(COLUMNS), device=device) for key in ("raw", "eval")}
        counts = {key: torch.zeros(len(COLUMNS), device=device) for key in ("raw", "eval")}
        for start in range(0, len(val_items), args.batch_size):
            items = val_items[start : start + args.batch_size]
            song, raw, valid, channel = make_batch(items, "raw", train=False)
            pred = predict(song, valid, channel)
            s, n = masked_mse(pred, raw)
            sums["raw"] += s
            counts["raw"] += n
            # 8 秒の比較は予測もならす (正解と同じ移動平均)
            for i, item in enumerate(items):
                length = int(valid[i].sum())
                smoothed = smooth_curve(pred[i, :length].cpu().numpy().astype(np.float64), EVAL_SMOOTH)
                target = torch.from_numpy(item["eval"]).to(device)
                s, n = masked_mse(torch.from_numpy(smoothed).to(device)[None], target[None])
                sums["eval"] += s
                counts["eval"] += n
        planner.train()
        out = {}
        for key, label in (("raw", "2s"), ("eval", "8s")):
            per = sums[key] / counts[key].clamp_min(1)
            out[label] = float(sums[key].sum() / counts[key].sum().clamp_min(1))
            out.update({f"{label}_{c}": float(v) for c, v in zip(COLUMNS, per)})
        return out

    decay = [p for n, p in planner.named_parameters() if p.dim() >= 2 and "norm" not in n]
    no_decay = [p for n, p in planner.named_parameters() if not (p.dim() >= 2 and "norm" not in n)]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    initial = evaluate()
    print(f"[val] step 0 2s {initial['2s']:.4f} 8s {initial['8s']:.4f}")
    best, best_state, best_step = initial["8s"], None, 0
    planner.train()
    running, began = 0.0, time.time()
    for step in range(1, args.steps + 1):
        if step <= args.warmup_steps:
            lr = args.lr * step / args.warmup_steps
        else:
            progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
            lr = args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        song, target, valid, channel = make_batch(random.sample(train_items, args.batch_size), "train", train=True)
        s, n = masked_mse(predict(song, valid, channel), target)
        loss = s.sum() / n.sum().clamp_min(1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(planner.parameters(), 1.0)
        optimizer.step()
        running += float(loss)
        if step % args.val_every == 0:
            val = evaluate()
            mark = ""
            if val["8s"] < best:
                best, best_step, mark = val["8s"], step, " *"
                best_state = {k: v.detach().cpu().clone() for k, v in planner.state_dict().items()}
            print(
                f"[val] step {step} train {running / args.val_every:.4f} | 2s {val['2s']:.4f} 8s {val['8s']:.4f} ("
                + " ".join(f"{c} {val[f'8s_{c}']:.3f}" for c in COLUMNS)
                + f") {time.time() - began:.0f}s{mark}",
                flush=True,
            )
            running = 0.0

    if best_state is None:
        raise SystemExit("検証の損失が初期値より良くならなかったので保存しない")
    torch.save(
        {"planner": best_state, "checkpoint": args.checkpoint, "step": best_step, "val": best, "args": vars(args)},
        out_path,
    )
    print(f"step {best_step} の Planner (検証 8s {best:.4f}) を {out_path} に保存")


if __name__ == "__main__":
    main()
