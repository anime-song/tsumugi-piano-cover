"""モデル自身の生成から原曲に戻る学習 (on-policy resync) のための、生成済みのパッチ (ロールアウト) を作る・読む。

    python -m piano_cover.rollout --checkpoint checkpoints/piano_cover_v2t/best.pt --cache-dir data/piano_cover_v3

ずらし学習 (DriftConfig) で作るずれは「数秒かけてゆっくり滑る」ものだけだが、実際の生成では、8 分音符を 1 つ取り違える・
onset を飛ばす・密な所で別の onset に吸われる・曲の冒頭で原曲を無視して弾き出す、のような失敗が起きる。
学習の窓の一部をモデル自身が生成したパッチに差し替え、その後の正解のパッチで損失を取れば、自分の失敗から原曲に戻ることを
学べる。学習の中で毎回生成すると遅いので、少し前のモデルで先に作って保存し、学習では確率で差し込む
(piano_cover.data.CoverWindowDataset の rollouts)。

ロールアウトは 2 種類:
    start  曲の冒頭の窓で、最初の 1〜--max-start-patches パッチを生成したもの (冒頭の崩れから戻る)
    mid    曲の途中の窓で、損失を取る窓の中のどこかから 1〜--max-mid-patches パッチを生成したもの
生成は本番と同じ設定 (原曲の cfg 1.75・temperature 1・top-p 0.95、EOS は出さない)。onset-bias はかけない
(失敗が多く出るほうが練習になる)。窓は伸縮・移調なしで、強弱などの条件は正解の値を使う。

出力 (--out、npz): 窓 (カバー・損失を取る窓の始まり・記憶の頭)・差し替える位置・パッチのトークン・パッチの頭のペダル。
学習用と検証用 (--split val) で別に作る。
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

KINDS = ("start", "mid")


@dataclass
class Rollout:
    cover: int
    first: int  # 損失を取る窓の始まり (カバーのフレーム)
    start: int  # 窓 (記憶の頭) の始まり
    split: int  # 差し替える最初のパッチ (窓の中の番号)
    kind: int  # KINDS の番号
    patches: list[list[int]]
    pedal: list[bool]  # 差し替える各パッチの頭のペダル


class RolloutBank:
    """保存したロールアウトの一覧。カバーごとに引ける"""

    def __init__(self, rollouts: list[Rollout]) -> None:
        self.rollouts = rollouts
        self.by_cover: dict[int, list[int]] = {}
        for i, r in enumerate(rollouts):
            self.by_cover.setdefault(r.cover, []).append(i)

    def __len__(self) -> int:
        return len(self.rollouts)

    def __getitem__(self, index: int) -> Rollout:
        return self.rollouts[index]

    def pick(self, cover: int) -> Rollout | None:
        indices = self.by_cover.get(cover)
        return self.rollouts[random.choice(indices)] if indices else None

    def subset(self, kind: str) -> RolloutBank:
        return RolloutBank([r for r in self.rollouts if r.kind == KINDS.index(kind)])

    def save(self, path: Path) -> None:
        patches = [p for r in self.rollouts for p in r.patches]
        np.savez(
            path,
            cover=np.array([r.cover for r in self.rollouts], dtype=np.int64),
            first=np.array([r.first for r in self.rollouts], dtype=np.int64),
            start=np.array([r.start for r in self.rollouts], dtype=np.int64),
            split=np.array([r.split for r in self.rollouts], dtype=np.int64),
            kind=np.array([r.kind for r in self.rollouts], dtype=np.int64),
            count=np.array([len(r.patches) for r in self.rollouts], dtype=np.int64),
            tokens=np.concatenate([np.asarray(p, dtype=np.int16) for p in patches]),
            token_offsets=np.concatenate([[0], np.cumsum([len(p) for p in patches])]).astype(np.int64),
            pedal=np.array([q for r in self.rollouts for q in r.pedal], dtype=bool),
        )

    @classmethod
    def load(cls, path: str | Path) -> RolloutBank:
        z = np.load(path)
        tokens, offsets, pedal = z["tokens"].astype(np.int64), z["token_offsets"], z["pedal"]
        rollouts, patch = [], 0
        for i in range(len(z["cover"])):
            count = int(z["count"][i])
            patches = [tokens[offsets[patch + j] : offsets[patch + j + 1]].tolist() for j in range(count)]
            rollouts.append(
                Rollout(
                    cover=int(z["cover"][i]),
                    first=int(z["first"][i]),
                    start=int(z["start"][i]),
                    split=int(z["split"][i]),
                    kind=int(z["kind"][i]),
                    patches=patches,
                    pedal=[bool(q) for q in pedal[patch : patch + count]],
                )
            )
            patch += count
        return cls(rollouts)


@torch.no_grad()
def rollout_batch(
    model,
    batch: dict[str, torch.Tensor],
    split: torch.Tensor,
    count: int,
    *,
    temperature: float = 1.0,
    top_p: float = 0.95,
    source_cfg: float = 1.75,
    summary_chunk: int = 256,
    sampler=None,
) -> tuple[list[list[list[int]]], list[list[bool]]]:
    """batch (collate_cover したもの、窓は伸縮なし) の各行で、パッチ split[b] から count パッチを生成する。

    学習の forward と同じ窓・同じ原曲の対応 (align) のまま、それより前は正解のパッチを文脈にして 1 パッチずつ生成する
    (原曲なしの行を並べて原曲の cfg をかける)。返り値は行ごとの (生成したパッチのトークン, 各パッチの頭のペダル)。
    sampler (piano_ar.model.LocalSampler、条件は model) を続けて渡すと、取り込んだ CUDA Graph を batch をまたいで使い回す。
    """
    from piano_ar.model import LocalSampler
    from piano_ar.tokenizer import PAD

    from .model import _TrainingCondition

    decoder, tokenizer = model.decoder, model.tokenizer
    B = batch["tokens"].shape[0]
    both = {key: torch.cat([value, value]) for key, value in batch.items()}
    both["has_source"] = torch.cat([batch["has_source"], torch.zeros_like(batch["has_source"])])
    split = torch.cat([split, split]).to(both["tokens"].device)
    rows = torch.arange(2 * B, device=split.device)
    valid = both["patch_valid"]

    # CoverModel.forward の前半と同じ: 生成するパッチの近くの原曲だけ行を作る
    F_ = model.patch_frames
    S = both["src_patch_valid"].shape[1]
    align = both["align"][valid]
    song_of = valid.nonzero()[:, 0]
    centers = torch.div(align.mean(-1), F_, rounding_mode="floor").long()
    r = model.config.local_cross_radius
    neighbors = centers[:, None] + torch.arange(-r, r + 1, device=centers.device)
    position = torch.arange(valid.shape[1], device=valid.device)[None]
    target = valid & (position >= split[:, None]) & (position < split[:, None] + count)
    inside = (neighbors >= 0) & (neighbors < S) & both["has_source"][song_of][:, None] & target[valid][:, None]
    needed = torch.zeros_like(both["src_patch_valid"])
    needed[song_of[:, None].expand_as(neighbors)[inside], neighbors[inside]] = True
    needed &= both["src_patch_valid"]
    # 原曲なしの行 (後半) は原曲を見ないので、エンコードは前半だけにして並べる (見ないようにするのは has_source)
    memory = model.encode_source(batch, needed[:B])
    twice = {name: torch.cat([getattr(memory, name)] * 2) for name in ("song", "song_pos", "song_valid", "lookup")}
    memory = replace(memory, **twice)
    condition = _TrainingCondition(model, memory, both, centers, song_of)

    # 正解のパッチの要約 (前半と後半で同じ)。Global は因果なので、生成するパッチより前 (split より前) の分だけ要る。
    # その先は生成したパッチで上書きする
    before = batch["patch_valid"] & (position < split[:B, None])
    flat = batch["tokens"][before]
    half = torch.zeros(*before.shape, decoder.config.dim, device=flat.device)
    if len(flat):
        flat = flat[:, : int((flat != PAD).sum(-1).max())]
        parts = [decoder.summarize_patches(flat[i : i + summary_chunk]) for i in range(0, len(flat), summary_chunk)]
        half[before] = torch.cat(parts).to(half.dtype)
    summaries = torch.cat([half, half])
    flat_index = (valid.flatten().cumsum(0) - 1).reshape(valid.shape)
    pedal_state = both["pedal_state"].clone()

    if sampler is None:
        sampler = LocalSampler(decoder, tokenizer, model)

    def guide(logits: torch.Tensor) -> torch.Tensor:
        return logits[B:] + source_cfg * (logits[:B] - logits[B:])

    out_tokens: list[list[list[int]]] = [[] for _ in range(B)]
    out_pedal: list[list[bool]] = [[] for _ in range(B)]
    for j in range(count):
        p = split + j
        context = decoder.global_forward(
            summaries,
            both["song_start"],
            pedal_state,
            both["channel"],
            cross=condition.global_cross(),
            dynamics=both.get("dynamics"),
        )[rows, p]
        pedal = pedal_state[rows[:B], p[:B]].bool()
        sequence, pedal_down, _ = sampler.sample_patch(
            context,
            condition.local_step_tensors(flat_index[rows, p]),
            pedal,
            guide=guide,
            temperature=temperature,
            top_p=top_p,
            allow_eos=False,
        )
        for b, (patch_tokens, down) in enumerate(zip(sequence.tolist(), pedal.tolist())):
            out_tokens[b].append([t for t in patch_tokens if t != PAD])
            out_pedal[b].append(down)
        summaries[rows, p] = decoder.summarize_patches(sequence).repeat(2, 1).to(summaries.dtype)
        following = p + 1 < valid.shape[1]
        next_pedal = pedal_down.long().repeat(2)
        pedal_state[rows[following], p[following] + 1] = next_pedal[following]
    return out_tokens, out_pedal


class _Windows(Dataset):
    """ロールアウトを作る窓 (カバー, 損失を取る窓の始まり) を順に作る。窓のトークン化は重いので DataLoader の worker で
    並べて作り、GPU の生成と重ねる"""

    def __init__(self, dataset, windows: list[tuple[int, int]]) -> None:
        self.dataset = dataset
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        cover, first = self.windows[index]
        return self.dataset.cover_item(cover, 0, first=first)


def main() -> None:
    from piano_ar.config import ModelConfig
    from piano_ar.model import LocalSampler
    from piano_ar.tokenizer import PianoTokenizer
    from piano_ar.train import limit_gpu_memory, to_device

    from .config import CoverConfig
    from .data import CoverCache, CoverWindowDataset, collate_cover
    from .model import CoverModel
    from .source import SourceVocab

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", default="data/piano_cover_v3")
    parser.add_argument("--out", default=None, help="既定は <cache-dir>/rollouts_<split>.npz")
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--count", type=int, default=20000, help="作るロールアウトの数")
    parser.add_argument("--start-fraction", type=float, default=0.5, help="曲の冒頭のロールアウトの割合")
    parser.add_argument("--max-start-patches", type=int, default=3)
    parser.add_argument("--max-mid-patches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--window-seconds", type=float, default=64.0)
    parser.add_argument("--memory-seconds", type=float, default=None, help="既定はチェックポイントの学習と同じ")
    parser.add_argument("--source-cfg", type=float, default=1.75)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu-memory-limit", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        limit_gpu_memory(args.gpu_memory_limit)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    cache = CoverCache(args.cache_dir)
    tokenizer = PianoTokenizer(cache.tokenizer_config)
    cover_config = CoverConfig.from_dict(checkpoint["cover_config"])
    model_config = ModelConfig.from_dict(checkpoint["model_config"])
    model = CoverModel(model_config, cover_config, tokenizer, SourceVocab(tokenizer.config).size)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()

    F = tokenizer.patch_frames
    window_patches = round(args.window_seconds / tokenizer.config.patch_seconds)
    memory_seconds = args.memory_seconds
    if memory_seconds is None:
        memory_seconds = checkpoint["args"].get("memory_seconds", 0.0)
    memory_patches = round(memory_seconds / tokenizer.config.patch_seconds)
    dataset = CoverWindowDataset(
        cache,
        split="train" if args.split == "train" else "val",
        window_patches=window_patches,
        cover_config=cover_config,
        channel_dropout=0.0,
        source_dropout=0.0,
        structure_dropout=0.0,
        memory_patches=memory_patches,
        dynamics_bins=model_config.dynamics_bins,
        dynamics_dropout=0.0,
        arrangement=model_config.dynamics_columns > 2,
    )
    # 学習と同じく、長いカバーほど多く選ぶ
    weights = cache.end_frames[dataset.covers].astype(np.float64)
    weights /= weights.sum()

    def make_spec(kind: int) -> tuple[int, int, int, int]:
        """(カバー, 損失を取る窓の始まり, 窓の中で差し替える位置, パッチ数)"""
        while True:
            cover = int(np.random.choice(dataset.covers, p=weights))
            end_frame = int(cache.end_frames[cover])
            num_patches = -(-end_frame // F)
            if kind == 0:
                return cover, 0, 0, random.randint(1, args.max_start_patches)
            if num_patches < 16:
                continue
            first = random.randint(4 * F, max(4 * F, end_frame - window_patches * F // 2))
            offset = (first - max(0, first - memory_patches * F)) // F
            count = random.randint(1, args.max_mid_patches)
            split = offset + random.randint(0, 6)
            if first // F + (split - offset) + count + 2 < num_patches:
                return cover, first, split, count

    np.random.seed(args.seed)
    specs = [make_spec(0 if random.random() < args.start_fraction else 1) for _ in range(args.count)]
    # 同じ種類をまとめて batch にする (生成するパッチ数を揃える)
    order = sorted(range(len(specs)), key=lambda i: (specs[i][2] == 0 and specs[i][1] == 0, specs[i][3]))
    loader = DataLoader(
        _Windows(dataset, [specs[i][:2] for i in order]),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_cover,
    )
    sampler = LocalSampler(model.decoder, tokenizer, model)
    rollouts: list[Rollout] = []
    started = time.time()
    for batch_start, batch in zip(range(0, len(order), args.batch_size), loader):
        chosen = order[batch_start : batch_start + args.batch_size]
        count = max(specs[i][3] for i in chosen)
        batch = to_device(batch, device)
        split = torch.tensor([specs[i][2] for i in chosen])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            tokens, pedal = rollout_batch(model, batch, split, count, source_cfg=args.source_cfg, sampler=sampler)
        for row, i in enumerate(chosen):
            cover, first, split_i, count_i = specs[i]
            start = max(0, first - memory_patches * F)
            kind = 0 if first == 0 and split_i == 0 else 1
            rollouts.append(Rollout(cover, first, start, split_i, kind, tokens[row][:count_i], pedal[row][:count_i]))
        done = batch_start + len(chosen)
        if done // args.batch_size % 20 == 0 or done == len(order):
            elapsed = time.time() - started
            print(
                f"{done}/{len(order)} ({elapsed / 60:.1f} 分、残り約 {elapsed / done * (len(order) - done) / 60:.0f} 分)",
                flush=True,
            )
    out = Path(args.out or Path(args.cache_dir) / f"rollouts_{args.split}.npz")
    RolloutBank(rollouts).save(out)
    kinds = np.bincount([r.kind for r in rollouts], minlength=2)
    notes = np.mean([sum(len(p) for p in r.patches) / len(r.patches) for r in rollouts])
    print(f"{out}: 冒頭 {kinds[0]} / 途中 {kinds[1]}、1 パッチのトークン数の平均 {notes:.0f}")


if __name__ == "__main__":
    main()
