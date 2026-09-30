"""カバー学習のデータ。カバーの窓 (デコーダの入力) と、原曲の全曲 (エンコーダの入力) を組にして返す。"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from piano_ar.config import TokenizerConfig
from piano_ar.data import (
    DYNAMICS_MIN_NOTES,
    AugmentConfig,
    PianoWindowDataset,
    collate,
    dynamics_item,
    quantize_dynamics,
    sampling_weights,
    shift_velocity,
    stretch_events,
    transpose_events,
    window_locate,
)
from piano_ar.tokenizer import KIND, KIND_NOTE, ONSET, PianoTokenizer, sort_events

from .arrangement import ARRANGEMENT_NAMES, arrangement_values, map_to_source
from .config import CoverConfig
from .source import (
    ROW_TYPE,
    TYPE_BEAT,
    TYPE_CHORD,
    TYPE_KEY,
    SourceVocab,
    sort_rows,
    source_features,
    stretch_rows,
    transpose_rows,
)

SPLITS = {"train": 0, "val": 1, "test": 2}


@dataclass(frozen=True)
class DriftConfig:
    """窓の前半 (文脈) のカバーの時刻を原曲に対してゆっくりずらし、後半は正しいまま返す。損失は後半だけで取る。

    生成では自分の出した過去が原曲から少しずつずれていくが (exposure bias)、学習では過去が常に原曲と合っているので、
    ずれた状態から原曲に戻ることを学べない (v1 は約 2% 遅いテンポでずれていき、8 分音符 1 つ分で戻っていた)。
    文脈の最後の ramp 秒で 0 から shift まで線形にずらし (テンポのずれ)、そのあと正しい時刻に戻った続きを当てさせる。
    """

    prob: float = 0.5  # 窓をずらす確率
    min_shift: float = 0.02  # 文脈の終わりでのずれ (秒、符号はランダム)
    max_shift: float = 0.15
    min_ramp: float = 2.0  # ずれが溜まっていく長さ (秒)
    max_ramp: float = 10.0
    # 損失を取るのは戻った後の何パッチか (0 で窓の終わりまで)。検証で「戻る」力だけを測るときに絞る
    loss_patches: int = 0


def drift_events(events: np.ndarray, split: int, shift: float, ramp: float) -> np.ndarray:
    """split フレームより前のイベントを、split - ramp から split まで 0 -> shift フレームと線形に増えるだけずらす"""
    events = events.astype(np.int64)
    onset = events[:, ONSET]
    before = onset < split
    amount = shift * np.clip((onset - (split - ramp)) / ramp, 0.0, 1.0)
    events[before, ONSET] = np.maximum(0, np.round(onset[before] + amount[before]))
    return sort_events(events)


class CoverCache:
    """prepare.py が作ったキャッシュ。大きな配列は DataLoader の各ワーカーで遅延して mmap で開く"""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        sources = np.load(self.cache_dir / "sources.npz")
        self.source_offsets = sources["offsets"]
        self.source_end_frames = sources["end_frames"]
        self.source_ids = sources["video_ids"]
        covers = np.load(self.cache_dir / "covers.npz")
        self.cover_offsets = covers["offsets"]
        self.align_offsets = covers["align_offsets"]
        self.end_frames = covers["end_frames"]
        self.video_ids = covers["video_ids"]
        self.source_index = covers["source_index"]
        self.channels = covers["channels"]
        self.split = covers["split"]
        self.meta = json.loads((self.cache_dir / "meta.json").read_text(encoding="utf-8"))
        self.tokenizer_config = TokenizerConfig(**self.meta["tokenizer"])
        self.align_step = int(self.meta["align_step"])
        self._arrays: dict[str, np.ndarray] | None = None

    def _open(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            self._arrays = {
                name: np.load(self.cache_dir / f"{name}.npy", mmap_mode="r")
                for name in ("source_rows", "cover_events", "align")
            }
        return self._arrays

    def cover_events(self, cover: int) -> np.ndarray:
        return np.asarray(self._open()["cover_events"][self.cover_offsets[cover] : self.cover_offsets[cover + 1]])

    def cover_align(self, cover: int) -> np.ndarray:
        return np.asarray(self._open()["align"][self.align_offsets[cover] : self.align_offsets[cover + 1]])

    def source_rows(self, source: int) -> np.ndarray:
        return np.asarray(self._open()["source_rows"][self.source_offsets[source] : self.source_offsets[source + 1]])

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_arrays"] = None
        return state


def empty_source() -> dict[str, torch.Tensor]:
    return {
        "src_features": torch.zeros(1, 1, 5, dtype=torch.int16),
        "src_onset": torch.zeros(1, 1, dtype=torch.int32),
        "src_valid": torch.zeros(1, 1, dtype=torch.bool),
        "src_patch_valid": torch.zeros(1, dtype=torch.bool),
        "plan_target": torch.full((1, 2), float("nan")),
    }


def plan_target(
    events: np.ndarray, align: np.ndarray, align_step: int, factor: float, num_patches: int, patch_frames: int
) -> np.ndarray:
    """原曲のパッチごとの、カバーの強さ (平均ベロシティ) と音の多さ (音の数の log) を、カバーの中で標準化したもの [S, 2]。

    Planner (原曲から強弱の曲線を予測する) の正解。カバーの音をアラインメントで原曲の時刻に写して数える。
    カバーが弾いていない所と、強さを測れない所 (音が少ない) は NaN。
    """
    out = np.full((num_patches, 2), np.nan, dtype=np.float32)
    notes = events[events[:, 1] == 2]
    if len(notes) < 10:
        return out
    mapped = factor * np.interp(notes[:, 0] / factor / align_step, np.arange(len(align)), align)
    patch = np.clip((mapped // patch_frames).astype(np.int64), 0, num_patches - 1)
    count = np.bincount(patch, minlength=num_patches)
    mean = np.bincount(patch, weights=notes[:, 4].astype(np.float64), minlength=num_patches) / np.maximum(count, 1)
    # カバーが弾いている範囲 (端のパッチは半端なので除く)
    lo, hi = int(factor * align[0] // patch_frames) + 1, int(factor * align[-1] // patch_frames)
    inside = np.zeros(num_patches, dtype=bool)
    inside[max(lo, 0) : max(hi, 0)] = True
    loud = inside & (count >= DYNAMICS_MIN_NOTES)
    if loud.sum() >= 5 and mean[loud].std() > 0:
        out[loud, 0] = (mean[loud] - mean[loud].mean()) / mean[loud].std()
    density = np.log1p(count[inside])
    if inside.sum() >= 5 and density.std() > 0:
        out[inside, 1] = (density - density.mean()) / density.std()
    return out


def source_tensors(features: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        "src_features": torch.from_numpy(features["features"]),
        "src_onset": torch.from_numpy(features["onset"]),
        "src_valid": torch.from_numpy(features["valid"]),
        "src_patch_valid": torch.ones(features["features"].shape[0], dtype=torch.bool),
    }


class CoverWindowDataset(Dataset):
    """カバーから固定長の窓を切り出し、原曲の全曲と、窓の各パッチに対応する原曲の時刻を付けて返す。

    学習用は窓の位置がランダムで、原曲とカバーに同じ時間伸縮・移調をかける (ベロシティのずらしはカバーだけ)。
    source_dropout の確率で原曲を外し (CFG 用)、structure_dropout の確率で拍・コード・キーをそれぞれ外す
    (推定値なので、なくても動くようにする)。
    pretraining を渡すと、index が カバーの数 以上のときは事前学習の曲を原曲なしで返す (忘却を防ぐため混ぜる)。
    drift を渡すと、窓の前半のカバーの時刻をずらして後半で原曲に戻らせる (DriftConfig)。検証では窓ごとに決まった乱数で
    ずらすので、毎回同じ窓になる。
    memory_patches と val_start_patch は PianoWindowDataset と同じ (窓の前の記憶。pretraining も同じ長さにそろえる)。
    dynamics_bins > 0 なら、窓のパッチごとの強弱と音の多さの条件 (piano_ar.data.dynamics_item) と、
    Planner の正解 (原曲のパッチごとの曲線、plan_target) も返す。arrangement なら、その後ろに編曲の性質
    (piano_cover.arrangement の fill / above / span) の列も足す (条件は強弱と同じ確率で列ごとに落とす)。
    """

    def __init__(
        self,
        cache: CoverCache,
        *,
        split: str,
        window_patches: int,
        cover_config: CoverConfig,
        augment: AugmentConfig | None = None,
        song_start_prob: float = 0.1,
        channel_dropout: float = 0.15,
        source_dropout: float = 0.1,
        structure_dropout: float = 0.2,
        pretraining: PianoWindowDataset | None = None,
        drift: DriftConfig | None = None,
        memory_patches: int = 0,
        val_start_patch: int = 0,
        dynamics_bins: int = 0,
        dynamics_dropout: float = 0.2,
        arrangement: bool = False,
    ) -> None:
        self.cache = cache
        self.arrangement = arrangement and dynamics_bins > 0
        self.dynamics_bins = dynamics_bins
        self.dynamics_dropout = dynamics_dropout if split == "train" else 0.0
        self.tokenizer = PianoTokenizer(cache.tokenizer_config)
        self.vocab = SourceVocab(cache.tokenizer_config)
        self.window_patches = window_patches
        self.cover_config = cover_config
        self.train = split == "train"
        self.augment = augment if self.train else None
        self.song_start_prob = song_start_prob
        self.channel_dropout = channel_dropout if self.train else 0.0
        self.source_dropout = source_dropout
        self.structure_dropout = structure_dropout if self.train else 0.0
        self.covers = np.flatnonzero(cache.split == SPLITS[split])
        self.pretraining = pretraining
        self.drift = drift
        self.memory_patches = memory_patches
        self.val_start_patch = val_start_patch

    def __len__(self) -> int:
        return len(self.covers) + (len(self.pretraining) if self.pretraining is not None else 0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index >= len(self.covers):
            item = self.pretraining[index - len(self.covers)]
            item["align"] = torch.zeros(len(item["tokens"]), 2)
            item["has_source"] = torch.tensor(False)
            item.update(empty_source())
            if self.arrangement and "dynamics" in item:
                # 事前学習の曲には原曲がないので、編曲の性質の列は「指定なし」
                extra = torch.zeros(len(item["dynamics"]), len(ARRANGEMENT_NAMES), dtype=item["dynamics"].dtype)
                item["dynamics"] = torch.cat([item["dynamics"], extra], dim=1)
            return item

        cover = int(self.covers[index])
        source = int(self.cache.source_index[cover])
        events = self.cache.cover_events(cover)
        end_frame = int(self.cache.end_frames[cover])
        align = self.cache.cover_align(cover)
        rows = self.cache.source_rows(source)
        source_end = int(self.cache.source_end_frames[source])

        factor = 1.0
        if self.augment is not None:
            a = self.augment
            if a.time_stretch > 0:
                factor = random.uniform(1 - a.time_stretch, 1 + a.time_stretch)
                events, end_frame = stretch_events(events, end_frame, factor)
                rows = sort_rows(stretch_rows(rows, factor))
                source_end = int(round(source_end * factor))
            if a.transpose > 0:
                shift = random.randint(-a.transpose, a.transpose)
                events = transpose_events(events, shift, self.tokenizer.config)
                rows = transpose_rows(rows, shift)
            if a.velocity_shift > 0:
                events = shift_velocity(events, random.randint(-a.velocity_shift, a.velocity_shift))

        F = self.tokenizer.patch_frames
        window, patch_loss, start = window_locate(
            self.tokenizer,
            events,
            end_frame,
            self.window_patches,
            self.memory_patches,
            train=self.train,
            song_start_prob=self.song_start_prob,
            val_start_patch=self.val_start_patch,
        )
        song_start = start == 0
        total = len(patch_loss)
        patch_loss = self._drift(window, patch_loss, events, end_frame, start, index)

        # 窓の各パッチの始まりと終わりが、原曲のどのフレームに当たるか (伸縮したら両方の時刻を factor 倍する)
        bounds = start + np.arange(total + 1) * F
        mapped = factor * np.interp(bounds / factor / self.cache.align_step, np.arange(len(align)), align)
        align_window = np.stack([mapped[:-1], mapped[1:]], axis=1).astype(np.float32)
        frame_rate = self.tokenizer.config.frame_rate
        if self.arrangement:
            notes = events[events[:, KIND] == KIND_NOTE]
            note_times = map_to_source(notes[:, ONSET].astype(np.float64), align, self.cache.align_step, factor)
            # 窓の各パッチの編曲の性質 (パッチの境界を原曲の時刻に写した区間で測る)
            window_arrangement = arrangement_values(events, note_times, rows, mapped, frame_rate)

        has_source = random.random() >= self.source_dropout
        if has_source:
            for row_type in (TYPE_BEAT, TYPE_CHORD, TYPE_KEY):
                if random.random() < self.structure_dropout:
                    rows = rows[rows[:, ROW_TYPE] != row_type]
            features = source_features(rows, source_end, self.tokenizer, self.vocab, self.cover_config.max_source_rows)
            source_items = source_tensors(features)
            if self.dynamics_bins:
                S = len(features["features"])
                target = plan_target(events, align, self.cache.align_step, factor, S, F)
                if self.arrangement:
                    values = arrangement_values(events, note_times, rows, np.arange(S + 1) * F, frame_rate)
                    target = np.concatenate([target, values.astype(np.float32)], axis=1)
                source_items["plan_target"] = torch.from_numpy(target)
        else:
            source_items = empty_source()

        channel = int(self.cache.channels[cover])
        if random.random() < self.channel_dropout:
            channel = 0
        dynamics = dynamics_item(events, end_frame, start, total, F, self.dynamics_bins, self.dynamics_dropout)
        if self.arrangement:
            index = quantize_dynamics(window_arrangement, self.dynamics_bins)
            if self.dynamics_dropout > 0:
                if random.random() < self.dynamics_dropout:
                    index[:] = 0
                else:
                    for column in range(index.shape[1]):
                        if random.random() < self.dynamics_dropout / 2:
                            index[:, column] = 0
            dynamics["dynamics"] = torch.cat([dynamics["dynamics"], torch.from_numpy(index)], dim=1)
        return {
            "tokens": torch.from_numpy(window["tokens"]),
            "patch_valid": torch.from_numpy(window["patch_valid"]),
            "pedal_state": torch.from_numpy(window["pedal_state"]),
            "channel": torch.tensor(channel),
            "song_start": torch.tensor(int(song_start)),
            "align": torch.from_numpy(align_window),
            "has_source": torch.tensor(has_source),
            "patch_loss": torch.from_numpy(patch_loss),
            **dynamics,
            **source_items,
        }

    def _drift(
        self,
        window: dict[str, np.ndarray],
        patch_loss: np.ndarray,
        events: np.ndarray,
        end_frame: int,
        start: int,
        index: int,
    ) -> np.ndarray:
        """DriftConfig の確率で、損失を取る窓の前半 (とその前の記憶) をずらしたトークンに差し替え、損失を取るパッチを返す"""
        d = self.drift
        rng = random if self.train else random.Random(index)
        loss_patches = np.flatnonzero(patch_loss)
        if d is None or len(loss_patches) < 4 or rng.random() >= d.prob:
            return patch_loss
        patch_loss = patch_loss.copy()
        F = self.tokenizer.patch_frames
        fr = self.tokenizer.config.frame_rate
        first = int(loss_patches[0])
        # 戻るパッチ (損失を多く残すよう窓の前半から選ぶ)
        split = first + rng.randint(2, max(2, len(loss_patches) // 2))
        shift = rng.uniform(d.min_shift, d.max_shift) * rng.choice((-1, 1)) * fr
        ramp = rng.uniform(d.min_ramp, d.max_ramp) * fr
        drifted = drift_events(events, start + split * F, shift, ramp)
        context = self.tokenizer.tokenize_window(drifted, end_frame, start, split)
        window["tokens"][:split] = context["tokens"]
        window["pedal_state"][:split] = context["pedal_state"]
        patch_loss[:split] = False
        if d.loss_patches:
            patch_loss[split + d.loss_patches :] = False
        return patch_loss


SOURCE_KEYS = ("src_features", "src_onset", "src_valid", "src_patch_valid", "plan_target")


def collate_cover(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """デコーダ側は事前学習と同じ collate。原曲は曲ごとにパッチ数と行数が違うので最大に合わせて詰める"""
    out = collate([{k: v for k, v in item.items() if k not in SOURCE_KEYS} for item in batch])
    num_patches = max(item["src_features"].shape[0] for item in batch)
    num_rows = max(item["src_features"].shape[1] for item in batch)
    features = torch.zeros(len(batch), num_patches, num_rows, 5, dtype=torch.int16)
    onset = torch.zeros(len(batch), num_patches, num_rows, dtype=torch.int32)
    valid = torch.zeros(len(batch), num_patches, num_rows, dtype=torch.bool)
    patch_valid = torch.zeros(len(batch), num_patches, dtype=torch.bool)
    for i, item in enumerate(batch):
        s, r = item["src_features"].shape[:2]
        features[i, :s, :r] = item["src_features"]
        onset[i, :s, :r] = item["src_onset"]
        valid[i, :s, :r] = item["src_valid"]
        patch_valid[i, :s] = item["src_patch_valid"]
    out.update(src_features=features, src_onset=onset, src_valid=valid, src_patch_valid=patch_valid)
    if any("plan_target" in item for item in batch):
        # 原曲のない行 (事前学習の曲) の正解は列が少ない (すべて NaN) ので、一番多い列に合わせる
        columns = max(item["plan_target"].shape[1] for item in batch if "plan_target" in item)
        target = torch.full((len(batch), num_patches, columns), float("nan"))
        for i, item in enumerate(batch):
            if "plan_target" in item:
                s, c = item["plan_target"].shape
                target[i, :s, :c] = item["plan_target"]
        out["plan_target"] = target
    return out


def make_cover_sampler(
    dataset: CoverWindowDataset, num_samples: int, pretrain_mix: float, channel_alpha: float = 0.5
) -> WeightedRandomSampler:
    """カバーは長さに比例して選ぶ。事前学習の曲は全体の pretrain_mix の割合になるよう重みを付ける"""
    cover_weights = dataset.cache.end_frames[dataset.covers].astype(np.float64)
    cover_weights /= cover_weights.sum()
    weights = [cover_weights * (1 - pretrain_mix if dataset.pretraining is not None else 1.0)]
    if dataset.pretraining is not None:
        pretrain_weights = sampling_weights(dataset.pretraining, channel_alpha)
        weights.append(pretrain_weights / pretrain_weights.sum() * pretrain_mix)
    return WeightedRandomSampler(torch.from_numpy(np.concatenate(weights)), num_samples=num_samples, replacement=True)
