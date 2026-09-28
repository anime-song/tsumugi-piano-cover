"""カバー学習のデータ。カバーの窓 (デコーダの入力) と、原曲の全曲 (エンコーダの入力) を組にして返す。"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from piano_ar.config import TokenizerConfig
from piano_ar.data import (
    AugmentConfig,
    PianoWindowDataset,
    StyleConfig,
    collate,
    sampling_weights,
    shift_velocity,
    stretch_events,
    style_reference,
    transpose_events,
)
from piano_ar.tokenizer import PianoTokenizer

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
    }


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
    style を渡すとチャンネルの代わりにスタイル参照で条件付けする。参照は同じカバーの窓から離れた場所か、
    同じチャンネルの別のカバー・事前学習の曲 (学習時のみ) から取る。
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
        style: StyleConfig | None = None,
    ) -> None:
        self.cache = cache
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
        self.style = style
        # 同じチャンネルの別の曲 (カバーの番号、事前学習の曲の番号)。0 はチャンネル不明
        self.covers_of_channel: dict[int, np.ndarray] = {}
        self.pretraining_of_channel: dict[int, np.ndarray] = {}
        if style is not None:
            channels = cache.channels[self.covers]
            for channel in np.unique(channels[channels > 0]):
                self.covers_of_channel[int(channel)] = self.covers[channels == channel]
            if pretraining is not None and self.train:
                songs = pretraining.songs
                channels = pretraining.cache.channels[songs]
                for channel in np.unique(channels[channels > 0]):
                    self.pretraining_of_channel[int(channel)] = songs[channels == channel]

    def __len__(self) -> int:
        return len(self.covers) + (len(self.pretraining) if self.pretraining is not None else 0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index >= len(self.covers):
            item = self.pretraining[index - len(self.covers)]
            item["align"] = torch.zeros(self.window_patches, 2)
            item["has_source"] = torch.tensor(False)
            item.update(empty_source())
            return item

        cover = int(self.covers[index])
        source = int(self.cache.source_index[cover])
        raw_events = events = self.cache.cover_events(cover)
        raw_end = end_frame = int(self.cache.end_frames[cover])
        align = self.cache.cover_align(cover)
        rows = self.cache.source_rows(source)
        source_end = int(self.cache.source_end_frames[source])

        factor, velocity = 1.0, 0
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
                velocity = random.randint(-a.velocity_shift, a.velocity_shift)
                events = shift_velocity(events, velocity)

        F = self.tokenizer.patch_frames
        window_frames = self.window_patches * F
        song_start = not self.train or random.random() < self.song_start_prob
        if song_start:
            start = 0
        else:
            start = random.randint(0, max(0, end_frame - window_frames // 2))
            song_start = start == 0
        window = self.tokenizer.tokenize_window(events, end_frame, start, self.window_patches)

        # 窓の各パッチの始まりと終わりが、原曲のどのフレームに当たるか (伸縮したら両方の時刻を factor 倍する)
        bounds = start + np.arange(self.window_patches + 1) * F
        mapped = factor * np.interp(bounds / factor / self.cache.align_step, np.arange(len(align)), align)
        align_window = np.stack([mapped[:-1], mapped[1:]], axis=1).astype(np.float32)

        has_source = random.random() >= self.source_dropout
        if has_source:
            for row_type in (TYPE_BEAT, TYPE_CHORD, TYPE_KEY):
                if random.random() < self.structure_dropout:
                    rows = rows[rows[:, ROW_TYPE] != row_type]
            features = source_features(rows, source_end, self.tokenizer, self.vocab, self.cover_config.max_source_rows)
            source_items = source_tensors(features)
        else:
            source_items = empty_source()

        channel = int(self.cache.channels[cover])
        if self.style is not None or random.random() < self.channel_dropout:
            channel = 0
        item = {
            "tokens": torch.from_numpy(window["tokens"]),
            "patch_valid": torch.from_numpy(window["patch_valid"]),
            "pedal_state": torch.from_numpy(window["pedal_state"]),
            "channel": torch.tensor(channel),
            "song_start": torch.tensor(int(song_start)),
            "align": torch.from_numpy(align_window),
            "has_source": torch.tensor(has_source),
            **source_items,
        }
        if self.style is not None:
            window_range = (start / factor, (start + window_frames) / factor)
            item.update(self._reference(cover, raw_events, raw_end, window_range, velocity))
        return item

    def _reference(
        self, cover: int, events: np.ndarray, end_frame: int, window_range: tuple[float, float], velocity: int
    ) -> dict[str, torch.Tensor]:
        channel = int(self.cache.channels[cover])
        covers = self.covers_of_channel.get(channel, np.zeros(0, dtype=np.int64))
        covers = covers[covers != cover]
        songs = self.pretraining_of_channel.get(channel, np.zeros(0, dtype=np.int64))

        def pick_other() -> tuple[np.ndarray, int]:
            # カバーと事前学習の曲をまとめて 1 曲選ぶ
            k = random.randrange(len(covers) + len(songs))
            if k < len(covers):
                return self.cache.cover_events(int(covers[k])), int(self.cache.end_frames[covers[k]])
            song = int(songs[k - len(covers)])
            return self.pretraining.cache.song_events(song), int(self.pretraining.cache.end_frames[song])

        return style_reference(
            self.tokenizer,
            self.style,
            self.augment,
            self.train,
            events,
            end_frame,
            window_range,
            velocity,
            pick_other if len(covers) + len(songs) else None,
        )


SOURCE_KEYS = ("src_features", "src_onset", "src_valid", "src_patch_valid")


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
