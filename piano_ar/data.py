from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .config import TokenizerConfig
from .tokenizer import (
    DURATION,
    KIND,
    KIND_NOTE,
    KIND_PEDAL_OFF,
    KIND_PEDAL_ON,
    ONSET,
    PAD,
    PITCH,
    VELOCITY,
    PianoTokenizer,
    sort_events,
)


@dataclass(frozen=True)
class AugmentConfig:
    # 半音単位の移調幅 (±)
    transpose: int = 5
    # テンポの伸縮率 (±)
    time_stretch: float = 0.1
    # 全体のベロシティのずらし幅 (±)
    velocity_shift: int = 10


@dataclass(frozen=True)
class StyleConfig:
    """スタイル参照の取り方"""

    # 参照の長さ (パッチ数)。学習時はこの範囲からランダムに選ぶ
    min_patches: int = 4
    max_patches: int = 16
    # 同じ曲から取る確率。残りは同じ演奏者の別の曲 (演奏者が分からない・1 曲しかないときは同じ曲)
    same_song_prob: float = 0.7
    # 同じ曲から取るとき、学習する窓からこれだけ離す (同じフレーズを写さないように)
    gap_patches: int = 2
    # スタイルを外す確率 (CFG 用)
    dropout: float = 0.15


class PretrainingCache:
    """prepare.py が作ったキャッシュ。events.npy は DataLoader の各ワーカーで遅延して mmap で開く"""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        songs = np.load(self.cache_dir / "songs.npz")
        self.offsets = songs["offsets"]
        self.end_frames = songs["end_frames"]
        self.channels = songs["channels"]
        self.video_ids = songs["video_ids"]
        self.is_val = songs["is_val"]
        # 演奏者 (-1 は不明) とデータセットの番号。古いキャッシュにはないので、チャンネルとデータセット 0 で代用する
        self.performers = (
            songs["performers"] if "performers" in songs else np.where(self.channels > 0, self.channels, -1)
        )
        self.sources = songs["sources"] if "sources" in songs else np.zeros(len(self.end_frames), dtype=np.int64)
        self.meta = json.loads((self.cache_dir / "meta.json").read_text(encoding="utf-8"))
        self.source_names = self.meta.get("sources", ["channels"])
        self.tokenizer_config = TokenizerConfig(**self.meta["tokenizer"])
        self._events: np.ndarray | None = None

    def song_events(self, index: int) -> np.ndarray:
        if self._events is None:
            self._events = np.load(self.cache_dir / "events.npy", mmap_mode="r")
        return np.asarray(self._events[self.offsets[index] : self.offsets[index + 1]])

    def __getstate__(self) -> dict:
        # mmap を pickle するとデータごとコピーされるので、ワーカーには開く前の状態で渡す
        state = self.__dict__.copy()
        state["_events"] = None
        return state


def stretch_events(events: np.ndarray, end_frame: int, factor: float) -> tuple[np.ndarray, int]:
    events = events.astype(np.int64)
    events[:, ONSET] = np.round(events[:, ONSET] * factor)
    notes = events[:, KIND] == KIND_NOTE
    events[notes, DURATION] = np.maximum(1, np.round(events[notes, DURATION] * factor))
    end_frame = int(round(end_frame * factor))
    # 丸めでペダルの on と off が同じフレームに潰れないよう、on < off <= 次の on を保つ
    on = np.flatnonzero(events[:, KIND] == KIND_PEDAL_ON)
    off = np.flatnonzero(events[:, KIND] == KIND_PEDAL_OFF)
    if len(on):
        on_frames = events[on, ONSET]
        off_frames = np.maximum(events[off, ONSET], on_frames + 1)
        on_frames[1:] = np.maximum(on_frames[1:], off_frames[:-1])
        off_frames = np.maximum(off_frames, on_frames + 1)
        events[on, ONSET] = on_frames
        events[off, ONSET] = off_frames
        end_frame = max(end_frame, int(off_frames.max()) + 1)
    return sort_events(events), end_frame


def transpose_events(events: np.ndarray, shift: int, tokenizer_config: TokenizerConfig) -> np.ndarray:
    events = events.astype(np.int64)
    notes = events[:, KIND] == KIND_NOTE
    events[notes, PITCH] += shift
    pitch = events[:, PITCH]
    return events[~notes | ((pitch >= tokenizer_config.pitch_min) & (pitch <= tokenizer_config.pitch_max))]


def shift_velocity(events: np.ndarray, shift: int) -> np.ndarray:
    events = events.astype(np.int64)
    notes = events[:, KIND] == KIND_NOTE
    events[notes, VELOCITY] = np.clip(events[notes, VELOCITY] + shift, 1, 127)
    return events


class PianoWindowDataset(Dataset):
    """曲から固定長の窓を切り出してパッチごとのトークンにする。

    学習用 (split="train") は index を曲番号として受け取り、窓の位置はランダム。曲の選び方は
    make_sampler の重みで決める。検証用は曲の冒頭の窓に固定して、毎回同じ入力で損失を測る。
    """

    def __init__(
        self,
        cache: PretrainingCache,
        *,
        split: str,
        window_patches: int,
        augment: AugmentConfig | None = None,
        song_start_prob: float = 0.1,
        channel_dropout: float = 0.15,
        style: StyleConfig | None = None,
    ) -> None:
        self.cache = cache
        self.tokenizer = PianoTokenizer(cache.tokenizer_config)
        self.window_patches = window_patches
        self.train = split == "train"
        self.augment = augment if self.train else None
        self.song_start_prob = song_start_prob
        self.channel_dropout = channel_dropout if self.train else 0.0
        self.songs = np.flatnonzero(~cache.is_val if self.train else cache.is_val)
        # style を渡すとチャンネルの代わりにスタイル参照で条件付けする (チャンネルは常に 0)
        self.style = style
        self.songs_of_performer: dict[int, np.ndarray] = {}
        if style is not None:
            performers = cache.performers[self.songs]
            for performer in np.unique(performers[performers >= 0]):
                self.songs_of_performer[int(performer)] = self.songs[performers == performer]

    def __len__(self) -> int:
        return len(self.songs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        song = int(self.songs[index])
        events = self.cache.song_events(song)
        end_frame = int(self.cache.end_frames[song])
        factor, velocity = 1.0, 0
        if self.augment is not None:
            a = self.augment
            factor = random.uniform(1 - a.time_stretch, 1 + a.time_stretch) if a.time_stretch > 0 else 1.0
            velocity = random.randint(-a.velocity_shift, a.velocity_shift) if a.velocity_shift > 0 else 0
            shift = random.randint(-a.transpose, a.transpose) if a.transpose > 0 else 0
            events, end_frame = self._apply(events, end_frame, factor, shift, velocity)

        window_frames = self.window_patches * self.tokenizer.patch_frames
        song_start = not self.train or random.random() < self.song_start_prob
        if song_start:
            start = 0
        else:
            # 曲の終わり (EOS) も学習できるよう、窓の後半が曲の外にはみ出す位置まで選ぶ
            start = random.randint(0, max(0, end_frame - window_frames // 2))
            song_start = start == 0
        window = self.tokenizer.tokenize_window(events, end_frame, start, self.window_patches)

        channel = int(self.cache.channels[song])
        if self.style is not None or random.random() < self.channel_dropout:
            channel = 0
        item = {
            "tokens": torch.from_numpy(window["tokens"]),
            "patch_valid": torch.from_numpy(window["patch_valid"]),
            "pedal_state": torch.from_numpy(window["pedal_state"]),
            "channel": torch.tensor(channel),
            "song_start": torch.tensor(int(song_start)),
        }
        if self.style is not None:
            # 窓の範囲を元の (伸縮前の) 時間に戻して、参照が窓と重ならないようにする
            window_range = (start / factor, (start + window_frames) / factor)
            item.update(self._reference(song, window_range, velocity))
        return item

    def _apply(
        self, events: np.ndarray, end_frame: int, factor: float, shift: int, velocity: int
    ) -> tuple[np.ndarray, int]:
        if factor != 1.0:
            events, end_frame = stretch_events(events, end_frame, factor)
        if shift:
            events = transpose_events(events, shift, self.tokenizer.config)
        if velocity:
            events = shift_velocity(events, velocity)
        return sort_events(np.asarray(events)), end_frame

    def _reference(self, song: int, window_range: tuple[float, float], velocity: int) -> dict[str, torch.Tensor]:
        """スタイル参照のパッチ列。同じ曲の窓と離れた場所か、同じ演奏者の別の曲から取る。

        参照には窓とは別の移調と時間伸縮をかけ、音高やテンポをそのまま写しても合わないようにする
        (ベロシティのずらしは窓と同じにする。強弱の付け方はスタイルの一部なので)。
        検証では毎回同じ参照になるよう、同じ曲の窓の直後から固定の長さで取る。
        """
        c = self.style
        F = self.tokenizer.patch_frames
        empty = {
            "ref_tokens": torch.full((c.max_patches, 1), PAD, dtype=torch.long),
            "ref_valid": torch.zeros(c.max_patches, dtype=torch.bool),
            "style_present": torch.tensor(False),
        }
        # 学習時は dropout の確率で外す。1 以上なら検証でも常に外す (スタイルなしの損失を測る用)
        if c.dropout >= 1.0 or (self.train and random.random() < c.dropout):
            return empty

        if self.train:
            num_patches = random.randint(c.min_patches, c.max_patches)
            factor = (
                random.uniform(1 - self.augment.time_stretch, 1 + self.augment.time_stretch) if self.augment else 1.0
            )
            shift = random.randint(-self.augment.transpose, self.augment.transpose) if self.augment else 0
        else:
            num_patches, factor, shift = (c.min_patches + c.max_patches) // 2, 1.0, 0
        length = num_patches * F / factor  # 元の時間での参照の長さ
        gap = c.gap_patches * F

        ref_song, ref_start = song, None
        others = self.songs_of_performer.get(int(self.cache.performers[song]), np.zeros(0, dtype=np.int64))
        others = others[others != song]
        use_other = self.train and len(others) and random.random() >= c.same_song_prob
        if not use_other:
            end = float(self.cache.end_frames[song])
            if self.train:
                # 窓の前と後ろのうち、参照が収まる場所からランダムに選ぶ
                candidates = [(0.0, window_range[0] - gap - length), (window_range[1] + gap, end - length)]
                candidates = [(a, b) for a, b in candidates if b >= a]
                if candidates:
                    lo, hi = random.choice(candidates)
                    ref_start = random.uniform(lo, hi)
            elif window_range[1] + gap + length <= end:
                ref_start = window_range[1] + gap
            if ref_start is None and self.train and len(others):
                use_other = True
        if use_other:
            ref_song = int(random.choice(others))
            ref_start = random.uniform(0.0, max(0.0, float(self.cache.end_frames[ref_song]) - length))
        if ref_start is None:
            return empty

        events, end_frame = self._apply(
            self.cache.song_events(ref_song), int(self.cache.end_frames[ref_song]), factor, shift, velocity
        )
        window = self.tokenizer.tokenize_window(events, end_frame, int(ref_start * factor), num_patches)
        tokens = torch.full((c.max_patches, window["tokens"].shape[1]), PAD, dtype=torch.long)
        tokens[:num_patches] = torch.from_numpy(window["tokens"])
        valid = torch.zeros(c.max_patches, dtype=torch.bool)
        valid[:num_patches] = torch.from_numpy(window["patch_valid"])
        return {"ref_tokens": tokens, "ref_valid": valid, "style_present": torch.tensor(bool(valid.any()))}


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if "ref_tokens" in batch[0]:
        # 参照のトークン長は曲ごとに違うので、一番長いものに揃えてから積む
        width = max(item["ref_tokens"].shape[1] for item in batch)
        for item in batch:
            pad = width - item["ref_tokens"].shape[1]
            if pad:
                item["ref_tokens"] = torch.nn.functional.pad(item["ref_tokens"], (0, pad), value=PAD)
    out = {key: torch.stack([item[key] for item in batch]) for key in batch[0]}
    # パッチ内トークン長はバッチ内の最長に詰める (大半のパッチは上限よりずっと短い)
    for key in ("tokens", "ref_tokens"):
        if key in out:
            used = int((out[key] != PAD).sum(-1).max())
            out[key] = out[key][..., : max(used, 1)]
    return out


def sampling_weights(
    dataset: PianoWindowDataset,
    channel_alpha: float = 0.5,
    source_weights: dict[str, float] | None = None,
    balance_hours: float = 10.0,
) -> np.ndarray:
    """曲ごとの選ばれやすさ。曲の長さに比例させつつ、曲数の多い演奏者だけを抑える。

    総時間が balance_hours を超える演奏者は、選ばれる時間が (総時間)^channel_alpha に比例するまで下げる
    (1 で抑えない、0 でしきい値と同じ時間まで)。それ以下の演奏者と演奏者不明の曲 (MAESTRO など) は長さに比例のまま。
    演奏者の大半は 1〜数曲なので、全員を均等に近づけるとその少数の曲ばかりが選ばれてしまう。
    source_weights でデータセットごとの割合に倍率をかけられる。
    """
    cache = dataset.cache
    lengths = cache.end_frames[dataset.songs].astype(np.float64)
    performers = cache.performers[dataset.songs]
    known = performers >= 0
    totals = np.zeros(len(lengths))
    if known.any():
        _, inverse = np.unique(performers[known], return_inverse=True)
        totals[known] = np.bincount(inverse, weights=lengths[known])[inverse]
    threshold = balance_hours * 3600 * cache.tokenizer_config.frame_rate
    weights = lengths * (np.maximum(totals, threshold) / threshold) ** (channel_alpha - 1)
    for name, scale in (source_weights or {}).items():
        weights[cache.sources[dataset.songs] == cache.source_names.index(name)] *= scale
    return weights


def make_sampler(
    dataset: PianoWindowDataset,
    num_samples: int,
    channel_alpha: float = 0.5,
    source_weights: dict[str, float] | None = None,
    balance_hours: float = 10.0,
) -> WeightedRandomSampler:
    weights = sampling_weights(dataset, channel_alpha, source_weights, balance_hours)
    return WeightedRandomSampler(torch.from_numpy(weights), num_samples=num_samples, replacement=True)
