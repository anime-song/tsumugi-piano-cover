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
    memory_patches を渡すと、窓の前に最大その数のパッチを記憶 (文脈) として付ける (window_locate)。
    dynamics_bins > 0 なら、パッチごとの強弱と音の多さの条件 (patch_dynamics) も返す。dynamics_dropout の確率で
    まるごと外し、残したときもそれぞれを dynamics_dropout / 2 の確率で外す (条件なしでも生成できるように)。
    検証の val_start_patch は、損失を取る窓を曲の何パッチ目から始めるか (記憶が効くかを測るときに曲の途中にする)。
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
        memory_patches: int = 0,
        val_start_patch: int = 0,
        dynamics_bins: int = 0,
        dynamics_dropout: float = 0.2,
    ) -> None:
        self.cache = cache
        self.tokenizer = PianoTokenizer(cache.tokenizer_config)
        self.dynamics_bins = dynamics_bins
        self.dynamics_dropout = dynamics_dropout if split == "train" else 0.0
        self.window_patches = window_patches
        self.memory_patches = memory_patches
        self.val_start_patch = val_start_patch
        self.train = split == "train"
        self.augment = augment if self.train else None
        self.song_start_prob = song_start_prob
        self.channel_dropout = channel_dropout if self.train else 0.0
        self.songs = np.flatnonzero(~cache.is_val if self.train else cache.is_val)

    def __len__(self) -> int:
        return len(self.songs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        song = int(self.songs[index])
        events = self.cache.song_events(song)
        end_frame = int(self.cache.end_frames[song])
        if self.augment is not None:
            a = self.augment
            factor = random.uniform(1 - a.time_stretch, 1 + a.time_stretch) if a.time_stretch > 0 else 1.0
            velocity = random.randint(-a.velocity_shift, a.velocity_shift) if a.velocity_shift > 0 else 0
            shift = random.randint(-a.transpose, a.transpose) if a.transpose > 0 else 0
            events, end_frame = augment_events(events, end_frame, factor, shift, velocity, self.tokenizer.config)

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

        channel = int(self.cache.channels[song])
        if random.random() < self.channel_dropout:
            channel = 0
        return {
            "tokens": torch.from_numpy(window["tokens"]),
            "patch_valid": torch.from_numpy(window["patch_valid"]),
            "pedal_state": torch.from_numpy(window["pedal_state"]),
            "channel": torch.tensor(channel),
            "song_start": torch.tensor(int(start == 0)),
            "patch_loss": torch.from_numpy(patch_loss),
            **dynamics_item(
                events,
                end_frame,
                start,
                len(patch_loss),
                self.tokenizer.patch_frames,
                self.dynamics_bins,
                self.dynamics_dropout,
            ),
        }


# 強弱の条件: 標準化した値を ±DYNAMICS_RANGE で切って dynamics_bins 段階にする (0 は「指定なし」)
DYNAMICS_RANGE = 3.0
# 強さを測るのに要る、パッチの中の音の数
DYNAMICS_MIN_NOTES = 3


def quantize_dynamics(z: np.ndarray, bins: int) -> np.ndarray:
    """標準化した値 [..., 2] を 1..bins の番号にする。NaN (測れない) は 0"""
    clipped = np.clip(z, -DYNAMICS_RANGE, DYNAMICS_RANGE - 1e-6)
    index = np.floor((clipped + DYNAMICS_RANGE) / (2 * DYNAMICS_RANGE) * bins)
    return np.where(np.isnan(z), 0, index + 1).astype(np.int64)


def patch_dynamics(events: np.ndarray, end_frame: int, start: int, num_patches: int, patch_frames: int) -> np.ndarray:
    """窓の各パッチの強さ (平均ベロシティ) と音の多さ (音の数の log) を、曲全体の 2 秒ごとの値で標準化して返す [P, 2]。

    曲の中での相対的な強弱なので、演奏者や採譜による音量の絶対値の違いは入らない (学習データの YouTube の採譜は
    強弱の幅が MAESTRO の 6 割ほどしかない)。生成では、この曲線を予測して (piano_cover の Planner)、幅を広げて渡せる。
    音が少なくて強さを測れないパッチは NaN。
    """
    out = np.full((num_patches, 2), np.nan)
    notes = events[events[:, KIND] == KIND_NOTE]
    if len(notes) < 10:
        return out
    onset, velocity = notes[:, ONSET].astype(np.int64), notes[:, VELOCITY].astype(np.float64)
    total = max(1, -(-end_frame // patch_frames))
    index = np.clip(onset // patch_frames, 0, total - 1)
    count = np.bincount(index, minlength=total)
    mean = np.bincount(index, weights=velocity, minlength=total) / np.maximum(count, 1)
    loud = mean[count >= DYNAMICS_MIN_NOTES]
    density = np.log1p(count)
    relative = onset - start
    inside = (relative >= 0) & (relative < num_patches * patch_frames)
    window = relative[inside] // patch_frames
    wcount = np.bincount(window, minlength=num_patches)
    wmean = np.bincount(window, weights=velocity[inside], minlength=num_patches) / np.maximum(wcount, 1)
    if len(loud) >= 2 and loud.std() > 0:
        out[:, 0] = np.where(wcount >= DYNAMICS_MIN_NOTES, (wmean - loud.mean()) / loud.std(), np.nan)
    if density.std() > 0:
        out[:, 1] = (np.log1p(wcount) - density.mean()) / density.std()
    return out


def dynamics_item(
    events: np.ndarray, end_frame: int, start: int, num_patches: int, patch_frames: int, bins: int, dropout: float
) -> dict[str, torch.Tensor]:
    """データセットの 1 件に入れる強弱の条件 {"dynamics": [P, 2] の番号}。bins = 0 なら何も入れない"""
    if not bins:
        return {}
    index = quantize_dynamics(patch_dynamics(events, end_frame, start, num_patches, patch_frames), bins)
    if dropout > 0:
        if random.random() < dropout:
            index[:] = 0
        else:
            for column in range(2):
                if random.random() < dropout / 2:
                    index[:, column] = 0
    return {"dynamics": torch.from_numpy(index)}


def window_locate(
    tokenizer: PianoTokenizer,
    events: np.ndarray,
    end_frame: int,
    window_patches: int,
    memory_patches: int,
    *,
    train: bool,
    song_start_prob: float,
    val_start_patch: int = 0,
) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    """損失を取る窓 (window_patches) の位置を決め、その前に最大 memory_patches の記憶を付けてトークン化する。

    返り値は (tokenize_window の出力 [memory_patches + window_patches], 損失を取るパッチ, 窓 (記憶の頭) の始まりのフレーム)。
    記憶は曲の頭より前には延ばせないので、窓が曲の冒頭に近いときは記憶が短くなり、そのぶん窓の後ろが余る
    (余りは patch_valid を False にして計算しない)。memory_patches = 0 なら以前と同じ固定長の窓。
    """
    F = tokenizer.patch_frames
    if train:
        if random.random() < song_start_prob:
            first = 0
        else:
            # 曲の終わり (EOS) も学習できるよう、窓の後半が曲の外にはみ出す位置まで選ぶ
            first = random.randint(0, max(0, end_frame - window_patches * F // 2))
    else:
        num_patches = -(-end_frame // F)
        first = min(val_start_patch, max(0, num_patches - window_patches)) * F
    start = max(0, first - memory_patches * F)
    offset = (first - start) // F  # 窓の前の記憶のパッチ数 (パッチの境目は start にそろえる)
    window = tokenizer.tokenize_window(events, end_frame, start, memory_patches + window_patches)
    patch_loss = np.zeros(memory_patches + window_patches, dtype=bool)
    patch_loss[offset : offset + window_patches] = True
    window["patch_valid"][offset + window_patches :] = False
    return window, patch_loss & window["patch_valid"], start


def augment_events(
    events: np.ndarray, end_frame: int, factor: float, shift: int, velocity: int, config: TokenizerConfig
) -> tuple[np.ndarray, int]:
    """時間伸縮・移調・ベロシティのずらしをかけて並べ直す"""
    if factor != 1.0:
        events, end_frame = stretch_events(events, end_frame, factor)
    if shift:
        events = transpose_events(events, shift, config)
    if velocity:
        events = shift_velocity(events, velocity)
    return sort_events(np.asarray(events)), end_frame


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out = {key: torch.stack([item[key] for item in batch]) for key in batch[0]}
    # パッチ内トークン長はバッチ内の最長に詰める (大半のパッチは上限よりずっと短い)
    used = int((out["tokens"] != PAD).sum(-1).max())
    out["tokens"] = out["tokens"][..., : max(used, 1)]
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
