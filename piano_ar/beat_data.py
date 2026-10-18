"""拍のトークン (piano_ar.beat_tokenizer) の事前学習のデータ。窓は小節の単位で切り出す。

拍は piano_ar.prepare_beats がキャッシュに足した beats.npz から読む (拍のない曲は使わない)。
拡張は移調とベロシティのずらしと、テンポの条件の伸縮 (拍の刻みは時間の伸縮で変わらないので、伸縮は小節の
テンポの条件にだけ効く)。
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .beat_tokenizer import BeatGrid, BeatTokenizer, QuantizedSong
from .config import BeatTokenizerConfig
from .data import AugmentConfig, PretrainingCache, augment_events, quantize_dynamics
from .tokenizer import KIND_NOTE

# 強さを測るのに要る、小節の中の音の数
DYNAMICS_MIN_NOTES = 3


class BeatAnnotations:
    """キャッシュの beats.npz (曲ごとの拍のフレームと小節の頭)"""

    def __init__(self, cache_dir: str | Path) -> None:
        data = np.load(Path(cache_dir) / "beats.npz")
        self.offsets = data["offsets"]
        self.beats = data["beats"]
        self.downbeat = data["downbeat"]

    def has(self, song: int) -> bool:
        """拍の格子を作れるか (拍が 4 つ以上で、時刻が増えていく)"""
        beats = self.beats[self.offsets[song] : self.offsets[song + 1]]
        return len(beats) >= 4 and bool(np.all(np.diff(beats) > 0))

    def grid(self, song: int, first_frame: float, last_frame: float, max_bar_beats: int) -> BeatGrid | None:
        lo, hi = self.offsets[song], self.offsets[song + 1]
        beats = self.beats[lo:hi].astype(np.float64)
        downbeats = beats[self.downbeat[lo:hi]]
        return BeatGrid.from_frames(beats, downbeats, first_frame, last_frame, max_bar_beats)


def bar_dynamics(song: QuantizedSong, grid: BeatGrid, tokenizer: BeatTokenizer, start_bar: int, num_bars: int):
    """窓の各小節の強さ (平均ベロシティ) と音の多さ (1 拍あたりの音の数の log) を、曲の全小節の値で標準化する [P, 2]。
    piano_ar.data.patch_dynamics の小節版。音が少なくて強さを測れない小節は NaN"""
    out = np.full((num_bars, 2), np.nan)
    note = song.kind == KIND_NOTE
    if note.sum() < 10:
        return out
    bar_ticks = grid.bar_starts * tokenizer.ticks_per_beat
    bar = np.clip(np.searchsorted(bar_ticks, song.tick[note], side="right") - 1, 0, grid.num_bars - 1)
    count = np.bincount(bar, minlength=grid.num_bars).astype(np.float64)
    mean = np.bincount(bar, weights=song.velocity[note].astype(np.float64), minlength=grid.num_bars) / np.maximum(
        count, 1
    )
    density = np.log1p(count / np.maximum(grid.bar_beats, 1))
    last = min(grid.num_bars, int(np.searchsorted(bar_ticks, song.end_tick, side="right")))
    loud = mean[:last][count[:last] >= DYNAMICS_MIN_NOTES]
    bars = np.arange(start_bar, start_bar + num_bars)
    inside = (bars >= 0) & (bars < last)
    if len(loud) >= 2 and loud.std() > 0:
        values = np.where(count >= DYNAMICS_MIN_NOTES, (mean - loud.mean()) / loud.std(), np.nan)
        out[inside, 0] = values[bars[inside]]
    if density[:last].std() > 0:
        out[inside, 1] = ((density - density[:last].mean()) / density[:last].std())[bars[inside]]
    return out


class BeatWindowDataset(Dataset):
    """曲から小節の窓を切り出してトークンにする (piano_ar.data.PianoWindowDataset の小節版)。

    学習用は窓の位置がランダムで、song_start_prob の確率で曲の頭から。窓の前に最大 memory_bars の記憶を付ける。
    検証用は val_start_bar から (記憶はその前の分)。dynamics_bins > 0 なら小節ごとの強弱の条件 (bar_dynamics) も返す。
    """

    def __init__(
        self,
        cache: PretrainingCache,
        annotations: BeatAnnotations,
        tokenizer_config: BeatTokenizerConfig,
        *,
        split: str,
        window_bars: int,
        memory_bars: int = 0,
        augment: AugmentConfig | None = None,
        song_start_prob: float = 0.1,
        channel_dropout: float = 0.15,
        val_start_bar: int = 0,
        dynamics_bins: int = 0,
        dynamics_dropout: float = 0.2,
    ) -> None:
        self.cache = cache
        self.annotations = annotations
        self.tokenizer = BeatTokenizer(tokenizer_config)
        self.window_bars = window_bars
        self.memory_bars = memory_bars
        self.train = split == "train"
        self.augment = augment if self.train else None
        self.song_start_prob = song_start_prob
        self.channel_dropout = channel_dropout if self.train else 0.0
        self.val_start_bar = val_start_bar
        self.dynamics_bins = dynamics_bins
        self.dynamics_dropout = dynamics_dropout if self.train else 0.0
        split_songs = np.flatnonzero(~cache.is_val if self.train else cache.is_val)
        self.songs = np.array([s for s in split_songs if annotations.has(int(s))], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.songs)

    def song_window(self, song: int, first_bar: int | None = None) -> dict[str, torch.Tensor]:
        events = self.cache.song_events(song)
        end_frame = int(self.cache.end_frames[song])
        frame_rate = self.cache.tokenizer_config.frame_rate
        tempo_rate = float(frame_rate)
        if self.augment is not None:
            a = self.augment
            velocity = random.randint(-a.velocity_shift, a.velocity_shift) if a.velocity_shift > 0 else 0
            shift = random.randint(-a.transpose, a.transpose) if a.transpose > 0 else 0
            events, end_frame = augment_events(events, end_frame, 1.0, shift, velocity, self.cache.tokenizer_config)
            # テンポの伸縮: 拍の刻みは変わらないので、小節のテンポの条件だけを変える (factor 倍に伸ばす = BPM が 1/factor)
            if a.time_stretch > 0:
                tempo_rate /= random.uniform(1 - a.time_stretch, 1 + a.time_stretch)
        grid = self.annotations.grid(
            song, float(events[:, 0].min(initial=0)), end_frame, self.tokenizer.config.max_bar_beats
        )
        quantized = self.tokenizer.quantize(events, grid)
        bar_ticks = grid.bar_starts * self.tokenizer.ticks_per_beat
        num_bars = int(np.searchsorted(bar_ticks, max(quantized.end_tick - 1, 0), side="right"))
        num_bars = max(1, min(num_bars, grid.num_bars))

        if first_bar is None:
            if self.train:
                if random.random() < self.song_start_prob:
                    first_bar = 0
                else:
                    first_bar = random.randint(0, max(0, num_bars - self.window_bars // 2))
            else:
                first_bar = min(self.val_start_bar, max(0, num_bars - self.window_bars))
        start = max(0, first_bar - self.memory_bars)
        offset = first_bar - start
        total = self.memory_bars + self.window_bars
        window = self.tokenizer.tokenize_window(quantized, grid, start, total, frame_rate=tempo_rate)
        patch_loss = np.zeros(total, dtype=bool)
        patch_loss[offset : offset + self.window_bars] = True
        window["patch_valid"][offset + self.window_bars :] = False
        patch_loss &= window["patch_valid"]

        channel = int(self.cache.channels[song])
        if random.random() < self.channel_dropout:
            channel = 0
        item = {
            "tokens": torch.from_numpy(window["tokens"]),
            "patch_valid": torch.from_numpy(window["patch_valid"]),
            "pedal_state": torch.from_numpy(window["pedal_state"]),
            "bar_beats": torch.from_numpy(window["bar_beats"]),
            "tempo": torch.from_numpy(window["tempo"]),
            "channel": torch.tensor(channel),
            "song_start": torch.tensor(int(start == 0)),
            "patch_loss": torch.from_numpy(patch_loss),
        }
        if self.dynamics_bins:
            index = quantize_dynamics(bar_dynamics(quantized, grid, self.tokenizer, start, total), self.dynamics_bins)
            if self.dynamics_dropout > 0:
                if random.random() < self.dynamics_dropout:
                    index[:] = 0
                else:
                    for column in range(2):
                        if random.random() < self.dynamics_dropout / 2:
                            index[:, column] = 0
            item["dynamics"] = torch.from_numpy(index)
        return item

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.song_window(int(self.songs[index]))
