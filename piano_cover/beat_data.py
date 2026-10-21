"""拍のトークン (piano_ar.beat_tokenizer) のカバーの学習データ。

カバーは piano_cover.prepare_beats で原曲の階層にそろえた拍の上でトークン化し (1 パッチ = カバーの 1 小節)、原曲は
原曲の拍の上で小節ごとのパッチにする (source_features の grid)。どちらの位置も「拍の番号 x 刻み」にそろえるので、
カバーの小節 p が原曲のどこに当たるかは、小節の頭の拍に対応する原曲の拍の番号で決まる (align)。生成では
カバーの小節 = 原曲の小節 (対応は恒等) として弾く。

返す 1 件の中身は CoverWindowDataset と同じ名前に、次を足したもの:
    bar_beats / tempo [P]    小節の拍数とテンポの段階 (デコーダの Global の条件)
    align [P, 2]             カバーの小節 p の頭に対応する原曲の刻みと、頭 + max_bar_ticks x (原曲の拍 / カバーの拍)。
                             CoverModel は小節の中の位置 t を「頭 + (align[1] - align[0]) x t / max_bar_ticks」に写す
    align_bar [P]            小節の頭が原曲の何小節目に当たるか (小数。Global の cross-attention の位置)
    align_center [P]         小節の中央が当たる原曲の小節 (Local が見る原曲のパッチの中心)
    src_patch_start [S]      原曲の各小節の頭の刻み
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from piano_ar.beat_data import bar_dynamics
from piano_ar.beat_tokenizer import BeatGrid, BeatTokenizer
from piano_ar.config import BeatTokenizerConfig
from piano_ar.data import AugmentConfig, quantize_dynamics, shift_velocity, transpose_events
from piano_ar.tokenizer import KIND, KIND_NOTE, ONSET, PianoTokenizer

from .arrangement import ARRANGEMENT_NAMES, arrangement_values
from .config import CoverConfig
from .data import MELODY_ID, CoverCache, empty_source, source_tensors
from .source import (
    ROW_D,
    ROW_TYPE,
    TYPE_BEAT,
    TYPE_CHORD,
    TYPE_KEY,
    TYPE_NOTE,
    SourceVocab,
    is_sheetsage,
    source_features,
    transpose_rows,
)

SPLITS = {"train": 0, "val": 1, "test": 2}
DYNAMICS_MIN_NOTES = 3


class BeatCoverCache(CoverCache):
    """prepare_beats が作ったキャッシュ (CoverCache に、カバーと原曲の拍を足したもの)。
    piano_cover.prepare_frames の frames.npy / frames_index.npz があれば、原曲の音声の特徴も読める (source_frames)"""

    def __init__(self, cache_dir: str | Path) -> None:
        super().__init__(cache_dir)
        cover = np.load(self.cache_dir / "cover_beats.npz")
        self.beat_offsets = cover["offsets"]
        self.beat_times = cover["times"]
        self.beat_down = cover["downbeat"]
        self.beat_source = cover["source_beat"]
        self.beat_ok = cover["ok"]
        source = np.load(self.cache_dir / "source_beats.npz")
        self.source_beat_offsets = source["offsets"]
        self.source_beats = source["beats"]
        self.source_down = source["downbeat"]
        self.has_frames = (self.cache_dir / "frames_index.npz").exists()
        if self.has_frames:
            index = np.load(self.cache_dir / "frames_index.npz")
            self.frame_offsets = index["offsets"]
            self.frame_hz = float(index["frame_rate"])
        self._frames = None  # DataLoader の各ワーカーで遅延して mmap で開く

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state["_frames"] = None
        return state

    def source_frames(self, source: int) -> np.ndarray | None:
        """原曲の音声の特徴 [フレーム, D] (float16、frame_hz、0 番目は音声の頭)。ない原曲は None"""
        if not self.has_frames:
            return None
        lo, hi = self.frame_offsets[source], self.frame_offsets[source + 1]
        if hi <= lo:
            return None
        if self._frames is None:
            self._frames = np.load(self.cache_dir / "frames.npy", mmap_mode="r")
        return np.asarray(self._frames[lo:hi])

    def has_beats(self, cover: int) -> bool:
        source = int(self.source_index[cover])
        return (
            self.beat_offsets[cover + 1] - self.beat_offsets[cover] >= 4
            and self.source_beat_offsets[source + 1] - self.source_beat_offsets[source] >= 4
        )

    def cover_beats(self, cover: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lo, hi = self.beat_offsets[cover], self.beat_offsets[cover + 1]
        return (
            self.beat_times[lo:hi].astype(np.float64),
            self.beat_down[lo:hi],
            self.beat_source[lo:hi].astype(np.float64),
            self.beat_ok[lo:hi],
        )

    def source_grid(self, source: int, end_frame: int, max_bar_beats: int) -> BeatGrid | None:
        lo, hi = self.source_beat_offsets[source], self.source_beat_offsets[source + 1]
        beats = self.source_beats[lo:hi].astype(np.float64)
        return BeatGrid.from_frames(beats, beats[self.source_down[lo:hi]], 0.0, end_frame, max_bar_beats)


def _lead(grid: BeatGrid, first_beat: float) -> int:
    """grid の拍のうち、元の注釈の最初の拍の番号 (前に延ばした拍の数)"""
    return int(np.argmin(np.abs(grid.beats - first_beat)))


def extend_positions(values: np.ndarray, lead: int, total: int) -> np.ndarray:
    """注釈の拍ごとの値 (原曲の拍の番号) を、前後に延ばした grid の拍 (total 個) に広げる (外は 1 拍ずつ進める)"""
    out = np.empty(total, dtype=np.float64)
    n = len(values)
    index = np.arange(total) - lead
    inside = (index >= 0) & (index < n)
    out[inside] = values[index[inside]]
    out[index < 0] = values[0] + index[index < 0]
    out[index >= n] = values[-1] + (index[index >= n] - (n - 1))
    return out


class BeatCoverDataset(Dataset):
    """カバーから小節の窓を切り出し、原曲の全曲 (小節ごとのパッチ) と、カバーの各小節が当たる原曲の位置を付けて返す。

    CoverWindowDataset の拍のトークン版。ロールアウトとずらし学習 (drift) は使わない。
    pretraining (piano_ar.beat_data.BeatWindowDataset) を渡すと、index がカバーの数以上のときは事前学習の曲を原曲なしで返す。
    """

    def __init__(
        self,
        cache: BeatCoverCache,
        *,
        split: str,
        window_bars: int,
        cover_config: CoverConfig,
        tokenizer_config: BeatTokenizerConfig | None = None,
        augment: AugmentConfig | None = None,
        song_start_prob: float = 0.1,
        channel_dropout: float = 0.15,
        source_dropout: float = 0.1,
        structure_dropout: float = 0.2,
        melody_dropout: float = 0.0,
        frame_dropout: float = 0.0,
        pretraining=None,
        memory_bars: int = 0,
        val_start_bar: int = 0,
        dynamics_bins: int = 0,
        dynamics_dropout: float = 0.2,
        arrangement: bool = False,
        sync_mask: bool = False,
    ) -> None:
        self.cache = cache
        self.tokenizer = BeatTokenizer(tokenizer_config)
        self.source_tokenizer = PianoTokenizer(cache.tokenizer_config)  # 原曲の行の長さ・ベロシティの番号
        self.vocab = SourceVocab(cache.tokenizer_config)
        self.cover_config = cover_config
        self.window_bars = window_bars
        self.memory_bars = memory_bars
        self.val_start_bar = val_start_bar
        self.train = split == "train"
        self.augment = augment if self.train else None
        self.song_start_prob = song_start_prob
        self.channel_dropout = channel_dropout if self.train else 0.0
        self.source_dropout = source_dropout
        self.structure_dropout = structure_dropout if self.train else 0.0
        self.melody_dropout = melody_dropout if self.train else 0.0
        # 原曲の音声の特徴を使うか (cover_config.frame_dims) と、学習で外す確率。音声は移調できないので、
        # 特徴を使う窓は移調せず、外した窓だけ移調する
        self.use_frames = bool(cover_config.frame_dims) and cache.has_frames
        self.frame_dropout = frame_dropout if self.train else 0.0
        self.dynamics_bins = dynamics_bins
        self.dynamics_dropout = dynamics_dropout if self.train else 0.0
        self.arrangement = arrangement and dynamics_bins > 0
        self.sync_mask = sync_mask
        self.pretraining = pretraining
        self.covers = np.array(
            [c for c in np.flatnonzero(cache.split == SPLITS[split]) if cache.has_beats(int(c))], dtype=np.int64
        )

    def __len__(self) -> int:
        return len(self.covers) + (len(self.pretraining) if self.pretraining is not None else 0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index >= len(self.covers):
            item = self.pretraining[index - len(self.covers)]
            P = len(item["tokens"])
            item["align"] = torch.zeros(P, 2)
            item["align_bar"] = torch.zeros(P)
            item["align_center"] = torch.zeros(P, dtype=torch.long)
            item["has_source"] = torch.tensor(False)
            item.update(empty_source())
            item["src_patch_start"] = torch.zeros(1, dtype=torch.long)
            if self.arrangement and "dynamics" in item:
                extra = torch.zeros(P, len(ARRANGEMENT_NAMES), dtype=item["dynamics"].dtype)
                item["dynamics"] = torch.cat([item["dynamics"], extra], dim=1)
            return item
        return self.cover_item(int(self.covers[index]))

    def cover_item(self, cover: int, first_bar: int | None = None) -> dict[str, torch.Tensor]:
        cache = self.cache
        tok = self.tokenizer
        tpb = tok.ticks_per_beat
        F_ = tok.config.max_bar_ticks
        fr = cache.tokenizer_config.frame_rate
        source = int(cache.source_index[cover])
        events = cache.cover_events(cover)
        end_frame = int(cache.end_frames[cover])
        rows = cache.source_rows(source)
        source_end = int(cache.source_end_frames[source])

        frames = cache.source_frames(source) if self.use_frames and random.random() >= self.frame_dropout else None
        tempo_rate = float(fr)
        if self.augment is not None:
            a = self.augment
            if a.transpose > 0 and frames is None:
                shift = random.randint(-a.transpose, a.transpose)
                events = transpose_events(events, shift, cache.tokenizer_config)
                rows = transpose_rows(rows, shift)
            if a.velocity_shift > 0:
                events = shift_velocity(events, random.randint(-a.velocity_shift, a.velocity_shift))
            if a.time_stretch > 0:  # 拍の刻みは変わらないので、テンポの条件だけを伸縮する
                tempo_rate /= random.uniform(1 - a.time_stretch, 1 + a.time_stretch)

        # カバーの拍 (原曲の階層にそろえたもの) と、それぞれが当たる原曲の拍
        times, down, source_beat, ok = cache.cover_beats(cover)
        first_event = float(events[:, ONSET].min(initial=0))
        grid = BeatGrid.from_frames(times, times[down], first_event, end_frame, tok.config.max_bar_beats)
        source_grid = cache.source_grid(source, source_end, tok.config.max_bar_beats)
        lead = _lead(grid, times[0])
        source_lead = _lead(source_grid, float(cache.source_beats[cache.source_beat_offsets[source]]))
        # カバーの grid の各拍が当たる、原曲の grid の拍の番号 (小数)
        position = extend_positions(source_beat, lead, len(grid.beats)) + source_lead
        beat_ok = np.zeros(len(grid.beats), dtype=bool)
        beat_ok[lead : lead + len(ok)] = ok

        quantized = tok.quantize(events, grid)
        bar_ticks = grid.bar_starts * tpb
        num_bars = max(
            1, min(int(np.searchsorted(bar_ticks, max(quantized.end_tick - 1, 0), side="right")), grid.num_bars)
        )
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
        window = tok.tokenize_window(quantized, grid, start, total, frame_rate=tempo_rate)
        patch_loss = np.zeros(total, dtype=bool)
        patch_loss[offset : offset + self.window_bars] = True
        window["patch_valid"][offset + self.window_bars :] = False
        patch_loss &= window["patch_valid"]

        # 窓の各小節が当たる原曲の位置
        bars = np.clip(np.arange(start, start + total), 0, grid.num_bars - 1)
        b0, b1 = grid.bar_starts[bars], grid.bar_starts[bars + 1]
        s0, s1 = position[b0], position[np.minimum(b1, len(position) - 1)]
        slope = (s1 - s0) / np.maximum(b1 - b0, 1)
        align = np.stack([s0 * tpb, s0 * tpb + F_ * slope], axis=1).astype(np.float32)
        source_bar_beats = source_grid.bar_starts.astype(np.float64)
        bar_index = np.arange(len(source_bar_beats), dtype=np.float64)
        align_bar = np.interp(s0, source_bar_beats, bar_index).astype(np.float32)
        align_center = np.floor(np.interp((s0 + s1) / 2, source_bar_beats, bar_index)).astype(np.int64)
        if self.sync_mask:
            cumulative = np.concatenate([[0], np.cumsum(beat_ok)])
            rate = (cumulative[b1] - cumulative[b0]) / np.maximum(b1 - b0, 1)
            masked = patch_loss & (rate >= 0.5)
            if masked.any():
                patch_loss = masked

        has_source = random.random() >= self.source_dropout
        if self.arrangement or self.dynamics_bins:
            # カバーの音とパッチの境界を原曲の時刻 (フレーム) に写す (編曲の性質と Planner の正解)
            notes = events[events[:, KIND] == KIND_NOTE]
            note_beats = grid.beat_position(notes[:, ONSET])
            note_source = source_grid.frames_at(np.interp(note_beats, np.arange(len(position)), position))
            bounds = source_grid.frames_at(np.concatenate([s0, s1[-1:]]))
        item_source = empty_source()
        item_source["src_patch_start"] = torch.zeros(1, dtype=torch.long)
        if has_source:
            for row_type in (TYPE_BEAT, TYPE_CHORD, TYPE_KEY):
                if random.random() < self.structure_dropout:
                    rows = rows[rows[:, ROW_TYPE] != row_type]
            if self.melody_dropout > 0:
                tsumugi = (rows[:, ROW_TYPE] == TYPE_NOTE) & (rows[:, ROW_D] == MELODY_ID)
                drop = np.zeros(len(rows), dtype=bool)
                for melody in (tsumugi, is_sheetsage(rows)):
                    if random.random() < self.melody_dropout:
                        drop |= melody
                rows = rows[~drop]
            features = source_features(
                rows, source_end, self.source_tokenizer, self.vocab, self.cover_config.max_source_rows,
                grid=source_grid, ticks_per_beat=tpb, frames=frames,
                frame_hz=cache.frame_hz if frames is not None else 25.0,
                max_frame_rows=self.cover_config.max_frame_rows,
            )  # fmt: skip
            item_source = source_tensors(features)
            item_source["src_patch_start"] = torch.from_numpy(features["patch_start"])
            if self.dynamics_bins:
                S = len(features["features"])
                target = self.plan_target(quantized, grid, position, source_grid, S)
                if self.arrangement:
                    source_bounds = source_grid.frames_at(source_grid.bar_starts[: S + 1].astype(np.float64))
                    values = arrangement_values(events, note_source, rows, source_bounds, fr)
                    target = np.concatenate([target, values.astype(np.float32)], axis=1)
                item_source["plan_target"] = torch.from_numpy(target)

        channel = int(cache.channels[cover])
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
            "align": torch.from_numpy(align),
            "align_bar": torch.from_numpy(align_bar),
            "align_center": torch.from_numpy(align_center),
            "has_source": torch.tensor(has_source),
            "patch_loss": torch.from_numpy(patch_loss),
            **item_source,
        }
        if self.dynamics_bins:
            index = quantize_dynamics(bar_dynamics(quantized, grid, tok, start, total), self.dynamics_bins)
            if self.arrangement:
                values = arrangement_values(events, note_source, rows, bounds, fr)
                index = np.concatenate([index, quantize_dynamics(values, self.dynamics_bins)], axis=1)
            if self.dynamics_dropout > 0:
                if random.random() < self.dynamics_dropout:
                    index[:] = 0
                else:
                    for column in range(index.shape[1]):
                        if random.random() < self.dynamics_dropout / 2:
                            index[:, column] = 0
            item["dynamics"] = torch.from_numpy(index)
        return item

    def plan_target(self, quantized, grid: BeatGrid, position: np.ndarray, source_grid: BeatGrid, S: int) -> np.ndarray:
        return bar_plan_target(quantized, self.tokenizer, position, source_grid, S)


def bar_plan_target(
    quantized, tokenizer: BeatTokenizer, position: np.ndarray, source_grid: BeatGrid, S: int
) -> np.ndarray:
    """原曲の小節ごとの、カバーの強さと音の多さ (1 拍あたりの音の数の log) をカバーの中で標準化したもの [S, 2]
    (piano_cover.data.plan_target の小節版)。position はカバーの拍ごとの原曲の拍の番号。カバーが弾いていない小節と、
    強さを測れない小節は NaN"""
    out = np.full((S, 2), np.nan, dtype=np.float32)
    note = quantized.kind == KIND_NOTE
    if note.sum() < 10:
        return out
    beats = quantized.tick[note] / tokenizer.ticks_per_beat
    source_beats = np.interp(beats, np.arange(len(position)), position)
    bar = np.searchsorted(source_grid.bar_starts[: S + 1], source_beats, side="right") - 1
    inside_note = (bar >= 0) & (bar < S)
    bar, velocity = bar[inside_note], quantized.velocity[note][inside_note].astype(np.float64)
    count = np.bincount(bar, minlength=S).astype(np.float64)
    mean = np.bincount(bar, weights=velocity, minlength=S) / np.maximum(count, 1)
    lo, hi = (int(bar.min()) + 1, int(bar.max())) if len(bar) else (0, 0)
    inside = np.zeros(S, dtype=bool)
    inside[max(lo, 0) : max(hi, 0)] = True
    loud = inside & (count >= DYNAMICS_MIN_NOTES)
    if loud.sum() >= 5 and mean[loud].std() > 0:
        out[loud, 0] = (mean[loud] - mean[loud].mean()) / mean[loud].std()
    density = np.log1p(count / np.maximum(source_grid.bar_beats[:S], 1))[inside]
    if inside.sum() >= 5 and density.std() > 0:
        out[inside, 1] = (density - density.mean()) / density.std()
    return out
