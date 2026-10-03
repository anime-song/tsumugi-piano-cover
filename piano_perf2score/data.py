"""2 段目 (合成演奏 -> 楽譜) のデータ。楽譜のキャッシュから連続する小節の窓を切り出し、その場で合成演奏を描き出して組にする。

    楽譜のキャッシュ (piano_score.prepare --visible-only) -> 小節の窓 -> Measure に戻す -> render.render -> 演奏の行

演奏は窓の前 context_measures 小節から窓の 1 小節後までを描き出す。窓の最初の小節の MTIME (前の小節の開始からの秒数) と、
演奏のテンポ・強弱の流れが窓の頭で切れないようにするため。
MTIME の正解は描き出した小節線の時刻から作る (演奏の最初の音 = 0 秒)。モデルには丸めた後の時刻 (decode_starts) を渡す。
生成のときに手元にあるのは、生成した MTIME から戻した時刻だけなので、それと同じものを使う。

演奏の行 (1 音 = 1 行、ペダルの踏み・離しも 1 行) は 2 秒のパッチごとに並べる。時刻は 10ms のフレーム。
小節ごとに次の 3 つの時刻 (フレーム) を渡す:
    measure_ref    その小節を予測し始める時点で分かっている基準の時刻。前の小節の開始 (曲の最初の小節は最初の音 = 0)
    measure_start  その小節の開始 (MTIME から戻した時刻)
    measure_spq    4 分音符 1 つ分のフレーム数の見積もり (前の小節の長さから)
曲の最初の小節には前の小節がないので、4 分音符の長さはその小節の本当の長さから求め、学習ではずらす (FIRST_SPQ_NOISE)。
生成ではいくつかのテンポを試して、モデルがいちばん確からしいとするものを選ぶ (generate.transcribe)。
固定のテンポ (♩=120) を仮定すると、遅い曲を倍の速さの音価で書く (16 分を 8 分で、小節を半分の長さで) 誤りになるため。
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from piano_score.data import ScoreAugmentConfig, ScoreCache
from piano_score.tokenizer import PAD, ScoreTokenizer

from .render import Performance, RenderConfig, measure_qpm, render

FRAMES_PER_SECOND = 100
PATCH_FRAMES = 200  # 2 秒
DEFAULT_SPQ = 0.5 * FRAMES_PER_SECOND  # 4 分音符 = 120
FIRST_SPQ_NOISE = 0.1  # 曲の最初の小節の 4 分音符の長さにかけるずれ (対数の標準偏差、学習だけ)


class PerformanceVocab:
    """演奏の行の特徴 (種類・音高・ベロシティ・長さ) を 1 つの埋め込み表で引くための番号。0 は使わない (パディング)"""

    TYPES = ("note", "pedal_down", "pedal_up")

    def __init__(
        self, velocity_bins: int = 16, duration_bins: int = 24, min_duration: float = 0.01, max_duration: float = 10.0
    ) -> None:
        self.velocity_bins = velocity_bins
        self.duration_bins = duration_bins
        self.min_duration = min_duration
        self.max_duration = max_duration
        sizes = {"type": len(self.TYPES), "pitch": 128, "velocity": velocity_bins, "duration": duration_bins}
        self.offset: dict[str, int] = {}
        start = 1
        for name, size in sizes.items():
            self.offset[name] = start
            start += size
        self.size = start

    def rows(self, performance: Performance) -> tuple[np.ndarray, np.ndarray]:
        """演奏の行の特徴 [n, 4] (埋め込み表の番号。使わない列は 0) と打鍵 (ペダル) の時刻のフレーム [n]"""
        o = self.offset
        notes = performance.notes
        features = np.zeros((len(notes) + len(performance.pedal), 4), dtype=np.int64)
        onset = np.zeros(len(features), dtype=np.int64)
        n = len(notes)
        if n:
            features[:n, 0] = o["type"]
            features[:n, 1] = o["pitch"] + notes[:, 2].astype(np.int64)
            velocity = np.clip(notes[:, 3] * self.velocity_bins / 128, 0, self.velocity_bins - 1).astype(np.int64)
            features[:n, 2] = o["velocity"] + velocity
            duration = np.clip(notes[:, 1] - notes[:, 0], self.min_duration, self.max_duration)
            scale = (self.duration_bins - 1) / math.log(self.max_duration / self.min_duration)
            features[:n, 3] = o["duration"] + np.round(np.log(duration / self.min_duration) * scale).astype(np.int64)
            onset[:n] = np.round(notes[:, 0] * FRAMES_PER_SECOND).astype(np.int64)
        if len(performance.pedal):
            down = performance.pedal[:, 1] > 0
            features[n:, 0] = o["type"] + np.where(down, 1, 2)
            onset[n:] = np.round(performance.pedal[:, 0] * FRAMES_PER_SECOND).astype(np.int64)
        order = np.argsort(onset, kind="stable")
        return features[order], onset[order]


def patch_rows(
    features: np.ndarray, onset: np.ndarray, max_rows: int, end_frame: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """行を 2 秒のパッチに並べる。(特徴 [S, R, 4], 時刻 [S, R], 有効 [S, R])。end_frame より後と負の時刻の行は捨てる。
    1 つのパッチに max_rows を超える行があれば、後ろの行を捨てる (非常に速いパッセージだけ)"""
    keep = (onset >= 0) & (onset < end_frame)
    features, onset = features[keep], onset[keep]
    S = max(1, -(-end_frame // PATCH_FRAMES))
    patch = onset // PATCH_FRAMES
    counts = np.bincount(patch, minlength=S)
    R = max(1, min(max_rows, int(counts.max()) if len(counts) else 1))
    out_features = np.zeros((S, R, features.shape[1] if len(features) else 4), dtype=np.int64)
    out_onset = np.zeros((S, R), dtype=np.int64)
    valid = np.zeros((S, R), dtype=bool)
    starts = np.concatenate([[0], np.cumsum(counts)])
    for s in range(S):
        take = min(R, counts[s])
        out_features[s, :take] = features[starts[s] : starts[s] + take]
        out_onset[s, :take] = onset[starts[s] : starts[s] + take]
        valid[s, :take] = True
    return out_features, out_onset, valid


class SynthWindowDataset(Dataset):
    """楽譜の窓と、その場で描き出した合成演奏の組。

    学習用 (split="train") は index を曲番号として受け取り、窓の位置・移調・描き出しの乱数はランダム。
    検証用は曲の冒頭の窓に固定し、描き出しの乱数も曲ごとに固定して、毎回同じ入力で損失を測る。
    """

    def __init__(
        self,
        cache: ScoreCache,
        *,
        split: str,
        window_measures: int,
        context_measures: int = 2,
        render_config: RenderConfig = RenderConfig(),
        augment: ScoreAugmentConfig | None = None,
        song_start_prob: float = 0.2,
        max_rows: int = 160,
        songs: np.ndarray | None = None,
    ) -> None:
        self.cache = cache
        self.tokenizer = ScoreTokenizer(cache.tokenizer_config)
        self.window_measures = window_measures
        self.context_measures = context_measures
        self.render_config = render_config
        self.train = split == "train"
        self.augment = augment if self.train else None
        self.song_start_prob = song_start_prob
        self.max_rows = max_rows
        self.vocab = PerformanceVocab()
        self.songs = np.flatnonzero(~cache.is_val if self.train else cache.is_val) if songs is None else songs
        self.mtime_first = self.tokenizer.ids[("mtime", 0)]
        self.key_first = self.tokenizer.ids[("key", -7)]
        self._tables: dict = {}

    def __len__(self) -> int:
        return len(self.songs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        song = int(self.songs[index])
        num_measures = self.cache.num_measures(song)
        W = self.window_measures
        song_start = not self.train or random.random() < self.song_start_prob
        start = 0 if song_start else random.randint(0, max(0, num_measures - W // 2))
        stop = min(num_measures, start + W)
        first = max(0, start - self.context_measures)
        last = min(num_measures, stop + 1)  # 窓の後の 1 小節も演奏に入れる
        rng = np.random.default_rng(None if self.train else song)

        bodies = self.cache.measure_tokens(song, first, last)
        if self.augment is not None and self.augment.transpose > 0:
            shift = random.randint(-self.augment.transpose, self.augment.transpose)
            if shift:
                bodies = self._transpose(bodies, shift)
        measures, _ = self.tokenizer.decode([[self.mtime_first, *b.tolist()] for b in bodies])
        qpm = measure_qpm(measures, self.cache.song_seconds(song)[first : last + 1])
        performance = render(measures, rng, self.render_config, qpm)

        # MTIME: 描き出した小節線の時刻から。窓の外 (前の文脈) の分も通して求めて、窓の分だけ使う
        bins = self.tokenizer.mtime_bins(performance.measure_starts)
        decoded = self.tokenizer.decode_starts(bins) * FRAMES_PER_SECOND
        lengths = np.array([float(m.length) for m in measures])
        window = range(start - first, stop - first)
        ref = np.array([decoded[k - 1] if k > 0 else 0.0 for k in window])
        spq = np.array([(decoded[k] - decoded[k - 1]) / lengths[k - 1] if k > 0 else DEFAULT_SPQ for k in window])
        if start == 0 and len(decoded) > 1:
            spq[0] = (decoded[1] - decoded[0]) / lengths[0]
            if self.train:
                spq[0] *= math.exp(random.gauss(0.0, FIRST_SPQ_NOISE))

        L = self.tokenizer.config.max_patch_tokens
        tokens = np.full((W, L), PAD, dtype=np.int64)
        for i, k in enumerate(window):
            sequence = np.concatenate([[self.mtime_first + bins[k]], bodies[k]])
            if len(sequence) > L:
                sequence = np.concatenate([sequence[: L - 1], sequence[-1:]])
            tokens[i, : len(sequence)] = sequence
        patch_valid = np.zeros(W, dtype=bool)
        patch_valid[: stop - start] = True

        features, onset = self.vocab.rows(performance)
        end_frame = int(onset.max()) + 1 if len(onset) else 1
        perf_features, perf_onset, perf_valid = patch_rows(features, onset, self.max_rows, end_frame)
        pad = W - len(window)
        return {
            "tokens": torch.from_numpy(tokens),
            "patch_valid": torch.from_numpy(patch_valid),
            "pedal_state": torch.zeros(W, dtype=torch.long),
            "channel": torch.tensor(0),
            "song_start": torch.tensor(int(start == 0)),
            "measure_ref": torch.from_numpy(np.pad(ref, (0, pad))).float(),
            "measure_start": torch.from_numpy(np.pad(decoded[list(window)], (0, pad))).float(),
            "measure_spq": torch.from_numpy(np.pad(spq, (0, pad), constant_values=DEFAULT_SPQ)).float(),
            "perf_features": torch.from_numpy(perf_features),
            "perf_onset": torch.from_numpy(perf_onset),
            "perf_valid": torch.from_numpy(perf_valid),
        }

    def _transpose(self, measures: list[np.ndarray], shift: int) -> list[np.ndarray]:
        """piano_score.data.ScoreWindowDataset._transpose と同じ。綴れない音や 88 鍵の外に出る音があれば移調しない"""
        flat = np.concatenate(measures)
        keys = flat[(flat >= self.key_first) & (flat < self.key_first + 15)] - self.key_first - 7
        cache_key = (shift, frozenset(keys.tolist()))
        if cache_key not in self._tables:
            self._tables[cache_key] = self.tokenizer.transposition(shift, set(cache_key[1]))
        table = self._tables[cache_key]
        if table is None:
            return measures
        moved = [table[m] for m in measures]
        if any((m < 0).any() for m in moved):
            return measures
        return moved


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """楽譜の側は piano_ar.data.collate と同じ (トークン長を最長に詰める)。演奏の側はパッチ数と行数を最大に合わせる"""
    perf_keys = ("perf_features", "perf_onset", "perf_valid")
    out = {key: torch.stack([item[key] for item in batch]) for key in batch[0] if key not in perf_keys}
    used = int((out["tokens"] != PAD).sum(-1).max())
    out["tokens"] = out["tokens"][..., : max(used, 1)]
    S = max(item["perf_valid"].shape[0] for item in batch)
    R = max(item["perf_valid"].shape[1] for item in batch)
    for key in perf_keys:
        first = batch[0][key]
        shape = (len(batch), S, R, *first.shape[2:])
        padded = torch.zeros(shape, dtype=first.dtype)
        for i, item in enumerate(batch):
            value = item[key]
            padded[i, : value.shape[0], : value.shape[1]] = value
        out[key] = padded
    out["perf_patch_valid"] = out["perf_valid"].any(-1)
    return out
