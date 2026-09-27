"""楽譜の事前学習のデータ。prepare.py のキャッシュから連続する小節の窓を切り出して、小節ごとのトークン列にする。

1 小節 = Global の 1 パッチ。各小節のトークン列の先頭に MTIME (演奏上の小節の開始時刻) を入れる。
楽譜だけの事前学習では、テンポ記号どおりの開始時刻に「曲全体のテンポの倍率」と「小節ごとの長さの揺れ」をかけて演奏の代わりにする。
MTIME は曲の冒頭から通して求めてから窓を切り出すので、途中から始まる窓の最初の小節も「前の小節の開始から何秒後か」になり、
生成時に窓をずらしながら続けるときと同じ形になる (曲の冒頭の小節だけ「最初の音から小節の頭まで何秒さかのぼるか」)。
移調は tokenizer.transposition の表で音高と調のトークンを差し替える。
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .config import ScoreTokenizerConfig
from .tokenizer import PAD, ScoreTokenizer


@dataclass(frozen=True)
class ScoreAugmentConfig:
    # 半音単位の移調幅 (±)。移調すると 88 鍵の外に出る音がある曲はその窓だけ移調しない
    transpose: int = 5
    # 曲全体のテンポの倍率の範囲 (対数で一様に選ぶ)。テンポ記号のない楽譜は 120 として見積もっているので広めにする
    tempo_min: float = 0.6
    tempo_max: float = 1.6
    # 小節ごとの長さの揺れ (対数の標準偏差)
    tempo_jitter: float = 0.05


class ScoreCache:
    """prepare.py が作ったキャッシュ。tokens.npy は DataLoader の各ワーカーで遅延して mmap で開く"""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        songs = np.load(self.cache_dir / "songs.npz")
        self.measure_offsets = songs["measure_offsets"]
        self.ids = songs["ids"]
        self.is_val = songs["is_val"]
        measures = np.load(self.cache_dir / "measures.npz")
        self.token_offsets = measures["token_offsets"]
        self.seconds = measures["seconds"]
        self.meta = json.loads((self.cache_dir / "meta.json").read_text(encoding="utf-8"))
        config = dict(self.meta["tokenizer"])
        config["fraction_denominators"] = tuple(config["fraction_denominators"])
        self.tokenizer_config = ScoreTokenizerConfig(**config)
        self._tokens: np.ndarray | None = None

    def num_measures(self, song: int) -> int:
        return int(self.measure_offsets[song + 1] - self.measure_offsets[song])

    def measure_tokens(self, song: int, first: int, last: int) -> list[np.ndarray]:
        """曲の first..last-1 小節のトークン列 (MTIME なし、終端トークン込み)"""
        if self._tokens is None:
            self._tokens = np.load(self.cache_dir / "tokens.npy", mmap_mode="r")
        base = int(self.measure_offsets[song])
        offsets = self.token_offsets[base + first : base + last + 1]
        flat = np.asarray(self._tokens[offsets[0] : offsets[-1]], dtype=np.int64)
        return np.split(flat, offsets[1:-1] - offsets[0])

    def song_seconds(self, song: int) -> np.ndarray:
        """テンポ記号どおりの各小節の開始時刻 (最初の音 = 0 秒)"""
        return self.seconds[self.measure_offsets[song] : self.measure_offsets[song + 1]].astype(np.float64)

    def __getstate__(self) -> dict:
        # mmap を pickle するとデータごとコピーされるので、ワーカーには開く前の状態で渡す
        state = self.__dict__.copy()
        state["_tokens"] = None
        return state


def vary_tempo(seconds: np.ndarray, scale: float, jitter: float, rng: random.Random) -> np.ndarray:
    """各小節の開始時刻に、全体の倍率 scale と小節ごとの揺れ (対数の標準偏差 jitter) をかける"""
    durations = np.diff(seconds) * scale
    if jitter > 0:
        durations *= np.exp(np.array([rng.gauss(0.0, jitter) for _ in durations]))
    return np.concatenate([[seconds[0] * scale], seconds[0] * scale + np.cumsum(durations)])


class ScoreWindowDataset(Dataset):
    """曲から連続する window_measures 小節を切り出す。

    学習用 (split="train") は index を曲番号として受け取り、窓の位置と移調・テンポはランダム。曲の選び方は make_sampler の
    重みで決める。検証用は曲の冒頭の窓に固定し、移調もテンポの変化もかけずに、毎回同じ入力で損失を測る。
    """

    def __init__(
        self,
        cache: ScoreCache,
        *,
        split: str,
        window_measures: int,
        augment: ScoreAugmentConfig | None = None,
        song_start_prob: float = 0.1,
    ) -> None:
        self.cache = cache
        self.tokenizer = ScoreTokenizer(cache.tokenizer_config)
        self.window_measures = window_measures
        self.train = split == "train"
        self.augment = augment if self.train else None
        self.song_start_prob = song_start_prob
        self.songs = np.flatnonzero(~cache.is_val if self.train else cache.is_val)
        self.key_first = self.tokenizer.ids[("key", -7)]
        self.mtime_first = self.tokenizer.ids[("mtime", 0)]
        self._tables: dict[tuple[int, frozenset[int]], np.ndarray | None] = {}

    def __len__(self) -> int:
        return len(self.songs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        song = int(self.songs[index])
        num_measures = self.cache.num_measures(song)
        W = self.window_measures
        song_start = not self.train or random.random() < self.song_start_prob
        # 曲の終わり (EOS) も学習できるよう、窓の後半が曲の外にはみ出す位置まで選ぶ
        start = 0 if song_start else random.randint(0, max(0, num_measures - W // 2))
        stop = min(num_measures, start + W)
        measures = self.cache.measure_tokens(song, start, stop)

        seconds = self.cache.song_seconds(song)[:stop]
        if self.augment is not None:
            a = self.augment
            scale = math.exp(random.uniform(math.log(a.tempo_min), math.log(a.tempo_max)))
            seconds = vary_tempo(seconds, scale, a.tempo_jitter, random)
            shift = random.randint(-a.transpose, a.transpose) if a.transpose > 0 else 0
            if shift:
                measures = self._transpose(measures, shift)
        mtime = self.tokenizer.mtime_bins(seconds)[start:stop]

        L = self.tokenizer.config.max_patch_tokens
        tokens = np.full((W, L), PAD, dtype=np.int64)
        for i, (bin_index, body) in enumerate(zip(mtime, measures)):
            sequence = np.concatenate([[self.mtime_first + bin_index], body])
            if len(sequence) > L:  # 上限を超える小節 (全体の 0.03%) は切り詰めて終端トークンで閉じる
                sequence = np.concatenate([sequence[: L - 1], sequence[-1:]])
            tokens[i, : len(sequence)] = sequence
        patch_valid = np.zeros(W, dtype=bool)
        patch_valid[: stop - start] = True
        return {
            "tokens": torch.from_numpy(tokens),
            "patch_valid": torch.from_numpy(patch_valid),
            # piano_ar のモデルをそのまま使うための入力 (楽譜ではペダルの状態とチャンネルは使わない)
            "pedal_state": torch.zeros(W, dtype=torch.long),
            "channel": torch.tensor(0),
            "song_start": torch.tensor(int(start == 0)),
        }

    def _transpose(self, measures: list[np.ndarray], shift: int) -> list[np.ndarray]:
        """窓のトークンを shift 半音移調する。綴れない音や 88 鍵の外に出る音があれば移調しない"""
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


def make_sampler(dataset: ScoreWindowDataset, num_samples: int) -> WeightedRandomSampler:
    """曲の選ばれやすさを小節数に比例させる (長い曲ほど多くの窓を取る)"""
    offsets = dataset.cache.measure_offsets
    weights = (offsets[dataset.songs + 1] - offsets[dataset.songs]).astype(np.float64)
    return WeightedRandomSampler(torch.from_numpy(weights), num_samples=num_samples, replacement=True)
