"""スイングの曲の、拍の上の演奏 -> 楽譜 (V2 のカバー向け)。

スイングの曲のカバーは、8 分の裏の音が拍の 1/2 (6 刻み) ではなく 7〜9 刻みにあり、その位置も曲の中で揺れる
(JAM GEM JUMP! のテイクでは 16 小節ごとに 6・7・8・9 刻みの割合が入れ替わる)。拍の頭の 1 刻み前後 (11・1 刻み) に
ずれた音も多い。そのまま入れると、モデルは同じ裏拍を 8 分・3 連符・16 分で書き分けてしまい、楽譜が崩れる。
学習の楽譜にもスイングの曲はある (beat_songs の 2%) が、描き出しのスイングは曲を通して一定の比なので、揺れには弱い。

そこで入力の側でスイングをまっすぐにして、楽譜にはスイングの指定 (Swing) を書く。
    1. 小節ごとに、前後 SWING_WINDOW 小節の音の拍の中の位置を数え、遅れた裏 (7・8 刻み) が、まっすぐの裏と
       16 分 (6・3 刻み) より多い小節をスイングとする。MIN_RUN 小節より短い区間は前後に合わせる
    2. スイングの小節では、7・8 刻みの音を 6 刻み (8 分の裏) へ、9 刻みも 16 分 (3 刻み) がほとんどなければ 6 刻みへ、
       11・1 刻みの音は拍の頭へ寄せる
    3. 書き出す前に、スイングの小節の Measure.swing を "8th" にする (書き出しで Swing / Straight の文字になる)
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from piano_score.score import Measure

from .beat_input import BEAT_SECONDS
from .render import Performance

TICKS = 12  # 1 拍の刻み (beat_input の丸めと同じ)
SWING_WINDOW = 4
MIN_RUN = 4
MIN_LATE = 8  # スイングとみなす遅れた裏の音の数 (窓の中)


def _beat_ticks(performance: Performance) -> tuple[np.ndarray, np.ndarray]:
    """各音の (小節の番号, 小節の頭からの刻み)"""
    starts = np.asarray(performance.measure_starts, dtype=np.float64)
    onsets = performance.notes[:, 0]
    bars = np.clip(np.searchsorted(starts, onsets + 1e-6, side="right") - 1, 0, len(starts) - 1)
    ticks = np.round((onsets - starts[bars]) / BEAT_SECONDS * TICKS).astype(np.int64)
    return bars, ticks


def swing_measures(performance: Performance) -> np.ndarray:
    """スイングの小節 [M] (performance.measure_starts と同じ並び)"""
    count = len(performance.measure_starts)
    if not count or not len(performance.notes):
        return np.zeros(count, dtype=bool)
    bars, ticks = _beat_ticks(performance)
    counts = np.zeros((count, TICKS), dtype=np.int64)
    np.add.at(counts, (bars, ticks % TICKS), 1)
    kernel = np.ones(2 * SWING_WINDOW + 1, dtype=np.int64)
    window = np.stack([np.convolve(counts[:, k], kernel, mode="same") for k in range(TICKS)], axis=1)
    late = window[:, 7] + window[:, 8]
    swing = (late >= MIN_LATE) & (late > window[:, 6] + window[:, 3])
    # 短い区間は前後に合わせる (Swing / Straight の文字が細かく入れ替わらないように)
    changes = np.flatnonzero(np.diff(swing.astype(np.int8))) + 1
    bounds = [0, *changes.tolist(), count]
    for a, b in pairwise(bounds):
        if b - a < MIN_RUN and (a, b) != (0, count):
            swing[a:b] = swing[a - 1] if a > 0 else swing[b]
    return swing


def straighten_swing(performance: Performance) -> np.ndarray:
    """スイングの小節の音をまっすぐの位置へ寄せ (その場で書き換え、打鍵の順に並べ直す)、スイングの小節 [M] を返す。
    メロディの印 (performance.melody) も同じ順に並べ直す"""
    swing = swing_measures(performance)
    if not swing.any():
        return swing
    bars, ticks = _beat_ticks(performance)
    counts = np.zeros(TICKS, dtype=np.int64)
    np.add.at(counts, ticks[swing[bars]] % TICKS, 1)
    in_beat = ticks % TICKS
    target = in_beat.copy()
    target[(in_beat == 7) | (in_beat == 8)] = 6
    if counts[3] * 4 < counts[9]:
        target[in_beat == 9] = 6
    target[in_beat == 11] = TICKS
    target[in_beat == 1] = 0
    move = swing[bars] & (target != in_beat)
    shift = (target - in_beat) * BEAT_SECONDS / TICKS
    notes = performance.notes
    notes[move, 0] += shift[move]
    notes[:, 1] = np.maximum(notes[:, 1], notes[:, 0] + BEAT_SECONDS / TICKS)
    order = np.argsort(notes[:, 0], kind="stable")
    performance.notes = notes[order]
    if len(performance.melody) == len(order):
        performance.melody = performance.melody[order]
    return swing


def mark_swing(measures: list[Measure], swing: np.ndarray, skipped: int = 0) -> int:
    """measures[k] (performance.measure_starts[k + skipped] から始まる小節) のスイングの指定を swing に合わせ、
    スイングにした小節の数を返す。モデルが書いた小節の中のスイングの指定は消す"""
    marked = 0
    for k, measure in enumerate(measures):
        on = bool(swing[k + skipped]) if k + skipped < len(swing) else False
        measure.swing = "8th" if on else "none"
        measure.directions = [d for d in measure.directions if not d[1].startswith("swing_")]
        marked += on
    return marked
