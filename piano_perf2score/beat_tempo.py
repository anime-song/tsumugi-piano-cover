"""拍の上の演奏 -> 楽譜で、楽譜のテンポを SheetSage2-Refined の拍から決める。

モデルの入力は「1 拍 = BEAT_SECONDS」の拍の時刻なので、モデルは本当のテンポを知らず、メトロノーム記号
(♩ = 80 など) は当て推量になる。そこで小節ごとの本当の長さ (原曲の小節線の秒、beat_performance の
performance.bar_seconds) から 4 分音符のテンポを求め、
    - 再生用に各小節の頭のテンポ (Measure.tempos、書き出しの <sound tempo>) をその値にし
    - 見せるメトロノーム記号は曲の頭に 1 つ、小節のテンポの中央値を METRONOME_BPMS に丸めて置く (モデルの記号は消す)
複合拍子 (x/8 で拍子の数が 3 の倍数) は付点 4 分音符の記号にする。
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from piano_score.score import Measure, metronome_bpm

from .render import Performance


def apply_bar_tempos(measures: list[Measure], performance: Performance, skipped: int = 0) -> float | None:
    """measures[k] (原曲の小節 performance.bar_seconds[k + skipped] から始まる小節) のテンポをその場で決め、
    メトロノーム記号の 4 分音符のテンポを返す。小節の秒が分からなければ何もせず None"""
    seconds = np.asarray(performance.bar_seconds, dtype=np.float64)
    qpms = []
    for k, measure in enumerate(measures):
        i = k + skipped
        if i + 1 < len(seconds) and seconds[i + 1] > seconds[i]:
            qpm = 60.0 * float(measure.length) / (seconds[i + 1] - seconds[i])
            measure.tempos = [(Fraction(0), qpm)]  # モデルが見積もった小節の中のテンポの変化も捨てる
            qpms.append(qpm)
    if not qpms:
        return None
    for measure in measures:
        measure.directions = [d for d in measure.directions if not d[1].startswith("metronome_")]
    beats, beat_type = measures[0].time_signature
    compound = beat_type == 8 and beats % 3 == 0
    qpm = float(np.median(qpms))
    if compound:
        name = f"metronome_quarter._{metronome_bpm(qpm / 1.5)}"
    else:
        name = f"metronome_quarter_{metronome_bpm(qpm)}"
    measures[0].directions.insert(0, (Fraction(0), name))
    return qpm
