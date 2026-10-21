"""拍の上の演奏 -> 楽譜で、楽譜のテンポを SheetSage2-Refined の拍から決める。

モデルの入力は「1 拍 = BEAT_SECONDS」の拍の時刻なので、モデルは本当のテンポを知らず、メトロノーム記号
(♩ = 80 など) は当て推量になる。そこで小節ごとの本当の長さ (原曲の小節線の秒、beat_performance の
performance.bar_seconds) から 4 分音符のテンポを求める。小節線の秒は少し揺れる (同じテンポでも小節ごとに 1〜2%) ので、
前の区間のテンポから TEMPO_TOLERANCE 以内の小節はまとめて 1 つのテンポの区間にし、
    - 再生用のテンポ (Measure.tempos、書き出しの <sound tempo>) は区間の頭にだけ、区間の小節のテンポの中央値で置き
    - 見せるメトロノーム記号は曲の頭と、MARK_BARS 小節以上続く区間で記号の値が変わるところに置く (モデルの記号は消す)。
      短い区間 (拍の取り違えの 1 小節や rit.) は再生のテンポだけにする
複合拍子 (x/8 で拍子の数が 3 の倍数) は付点 4 分音符の記号にする。
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from piano_score.score import Measure, metronome_bpm

from .render import Performance

TEMPO_TOLERANCE = 0.05  # 区間のテンポ (中央値) からこの割合以内の小節は同じテンポとみなす
MARK_BARS = 4  # メトロノーム記号を置く区間の小節の数の下限


def tempo_segments(qpms: list[float]) -> list[tuple[int, int, float]]:
    """小節のテンポの並びを、近い値が続く区間に分ける。[(最初の番号, 終わりの番号 (含まない), 区間のテンポ)]。
    区間のテンポはその小節のテンポの中央値で、次の小節がそこから TEMPO_TOLERANCE より離れたら新しい区間にする"""
    segments, start = [], 0
    for k in range(1, len(qpms) + 1):
        if k == len(qpms) or abs(qpms[k] / np.median(qpms[start:k]) - 1) > TEMPO_TOLERANCE:
            segments.append((start, k, float(np.median(qpms[start:k]))))
            start = k
    return segments


def _metronome(measure: Measure, qpm: float) -> str:
    beats, beat_type = measure.time_signature
    if beat_type == 8 and beats % 3 == 0:
        return f"metronome_quarter._{metronome_bpm(qpm / 1.5)}"
    return f"metronome_quarter_{metronome_bpm(qpm)}"


def apply_bar_tempos(measures: list[Measure], performance: Performance, skipped: int = 0) -> float | None:
    """measures[k] (原曲の小節 performance.bar_seconds[k + skipped] から始まる小節) のテンポをその場で決め、
    曲の頭のメトロノーム記号の 4 分音符のテンポを返す。小節の秒が分からなければ何もせず None"""
    seconds = np.asarray(performance.bar_seconds, dtype=np.float64)
    known, qpms = [], []
    for k, measure in enumerate(measures):
        i = k + skipped
        if i + 1 < len(seconds) and seconds[i + 1] > seconds[i]:
            known.append(k)
            qpms.append(60.0 * float(measure.length) / (seconds[i + 1] - seconds[i]))
    if not qpms:
        return None
    for measure in measures:
        measure.directions = [d for d in measure.directions if not d[1].startswith("metronome_")]
        measure.tempos = []  # モデルが見積もった小節の中のテンポの変化も捨てる
    segments = tempo_segments(qpms)
    long = [(a, b, qpm) for a, b, qpm in segments if b - a >= MARK_BARS]
    # 曲の頭の記号は最初の長い区間のテンポ (なければ全体の中央値)。短い区間で始まっても記号は曲の頭に置く
    first = long[0][2] if long else float(np.median(qpms))
    measures[0].directions.insert(0, (Fraction(0), _metronome(measures[0], first)))
    shown = measures[0].directions[0][1]
    for a, b, qpm in segments:
        measure = measures[known[a]]
        measure.tempos = [(Fraction(0), qpm)]
        name = _metronome(measure, qpm)
        if b - a >= MARK_BARS and name != shown and known[a] > 0:
            measure.directions.insert(0, (Fraction(0), name))
            shown = name
    return first
