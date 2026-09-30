"""カバーの編曲の性質 (合いの手・オブリガート・音域の広さ) を、パッチごとに測る。

デコーダの条件 (強弱と音の多さの後ろの列) と Planner の正解に使う。原曲のメロディ (tsumugi の melody) とカバーを、
アラインメントで原曲の時刻にそろえて比べる。
    fill   メロディが鳴っていない間の、右手 (中央ド以上) の onset の数 / 秒 (合いの手やオブリの量)。
           パッチの中のメロディの隙間が FILL_MIN_GAP 秒より短ければ測れない
    above  カバーが弾いているメロディの音のうち、それより上に音を重ねている割合 (メロディが最高音ではなく、
           上にオブリが乗っている)。弾いているメロディの音が ABOVE_MIN_NOTES より少なければ測れない
    span   パッチの中の最高音と最低音の差 (音域の広がり)。音が SPAN_MIN_NOTES より少なければ測れない
強弱と音の多さ (曲の中で標準化) と違い、全カバーの 2 秒ごとの値の平均と標準偏差で標準化する。
演奏者や曲ごとの弾き方の違い (合いの手の多い人・少ない人) をそのまま残して、生成で強めたり弱めたりできるようにする。
"""

from __future__ import annotations

import numpy as np

from piano_ar.tokenizer import KIND, KIND_NOTE, ONSET, PITCH

from .source import INSTRUMENTS, ROW_A, ROW_B, ROW_D, ROW_ONSET, ROW_TYPE, TYPE_NOTE

ARRANGEMENT_NAMES = ("fill", "above", "span")
MELODY = INSTRUMENTS.index("melody")
MIDDLE_C = 60
FILL_MIN_GAP = 0.5
ABOVE_MIN_NOTES = 2
SPAN_MIN_NOTES = 4
MATCH_SECONDS = 0.05
MIN_MELODY_FRAMES = 10
# 標準化の (平均, 標準偏差)。fill は log1p(1 秒あたりの数)、above は割合の平方根、span は半音。
# 学習用のカバー全体 (約 2000 本) の 2 秒ごとの値から測った
ARRANGEMENT_STATS = ((1.33, 0.65), (0.07, 0.19), (38.0, 11.7))


def _melody_gap(rows: np.ndarray, length: int, first: float, last: float) -> tuple[np.ndarray, np.ndarray]:
    """原曲のメロディの行と、first から last の間でメロディが鳴っていないフレーム [length]"""
    source_notes = rows[rows[:, ROW_TYPE] == TYPE_NOTE]
    melody = source_notes[source_notes[:, ROW_D] == MELODY]
    sounding = np.zeros(length, dtype=bool)
    for start, duration in melody[:, [ROW_ONSET, ROW_B]]:
        sounding[max(int(start), 0) : int(start + max(duration, MIN_MELODY_FRAMES))] = True
    gap = ~sounding
    gap[: max(int(first), 0)] = False
    gap[int(last) + 1 :] = False
    return melody, gap


def _gap_seconds(gap: np.ndarray, bounds: np.ndarray, frame_rate: int) -> np.ndarray:
    edges = np.clip(np.round(bounds).astype(np.int64), 0, len(gap))
    return np.diff(np.concatenate([[0], np.cumsum(gap)])[edges]) / frame_rate


def measurable(rows: np.ndarray, num_patches: int, patch_frames: int, frame_rate: int) -> np.ndarray:
    """生成用: 原曲だけから分かる、パッチごとに各列を測れるか [P, 3]。学習では測れない所は条件が「指定なし」
    なので、生成でも Planner の予測をそこでは渡さない (メロディの隙間がないパッチに fill を指定しない)"""
    bounds = np.arange(num_patches + 1) * patch_frames
    length = int(max(bounds[-1], rows[:, ROW_ONSET].max() if len(rows) else 0)) + 2
    melody, gap = _melody_gap(rows, length, 0, length)
    notes = np.bincount(
        np.clip(melody[:, ROW_ONSET] // patch_frames, 0, num_patches).astype(np.int64), minlength=num_patches + 1
    )[:num_patches]
    return np.stack(
        [
            _gap_seconds(gap, bounds, frame_rate) >= FILL_MIN_GAP,
            notes >= ABOVE_MIN_NOTES,
            np.ones(num_patches, dtype=bool),
        ],
        axis=1,
    )


def arrangement_values(
    events: np.ndarray, mapped: np.ndarray, rows: np.ndarray, bounds: np.ndarray, frame_rate: int
) -> np.ndarray:
    """原曲の時刻で区切った区間 (bounds [P + 1]、原曲のフレーム) ごとの fill / above / span を標準化した値 [P, 3]。

    events はカバーのイベント配列、mapped はその音 (KIND_NOTE の行) の原曲の時刻、rows は原曲の行。
    Planner の正解なら bounds は原曲のパッチの境界、デコーダの条件ならカバーのパッチの境界を原曲の時刻に写したもの。
    測れない所は NaN。
    """
    P = len(bounds) - 1
    out = np.full((P, 3), np.nan)
    notes = events[events[:, KIND] == KIND_NOTE]
    if len(notes) < 10 or P <= 0:
        return out
    pitch, onset = notes[:, PITCH].astype(np.int64), notes[:, ONSET]
    order = np.argsort(mapped, kind="stable")
    mapped, pitch, onset = mapped[order], pitch[order], onset[order]
    first, last = mapped[0], mapped[-1]

    length = int(max(bounds[-1], last, rows[:, ROW_ONSET].max() if len(rows) else 0)) + 2
    # カバーが弾いている範囲の、メロディが鳴っていない所
    melody, gap = _melody_gap(rows, length, first, last)
    gap_seconds = _gap_seconds(gap, bounds, frame_rate)

    def bucket(times: np.ndarray) -> np.ndarray:
        index = np.searchsorted(bounds, times, side="right") - 1
        return np.where((index >= 0) & (index < P) & (times < bounds[-1]), index, -1)

    # fill: 隙間に入った右手の onset (和音は 1 つと数える)
    in_gap = gap[np.clip(mapped.astype(np.int64), 0, length - 1)] & (pitch >= MIDDLE_C)
    _, unique = np.unique(onset[in_gap], return_index=True)
    fill_index = bucket(mapped[in_gap][unique])
    count = np.bincount(fill_index[fill_index >= 0], minlength=P)
    enough = gap_seconds >= FILL_MIN_GAP
    out[enough, 0] = np.log1p(count[enough] / gap_seconds[enough])

    # above: メロディの音の近くで同じ音名の音を弾いていて、その上にも音があるか
    tol = MATCH_SECONDS * frame_rate
    melody = melody[(melody[:, ROW_ONSET] >= first) & (melody[:, ROW_ONSET] <= last)]
    lo = np.searchsorted(mapped, melody[:, ROW_ONSET] - tol)
    hi = np.searchsorted(mapped, melody[:, ROW_ONSET] + tol, side="right")
    covered, above = np.zeros(P), np.zeros(P)
    for (start, melody_pitch), a, b, index in zip(
        melody[:, [ROW_ONSET, ROW_A]], lo, hi, bucket(melody[:, ROW_ONSET].astype(np.float64))
    ):
        if a == b or index < 0:
            continue
        near = pitch[a:b]
        same = near[near % 12 == melody_pitch % 12]
        if len(same):
            covered[index] += 1
            above[index] += near.max() > same.max()
    enough = covered >= ABOVE_MIN_NOTES
    out[enough, 1] = np.sqrt(above[enough] / covered[enough])

    # span: 区間の中の最高音 - 最低音
    index = bucket(mapped)
    inside = index >= 0
    count = np.bincount(index[inside], minlength=P)
    high, low = np.full(P, -1), np.full(P, 128)
    np.maximum.at(high, index[inside], pitch[inside])
    np.minimum.at(low, index[inside], pitch[inside])
    enough = count >= SPAN_MIN_NOTES
    out[enough, 2] = high[enough] - low[enough]

    mean, std = np.array(ARRANGEMENT_STATS).T
    return (out - mean) / std


def map_to_source(times: np.ndarray, align: np.ndarray, align_step: int, factor: float = 1.0) -> np.ndarray:
    """カバーの時刻 (フレーム) を原曲の時刻に写す。factor は学習の時間伸縮 (カバーと原曲の両方にかけたもの)"""
    return factor * np.interp(times / factor / align_step, np.arange(len(align)), align)
