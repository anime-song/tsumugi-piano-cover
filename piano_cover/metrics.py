"""生成したカバーが原曲の時刻とメロディにどれだけ合っているか。

検証の損失は正しいカバーの過去から次を当てるので、生成中に自分の出した過去が原曲からずれていく問題 (exposure bias) は
数字に出ない。生成したものを原曲と直接比べて測る。生成は原曲の時間軸の上なので、出力と原曲の行は同じフレームで比べられる。
"""

from __future__ import annotations

import numpy as np

from piano_ar.tokenizer import KIND, KIND_NOTE, ONSET, PITCH

from .source import INSTRUMENTS, ROW_A, ROW_D, ROW_ONSET, ROW_TYPE, TYPE_NOTE

MELODY = INSTRUMENTS.index("melody")
DRUMS = INSTRUMENTS.index("drums")


def sync_metrics(events: np.ndarray, rows: np.ndarray, frame_rate: int, bin_seconds: float = 5.0) -> dict[str, float]:
    """events は生成したイベント配列、rows は原曲の行 (どちらも原曲の時間軸のフレーム)。

    onset_sync     出力の onset のうち、±30ms 以内に原曲の onset (ドラム以外) があるものの割合
    drift_std_ms   出力の onset と最寄りの原曲の onset の差の、bin_seconds ごとの中央値のばらつき。
                   テンポが少しずつずれていくと大きくなる。原曲に音がない区間 (間奏のドラムだけの所など) で
                   遠い onset と比べないよう、差が 150ms 以内の onset だけで測る
    melody_top     原曲のメロディの音の ±50ms で、出力の最高音が同じ音名である割合
    melody_late    メロディの音と、同じ音名の一番近い出力の音が 50ms より離れている割合 (±150ms 以内で探す)
    """
    notes = events[events[:, KIND] == KIND_NOTE]
    source_notes = rows[rows[:, ROW_TYPE] == TYPE_NOTE]
    reference = np.unique(source_notes[source_notes[:, ROW_D] != DRUMS, ROW_ONSET]).astype(np.float64)
    out: dict[str, float] = {}
    if len(notes) == 0 or len(reference) < 2:
        return out
    tol30, tol50, tol150 = (round(s * frame_rate) for s in (0.03, 0.05, 0.15))
    onsets = np.unique(notes[:, ONSET]).astype(np.float64)
    index = np.clip(np.searchsorted(reference, onsets), 1, len(reference) - 1)
    after, before = reference[index], reference[index - 1]
    offset = onsets - np.where(after - onsets < onsets - before, after, before)
    out["onset_sync"] = float(np.mean(np.abs(offset) <= tol30))
    near = np.abs(offset) <= tol150
    bins = (onsets[near] // (bin_seconds * frame_rate)).astype(np.int64)
    medians = [np.median(offset[near][bins == b]) for b in np.unique(bins) if (bins == b).sum() >= 10]
    if len(medians) >= 2:
        out["drift_std_ms"] = float(np.std(medians) / frame_rate * 1000)

    melody = source_notes[(source_notes[:, ROW_D] == MELODY) & (source_notes[:, ROW_ONSET] <= notes[:, ONSET].max())]
    if len(melody):
        top, late = [], []
        for onset, pitch in melody[:, [ROW_ONSET, ROW_A]]:
            near = notes[np.abs(notes[:, ONSET] - onset) <= tol50]
            top.append(len(near) > 0 and near[:, PITCH].max() % 12 == pitch % 12)
            same = notes[notes[:, PITCH] % 12 == pitch % 12, ONSET] - onset
            same = np.abs(same[np.abs(same) <= tol150])
            late.append(len(same) == 0 or same.min() > tol50)
        out["melody_top"] = float(np.mean(top))
        out["melody_late"] = float(np.mean(late))
    return out
