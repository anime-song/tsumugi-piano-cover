"""楽譜を五線の近くに収める後処理 (拍の上の演奏 -> 楽譜で、V2 のカバーを楽譜にしたもの向け)。

カバーは音域が広く、モデルの楽譜は右手の高音や左手の中音域を加線の多い位置にそのまま書くことが多い。
そこで段ごとに、小節ごとの書き方 (音部記号とオクターブ記号の組) を選び直す。
    上の段: ト音記号 / ト音記号 + 8va / ト音記号 + 15ma / ヘ音記号
    下の段: ヘ音記号 / ヘ音記号 + 8vb / ト音記号
小節の費用は、その書き方での加線の多さ (ALLOWED_LEDGERS 本を超えた分) と書き方そのものの費用 (8va などは
少しだけ避ける) の和で、書き方を変えるたびに切り替えの費用を足し、曲を通して費用が最小の並びを選ぶ (Viterbi)。
音高 (Note.pitch) は実音なので変えず、Measure.clefs と Measure.ottavas (小節の頭の状態) だけを書き換える。
モデルが書いた小節の中の音部記号とオクターブ記号の変更は消す。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from piano_score.score import Measure, spell

DIATONIC = "CDEFGAB"
# 五線の一番下と一番上の線の位置 (オクターブ x 7 + 幹音の番号)。ト音記号は E4〜F5、ヘ音記号は G2〜A3
STAFF_LINES = {"G2": (4 * 7 + 2, 5 * 7 + 3), "F4": (2 * 7 + 4, 3 * 7 + 5)}
OTTAVA_SHIFT = {"none": 0, "8va": 1, "15ma": 2, "8vb": -1, "15mb": -2}
# 段ごとの書き方 (音部記号, オクターブ記号, 小節ごとの費用)
MODES = {
    1: (("G2", "none", 0.0), ("G2", "8va", 1.5), ("G2", "15ma", 3.0), ("F4", "none", 1.5)),
    2: (("F4", "none", 0.0), ("F4", "8vb", 1.5), ("G2", "none", 0.5)),
}
ALLOWED_LEDGERS = 2  # これ以下の加線は費用にしない
CLEF_SWITCH = 4.0
OTTAVA_SWITCH = 2.5


@dataclass
class StaffStats:
    clef_changes: int = 0  # 書き方を決めたあとの音部記号の変更の数
    ottava_measures: int = 0  # オクターブ記号の付いた小節 (段ごと)
    ledger_before: int = 0  # 加線が ALLOWED_LEDGERS 本を超える音 (直す前、モデルの書き方)
    ledger_after: int = 0


def _ledgers(pitch: int, alter: int, clef: str, ottava: str) -> int:
    step, octave = spell(pitch, alter)
    position = (octave - OTTAVA_SHIFT[ottava]) * 7 + DIATONIC.index(step)
    # オクターブ付きの音部記号 (G2+1 など) は記号の分だけ位置をずらす
    sign, _, change = clef.partition("+") if "+" in clef else clef.partition("-")
    if change:
        position -= int(change) * (1 if "+" in clef else -1) * 7
    bottom, top = STAFF_LINES.get(sign, STAFF_LINES["G2"])
    return max(0, (position - top) // 2, (bottom - position) // 2)


def _notes(measure: Measure, staff: int) -> list[tuple[int, int]]:
    return [(n.pitch, n.alter) for g in measure.groups if g.display_staff == staff for n in g.notes]


def _excess(notes: list[tuple[int, int]], clef: str, ottava: str) -> tuple[float, int]:
    """(費用, 加線が多すぎる音の数)"""
    over = [max(0, _ledgers(p, a, clef, ottava) - ALLOWED_LEDGERS) for p, a in notes]
    return float(sum(o**1.5 for o in over)), sum(o > 0 for o in over)


def readable_staves(measures: list[Measure]) -> StaffStats:
    """measures の音部記号とオクターブ記号をその場で選び直し、数を返す"""
    stats = StaffStats()
    if not measures:
        return stats
    for staff in (1, 2):
        modes = MODES[staff]
        notes = [_notes(m, staff) for m in measures]
        stats.ledger_before += sum(
            _excess(n, m.clefs[staff - 1], m.ottavas[staff - 1])[1] for n, m in zip(notes, measures)
        )
        cost = np.array([[_excess(n, clef, ottava)[0] + base for clef, ottava, base in modes] for n in notes])
        switch = np.array(
            [[CLEF_SWITCH * (a[0] != b[0]) + OTTAVA_SWITCH * (a[1] != b[1]) for b in modes] for a in modes]
        )
        total = cost[0] + switch[0]  # 曲の頭は既定の書き方からの切り替えとみなす
        back = np.zeros((len(measures), len(modes)), dtype=np.int64)
        for i in range(1, len(measures)):
            candidates = total[:, None] + switch
            back[i] = candidates.argmin(0)
            total = candidates.min(0) + cost[i]
        path = [int(total.argmin())]
        for i in range(len(measures) - 1, 0, -1):
            path.append(int(back[i, path[-1]]))
        path.reverse()
        previous = None
        for measure, mode, n in zip(measures, path, notes):
            clef, ottava, _ = modes[mode]
            clefs, ottavas = list(measure.clefs), list(measure.ottavas)
            clefs[staff - 1], ottavas[staff - 1] = clef, ottava
            measure.clefs, measure.ottavas = tuple(clefs), tuple(ottavas)
            prefix = (f"clef{staff}_", f"ottava{staff}_")
            measure.directions = [d for d in measure.directions if not d[1].startswith(prefix)]
            stats.clef_changes += previous is not None and previous != clef
            stats.ottava_measures += ottava != "none"
            stats.ledger_after += _excess(n, clef, ottava)[1]
            previous = clef
    return stats
