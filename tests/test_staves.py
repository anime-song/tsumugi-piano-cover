from fractions import Fraction

from piano_perf2score.staves import readable_staves
from piano_score.score import Duration, Group, Measure, Note

QUARTER = Duration("quarter", 0)


def measure(rh, lh, ottavas=("none", "none")):
    groups = [Group(Fraction(i), 1, 1, QUARTER, [Note(p, 0)]) for i, p in enumerate(rh)]
    groups += [Group(Fraction(i), 2, 1, QUARTER, [Note(p, 0)]) for i, p in enumerate(lh)]
    return Measure((4, 4), 0, ("G2", "F4"), Fraction(4), ottavas=ottavas, groups=groups)


def test_high_right_hand_gets_8va_and_middle_left_hand_gets_treble_clef():
    measures = [measure([72, 74, 76, 77], [43, 47, 50, 55]) for _ in range(2)]
    measures += [measure([91, 93, 95, 96], [64, 67, 72, 71]) for _ in range(6)]
    measures += [measure([72, 74, 76, 77], [43, 47, 50, 55]) for _ in range(2)]
    stats = readable_staves(measures)
    assert [m.ottavas[0] for m in measures] == ["none"] * 2 + ["8va"] * 6 + ["none"] * 2
    assert [m.clefs[1] for m in measures] == ["F4"] * 2 + ["G2"] * 6 + ["F4"] * 2
    assert stats.ledger_after == 0 < stats.ledger_before


def test_a_single_high_note_does_not_switch():
    measures = [measure([72, 74, 76, 77], [43, 47, 50, 55]) for _ in range(5)]
    measures[2] = measure([72, 74, 76, 89], [43, 47, 50, 55])  # 1 音だけ加線 3 本
    readable_staves(measures)
    assert all(m.ottavas == ("none", "none") and m.clefs == ("G2", "F4") for m in measures)
