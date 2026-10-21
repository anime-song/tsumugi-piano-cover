from fractions import Fraction

from piano_perf2score.playability import upper_to_right
from piano_score.score import Duration, Group, Measure, Note

QUARTER, HALF, SIXTEENTH = Duration("quarter", 0), Duration("half", 0), Duration("16th", 0)


def _measure(groups: list[Group]) -> Measure:
    return Measure((4, 4), 0, ("G2", "F4"), Fraction(4), groups=groups)


def test_chord_over_bass_goes_to_free_right_hand():
    # 左手が Ab2 の上に Gb3・Bb3 を同時に弾き (10 度を超える)、右手はその間 F4 だけ
    melody = Group(Fraction(0), 1, 1, HALF, [Note(65, 0)])
    left = Group(Fraction(0), 2, 1, QUARTER, [Note(44, -1), Note(54, -1), Note(58, -1)])
    measure = _measure([melody, left])
    assert upper_to_right([measure]) == 2
    assert [n.pitch for n in left.notes] == [44]
    moved = next(g for g in measure.groups if g.staff == 1 and g.voice == 2)
    assert [n.pitch for n in moved.notes] == [54, 58]


def test_oompah_chord_stays_in_left_hand():
    # 4 分の低音のあとに 4 分の和音 (1 拍離れている) は左手のまま
    bass = Group(Fraction(0), 2, 1, QUARTER, [Note(36, 0)])
    chord = Group(Fraction(1), 2, 1, QUARTER, [Note(52, 0), Note(55, 0)])
    measure = _measure([bass, chord])
    assert upper_to_right([measure]) == 0
    assert [n.pitch for n in chord.notes] == [52, 55]


def test_fast_leap_moves_chord_unless_right_hand_is_too_far():
    # 16 分で低音 -> 和音と跳ぶ所。右手が 2 オクターブ上で鳴っていれば動かさない
    bass = Group(Fraction(1), 2, 1, SIXTEENTH, [Note(34, -1)])
    chord = Group(Fraction(5, 4), 2, 1, SIXTEENTH, [Note(50, 0), Note(53, 0), Note(56, -1)])
    high = Group(Fraction(1), 1, 1, QUARTER, [Note(80, -1)])
    measure = _measure([bass, chord, high])
    assert upper_to_right([measure]) == 0
    high.notes = [Note(59, 0)]
    assert upper_to_right([measure]) == 3
    assert not any(g is chord for g in measure.groups)
