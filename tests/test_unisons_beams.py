from fractions import Fraction

from piano_perf2score.synth_score import remove_unisons
from piano_score.musicxml import beam_starts
from piano_score.score import Duration, Group, Measure, Note

EIGHTH, HALF = Duration("eighth", 0), Duration("half", 0)


def test_remove_unisons_keeps_melody_and_bass():
    melody = Group(Fraction(0), 1, 1, HALF, [Note(60, 0), Note(64, 0)])
    left = Group(Fraction(0), 2, 1, HALF, [Note(48, 0), Note(55, 0), Note(64, 0)])
    measure = Measure((4, 4), 0, ("G2", "F4"), Fraction(4), groups=[melody, left])
    assert remove_unisons([measure]) == 1
    assert [n.pitch for n in melody.notes] == [60, 64]  # 旋律の一番上 (E4) は残す
    assert [n.pitch for n in left.notes] == [48, 55]


def test_seven_eight_beams_follow_the_bass():
    # 7/8 の下の段: 低音が 0・1・2 拍目 (8 分の位置で 0・2・4) にある 2+2+3
    pitches = [38, 45, 38, 45, 38, 45, 50]
    groups = [Group(Fraction(i, 2), 2, 1, EIGHTH, [Note(p, 0)]) for i, p in enumerate(pitches)]
    measure = Measure((7, 8), 0, ("G2", "F4"), Fraction(7, 2), groups=groups)
    assert beam_starts(measure) == [0, 1, 2, Fraction(7, 2)]
    # 3+2+2 の並び
    pitches = [38, 45, 50, 38, 45, 38, 45]
    for g, p in zip(groups, pitches):
        g.notes = [Note(p, 0)]
    assert beam_starts(measure) == [0, Fraction(3, 2), Fraction(5, 2), Fraction(7, 2)]
