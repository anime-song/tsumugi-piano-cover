from fractions import Fraction

from piano_perf2score.tidy import tidy_score
from piano_score.score import Duration, Group, Measure, Note

QUARTER = Duration("quarter", 0)


def measure(groups, key=0, directions=()):
    return Measure((4, 4), key, ("G2", "F4"), Fraction(4), groups=groups, directions=list(directions))


def group(onset, *notes, staff=1):
    return Group(Fraction(onset), staff, 1, QUARTER, [Note(p, a, tie) for p, a, tie in notes])


def test_drops_leading_rests_and_moves_directions():
    measures = [measure([], directions=[(Fraction(2), "dyn_mf")]), measure([group(0, (60, 0, False))])]
    stats = tidy_score(measures)
    assert stats.rests == 1 and len(measures) == 1
    assert measures[0].directions == [(Fraction(0), "dyn_mf")]


def test_drops_ties_without_target():
    kept, dangling = Note(64, 0, True), Note(67, 0, True)
    measures = [measure([Group(Fraction(0), 1, 1, QUARTER, [kept, dangling]), group(1, (64, 0, False))])]
    assert tidy_score(measures).ties == 1
    assert kept.tie and not dangling.tie


def test_pedal_start_while_down_becomes_change_and_closes_at_end():
    measures = [
        measure([group(0, (60, 0, False))], directions=[(Fraction(0), "pedal_start"), (Fraction(2), "pedal_start")])
    ]
    measures.append(
        measure([group(0, (60, 0, False))], directions=[(Fraction(1), "pedal_stop"), (Fraction(2), "pedal_stop")])
    )
    tidy_score(measures)
    assert sorted(measures[0].directions) == [(0, "pedal_start"), (2, "pedal_start"), (2, "pedal_stop")]
    assert measures[1].directions == [(1, "pedal_stop")]


def test_drops_repeated_dynamics():
    measures = [measure([group(0, (60, 0, False))], directions=[(Fraction(0), "dyn_mf")])]
    measures.append(measure([group(0, (60, 0, False))], directions=[(Fraction(0), "dyn_mf"), (Fraction(2), "dyn_f")]))
    assert tidy_score(measures).dynamics == 1
    assert measures[1].directions == [(2, "dyn_f")]


def test_respells_double_flats_and_white_enharmonics():
    # 変ニ長調 (フラット 5 つ) の中の A (Bbb と書かれたもの) と E (Fb と書かれたもの)
    notes = [(61, -1, False), (63, -1, False), (65, 0, False), (66, -1, False), (68, -1, False), (70, -1, False)]
    measures = [
        measure(
            [group(i % 4, n) for i, n in enumerate(notes)] + [group(0, (69, -2, False)), group(1, (64, -1, False))],
            key=-5,
        )
    ]
    tidy_score(measures)
    alters = {n.pitch: n.alter for g in measures[0].groups for n in g.notes}
    assert alters[69] == 0 and alters[64] == 0 and alters[61] == -1


def test_rechooses_a_clearly_wrong_key():
    # 嬰ハ長調 / 変ニ長調の音を、ホ長調 (シャープ 4 つ) で書いたもの
    scale = [61, 63, 65, 66, 68, 70, 72]
    groups = [group(i % 4, (p, 0 if p % 12 in (0, 5) else 1, False)) for i, p in enumerate(scale * 3)]
    measures = [measure(groups, key=4)]
    stats = tidy_score(measures)
    assert stats.keys == 1 and measures[0].key == -5
    assert {n.alter for g in measures[0].groups for n in g.notes} == {-1, 0}


def test_chromatic_steps_are_spelled_by_direction():
    # ニ長調 (シャープ 2 つ) の E-F-F#-G-G# (F と G は E#・F## と綴るべき半音の上行) と、下行の A-Ab-G
    line = [(64, 0), (65, 0), (66, 1), (67, 0), (68, 1), (69, 0), (68, 1), (67, 0)]
    groups = [Group(Fraction(i, 2), 1, 1, Duration("eighth", 0), [Note(p, a)]) for i, (p, a) in enumerate(line)]
    measures = [measure(groups, key=2)]
    tidy_score(measures)
    alters = [g.notes[0].alter for g in measures[0].groups]
    assert alters[1] == 1  # E#
    assert alters[3] == 0  # G (調の音のまま)
    assert alters[6] == -1  # A から G へ下がる G#/Ab は Ab
