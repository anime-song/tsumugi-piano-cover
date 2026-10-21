from fractions import Fraction

import numpy as np

from piano_perf2score.beat_input import BEAT_SECONDS
from piano_perf2score.render import Performance
from piano_perf2score.swing import mark_swing, straighten_swing
from piano_score.score import Measure

TICK = BEAT_SECONDS / 12


def _performance(offbeat: int, bars: int = 8) -> Performance:
    """4/4 の小節の各拍に、拍の頭と offbeat 刻みの裏の 2 音"""
    onsets = [(b * 4 + beat) * BEAT_SECONDS + t * TICK for b in range(bars) for beat in range(4) for t in (0, offbeat)]
    notes = np.array([[t, t + 3 * TICK, 60, 80] for t in onsets], dtype=np.float64)
    return Performance(notes, np.zeros((0, 2)), np.arange(bars) * 4 * BEAT_SECONDS, melody=np.zeros(len(notes), bool))


def test_straightens_swung_eighths():
    performance = _performance(8)
    swing = straighten_swing(performance)
    assert swing.all()
    ticks = np.round(performance.notes[:, 0] / TICK).astype(int) % 12
    assert set(ticks.tolist()) == {0, 6}


def test_leaves_straight_eighths_and_sixteenths():
    for offbeat in (6, 3, 9):
        performance = _performance(offbeat)
        before = performance.notes.copy()
        assert not straighten_swing(performance).any()
        assert np.array_equal(performance.notes, before)


def test_mark_swing_sets_state_and_drops_model_marks():
    measures = [
        Measure((4, 4), 0, ("G2", "F4"), Fraction(4), directions=[(Fraction(1), "swing_16th")]) for _ in range(3)
    ]
    assert mark_swing(measures, np.array([False, True, True, False]), skipped=1) == 2
    assert [m.swing for m in measures] == ["8th", "8th", "none"]
    assert all(not m.directions for m in measures)
