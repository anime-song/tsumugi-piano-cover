from fractions import Fraction

import numpy as np

from piano_perf2score.beat_tempo import apply_bar_tempos, tempo_segments
from piano_perf2score.render import Performance
from piano_score.score import Measure


def _measures(count):
    return [Measure(time_signature=(4, 4), key=0, clefs=("G2", "F4"), length=Fraction(4)) for _ in range(count)]


def _performance(qpms):
    seconds = np.concatenate([[0.0], np.cumsum([240.0 / q for q in qpms])])
    return Performance(np.zeros((0, 4)), np.zeros((0, 2)), seconds[:-1], bar_seconds=seconds)


def test_small_bar_jitter_is_one_tempo():
    qpms = [135.0, 135.4, 134.7, 135.2, 133.9, 135.0]
    assert [(a, b) for a, b, _ in tempo_segments(qpms)] == [(0, 6)]


def test_tempo_change_and_single_bar_outlier_make_segments():
    qpms = [100.0, 101.0, 99.0, 100.5, 150.0, 151.0, 149.0, 150.0, 300.0, 150.0]
    assert [(a, b) for a, b, _ in tempo_segments(qpms)] == [(0, 4), (4, 8), (8, 9), (9, 10)]


def test_apply_bar_tempos_writes_tempo_only_at_segment_starts():
    qpms = [135.0, 135.4, 134.7, 135.2, 90.0, 90.5, 89.6, 90.2, 180.0, 90.0]
    measures = _measures(len(qpms))
    measures[3].directions.append((Fraction(0), "metronome_quarter_120"))  # モデルの記号は消す
    apply_bar_tempos(measures, _performance(qpms))
    assert [k for k, m in enumerate(measures) if m.tempos] == [0, 4, 8, 9]
    marks = [(k, n) for k, m in enumerate(measures) for _, n in m.directions if n.startswith("metronome_")]
    # 4 小節以上続く区間で値が変わるところだけに記号を置き、1 小節だけの外れには置かない
    assert [k for k, _ in marks] == [0, 4]
