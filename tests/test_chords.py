from fractions import Fraction

import numpy as np

from piano_perf2score.beat_input import BEAT_SECONDS
from piano_perf2score.chords import chord_symbol, place_chords
from piano_perf2score.render import Performance
from piano_score.score import Measure


def test_chord_symbol_follows_key_signature():
    assert chord_symbol("A:m7/E", 0) == "Am7/E"
    assert chord_symbol("C:maj", 0) == "C"
    assert chord_symbol("E/G#", 4) == "E/G#"
    assert chord_symbol("F#:m7", -6) == "Gbm7"  # フラット 6 つの曲
    assert chord_symbol("Cb:M7", 0) == "BM7"  # 調号にない白鍵の異名は使わない
    assert chord_symbol("N", 0) == "N.C."


def _measures(lengths, key=0):
    return [Measure((4, 4), key, ("G2", "F4"), Fraction(length)) for length in lengths]


def test_place_chords_on_eighths_and_skips_repeats():
    performance = Performance(np.zeros((0, 4)), np.zeros((0, 2)), np.arange(3) * 4 * BEAT_SECONDS)
    measures = _measures([4, 4, 4])
    chords = [(-0.5, "N"), (0.02, "C:maj"), (3.45 * BEAT_SECONDS, "G:7"), (4.1 * BEAT_SECONDS, "G:7")]
    chords.append((8 * BEAT_SECONDS, "N"))
    assert place_chords(measures, performance, chords) == 3
    assert measures[0].directions == [(0, "chord_C"), (Fraction(7, 2), "chord_G7")]
    assert measures[2].directions == [(0, "chord_N.C.")]


def test_place_chords_scales_a_short_first_measure_and_skipped_measures():
    # 4 拍の小節を 2 拍の長さで書いた最初の小節と、tidy で消した冒頭の小節
    performance = Performance(np.zeros((0, 4)), np.zeros((0, 2)), np.arange(3) * 4 * BEAT_SECONDS)
    measures = _measures([2, 4])
    chords = [(1.0 * BEAT_SECONDS, "D:m"), (6 * BEAT_SECONDS, "A:m")]
    assert place_chords(measures, performance, chords, skipped=1) == 2
    assert measures[0].directions == [(0, "chord_Dm"), (1, "chord_Am")]
