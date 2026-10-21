"""原曲のコード (tsumugi のコードマーカー) を、拍の上の演奏 -> 楽譜の楽譜にコード記号として置く。

Cover Studio の原曲 MIDI (source.mid) には、tsumugi が採譜したコードが「時刻 (秒) とコード名」のマーカー
("A:m7/E"・"C:maj"・"N") で入っている。楽譜の小節線は同じ SheetSage2-Refined の拍から作っているので、
マーカーの時刻を拍の時刻にして小節を決め、小節の中の割合を楽譜の小節の長さに掛けて 8 分音符の位置に丸める
(1 拍が 4 分音符なら半拍。最初の小節を短く書いたときも、その小節の長さで割り振る)。
丸めの残りは 90% が 0.05〜0.08 拍以内で、3.5 拍目のような裏拍 (次のコードを 8 分早く鳴らす先取り) もそのまま置く。

コード記号は小節の指示 (Measure.directions) に "chord_Gbm7/Db" の名前で入れ、書き出し (piano_score.musicxml) で
<harmony> にする。根音とベースの綴りは楽譜の調号に合わせる (tidy.spell_pitch_class、調の音でなければ tsumugi の綴り)。
同じコードが続くときは書かず、"N" はコードのあとに来たときだけ N.C. にする。
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np

from piano_score.score import PITCH_CLASS_STEP, STEP_PITCH_CLASS, Measure

from .beat_input import BEAT_SECONDS
from .render import Performance
from .tidy import spell_pitch_class

NO_CHORD = "N.C."
SNAP_QUARTERS = Fraction(1, 2)  # コード記号を置く位置の刻み (8 分音符)
ACCIDENTALS = {-1: "b", 0: "", 1: "#"}


def read_chord_markers(path: str | Path) -> list[tuple[float, str]]:
    """MIDI のマーカーのうちコードとして読めるもの [(秒, マーカーの文字列)]"""
    from symusic import Score

    markers = [(float(m.time), m.text.strip()) for m in Score(str(path)).to("second").markers]
    return [(t, text) for t, text in markers if text == "N" or _parse(text) is not None]


def _pitch_class(name: str) -> tuple[int, int] | None:
    """ "F#" -> (6, 1)。(音名, tsumugi の変化記号)"""
    if not name or name[0] not in STEP_PITCH_CLASS or any(c not in "#b" for c in name[1:]):
        return None
    alter = name.count("#") - name.count("b")
    return (STEP_PITCH_CLASS[name[0]] + alter) % 12, alter


def _parse(text: str) -> tuple[tuple[int, int], str, tuple[int, int] | None] | None:
    """ "A:m7/E" -> ((根音, 変化記号), "m7", (ベース, 変化記号))。":" のないものは長三和音 ("E/G#")"""
    main, _, bass = text.partition("/")
    root, _, quality = main.partition(":")
    root_pc = _pitch_class(root)
    bass_pc = _pitch_class(bass) if bass else None
    if root_pc is None or (bass and bass_pc is None):
        return None
    return root_pc, "" if quality in ("", "maj") else quality, bass_pc


def _note_name(pitch_class: int, tsumugi_alter: int, key: int) -> str:
    alter = spell_pitch_class(pitch_class, key, preferred=tsumugi_alter)
    return PITCH_CLASS_STEP[(pitch_class - alter) % 12] + ACCIDENTALS[alter]


def chord_symbol(text: str, key: int) -> str | None:
    """tsumugi のコード名を、調号 key に合わせた楽譜のコード記号にする ("A:m7/E" -> "Am7/E")。"N" は N.C."""
    if text == "N":
        return NO_CHORD
    parsed = _parse(text)
    if parsed is None:
        return None
    root, quality, bass = parsed
    symbol = _note_name(*root, key) + quality
    return symbol + (f"/{_note_name(*bass, key)}" if bass else "")


def place_chords(
    measures: list[Measure], performance: Performance, chords: list[tuple[float, str]], skipped: int = 0
) -> int:
    """chords [(拍の時刻の秒、beat_input の時間軸), tsumugi のコード名] を measures の指示に足し、置いた数を返す。
    measures[k] は performance.measure_starts[k + skipped] から始まる小節 (skipped は tidy で消した冒頭の小節の数)。
    消した冒頭の小節にあるコードは、最初の小節の頭に寄せる (最後の 1 つだけ)"""
    starts = np.asarray(performance.measure_starts, dtype=np.float64)[skipped : skipped + len(measures) + 1]
    if not len(starts):
        return 0
    # 各小節の拍の時刻での長さ。最後の小節は終わりが分からないので、他の小節の 4 分音符あたりの長さから
    spans = np.diff(starts)
    per_quarter = (
        float(np.median(spans / [float(m.length) for m in measures[: len(spans)]])) if len(spans) else BEAT_SECONDS
    )
    spans = [*spans, *(float(m.length) * per_quarter for m in measures[len(spans) :])]
    placed: dict[tuple[int, Fraction], str] = {}
    for seconds, text in sorted(chords):
        index = int(np.searchsorted(starts, seconds, side="right")) - 1
        if index < 0:
            index, position = 0, Fraction(0)
        elif index < len(measures):
            share = (seconds - starts[index]) / spans[index]
            position = round(Fraction(share) * measures[index].length / SNAP_QUARTERS) * SNAP_QUARTERS
        else:
            continue
        if position >= measures[index].length:
            index, position = index + 1, Fraction(0)
            if index >= len(measures):
                continue
        placed[(index, position)] = text  # 同じ位置に丸まったものは後のコード
    count = 0
    previous = None
    for (index, position), text in sorted(placed.items()):
        symbol = chord_symbol(text, measures[index].key)
        if symbol is None or symbol == previous or (symbol == NO_CHORD and previous is None):
            continue
        measures[index].directions.append((position, f"chord_{symbol}"))
        previous = symbol
        count += 1
    return count
