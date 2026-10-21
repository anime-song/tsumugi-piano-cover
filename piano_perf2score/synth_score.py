"""手順で作る合成の楽譜 (拍の上の演奏 -> 楽譜の学習データ)。

学習の楽譜コーパスは書き方の丁寧さがまちまちで (半分以上の曲にスラーがない、同じ伴奏を場所ごとに違う形で書く)、
モデルの楽譜もそれを引き継ぐ。そこで「人ならこう書く」書き方の決まりで楽譜を組み立て、描き出し (COVER_RENDER) と
組にして学習に混ぜる。和声や旋律の自然さには凝らず、次のものが表れるようにする。
    - 旋律の音高と時間の連続性: 順次進行の多い乱歩、2・4 小節のフレーズ、フレーズの終わりは長い音か休符
    - 重なる音と独立したリズム: 上の段の伸ばす内声 (声部 2)、下の段の伸ばす低音と刻み (声部 1・2)
    - 同時に鳴らす和音のまとまり: コードから片手で届く配置 (10 度以内) の和音
    - 分散和音・オクターブなどの反復: 区間ごとに伴奏型を選び、同じ区間を繰り返すときは同じ書き方
    - 声部の切り替わり: 内声を出し入れする頻度を曲ごとに変える
書き方の決まり (人が書くならこう書く):
    - 手をまたいで上がる分散和音 (左手から右手へ) と、レガートのまとまりにはスラー
    - 短く切る音は 16 分 + 休符ではなく、拍の音符にスタッカート (ズンチャの「チャ」など)
    - シンコペーションは、拍の区切り (4/4 の小節の中央、16 分の裏から始まる音の次の拍) をタイで見せる
    - 区間の頭に強弱記号、盛り上がる区間の前に松葉、コードの変わり目でペダルを踏み替える
    - 拍子は 4/4 を中心に 3/4・6/8・12/8・2/4・2/2・5/4・7/8 など。8 分主体の曲の一部は Swing
書き出す前に staves.readable_staves で 8va と音部記号を選び、綴りは調号に合わせる (tidy.spell_pitch_class)。

    python -m piano_perf2score.synth_score --songs 20000 --out-dir data/piano_score/generated \\
        --samples outputs/synth_score_samples     # キャッシュ (prepare と同じ形) と見本の MusicXML
"""

from __future__ import annotations

import argparse
import json
import random
import zlib
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from piano_score.score import Duration, Group, Measure, Note, metronome_bpm
from piano_score.tokenizer import ScoreTokenizer

from .staves import readable_staves
from .tidy import spell_pitch_class

F = Fraction
SCALES = {"major": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10)}
# コード進行 (音階の度数、1 小節に 1 つ)
PROGRESSIONS = {
    "major": ((1, 5, 6, 4), (6, 4, 1, 5), (1, 6, 4, 5), (4, 5, 3, 6), (2, 5, 1, 1), (4, 1, 5, 6), (1, 3, 4, 5)),
    "minor": ((1, 6, 7, 1), (1, 4, 5, 1), (6, 7, 1, 1), (1, 7, 6, 5), (4, 5, 1, 6), (1, 6, 3, 7)),
}
# 拍子 (拍子, 選ぶ重み)
METERS = (
    ((4, 4), 50), ((3, 4), 10), ((6, 8), 9), ((2, 4), 4), ((12, 8), 5), ((2, 2), 4), ((5, 4), 3), ((7, 8), 3),
    ((6, 4), 2), ((9, 8), 2), ((3, 8), 2),
)  # fmt: skip
MELODY_RANGE = (60, 84)  # 旋律の音域 (C4〜C6)
LH_PATTERNS = ("block", "oompah", "broken", "alberti", "octave", "pop8", "two_voice", "handoff")
DYNAMIC_LEVELS = ("pp", "p", "mp", "mf", "f", "ff")
# 書ける音価 (4 分音符単位 -> 音符の種類と付点)
WRITABLE = {
    F(4): ("whole", 0), F(3): ("half", 1), F(2): ("half", 0), F(3, 2): ("quarter", 1), F(1): ("quarter", 0),
    F(3, 4): ("eighth", 1), F(1, 2): ("eighth", 0), F(3, 8): ("16th", 1), F(1, 4): ("16th", 0), F(1, 8): ("32nd", 0),
}  # fmt: skip


@dataclass
class Song:
    meter: tuple[int, int]
    key: int  # 調号 (-6..6)
    mode: str
    qpm: float
    grid: Fraction  # 旋律のリズムの細かさ (8 分 or 16 分)
    swing: bool
    detached: float  # 旋律の音を短く切る (スタッカート) 割合
    syncopation: float
    inner_voice: float  # 上の段に内声を入れる小節の割合
    slur_phrases: float  # フレーズにスラーを付ける割合
    measures: list[Measure] = field(default_factory=list)

    @property
    def bar(self) -> Fraction:
        return F(4 * self.meter[0], self.meter[1])

    @property
    def beat(self) -> Fraction:
        """拍の長さ。複合拍子 (x/8 で 3 の倍数) は付点 4 分、5/8・7/8 は 8 分、2/2 は 2 分"""
        beats, unit = self.meter
        if unit == 8:
            return F(3, 2) if beats % 3 == 0 else F(1, 2)
        return F(4, unit)

    @property
    def tonic(self) -> int:
        major = (7 * self.key) % 12
        return major if self.mode == "major" else (major + 9) % 12


def _choose(rng: random.Random, weighted) -> object:
    items, weights = zip(*weighted)
    return rng.choices(items, weights=weights)[0]


def new_song(rng: random.Random) -> Song:
    meter = _choose(rng, METERS)
    beat_unit = meter[1]
    compound = beat_unit == 8 and meter[0] % 3 == 0
    grid = F(1, 2) if compound or rng.random() < 0.45 else F(1, 4)
    swing = not compound and grid == F(1, 2) and meter in ((4, 4), (2, 4), (2, 2)) and rng.random() < 0.3
    return Song(
        meter=meter, key=rng.randint(-6, 6), mode="major" if rng.random() < 0.65 else "minor",
        qpm=rng.uniform(64, 176) * (1.5 if compound else 1.0), grid=grid, swing=swing,
        detached=rng.choice((0.0, 0.0, 0.15, 0.4)), syncopation=rng.uniform(0.0, 0.5),
        inner_voice=rng.choice((0.0, 0.0, 0.3, 0.7)), slur_phrases=rng.choice((0.3, 0.6, 0.9)),
    )  # fmt: skip


# ----------------------------------------------------------------------
# 音価の書き方
# ----------------------------------------------------------------------
def notate(onset: Fraction, length: Fraction, song: Song) -> list[tuple[Fraction, Duration]]:
    """小節の中の onset から length の音を、書き方の決まりでタイの鎖 [(位置, 音価)] にする"""
    pieces = []
    beat, bar = song.beat, song.bar
    end = min(onset + length, bar)
    while onset < end:
        stop = end
        next_beat = (onset // beat + 1) * beat
        on_beat = onset % beat == 0
        if song.meter == (4, 4) and onset < 2 < stop and onset != 0 and not (onset == 1 and stop == 3):
            stop = F(2)  # 4/4 の小節の中央はまたがない (頭からの音と 4 分-2 分-4 分のシンコペーションは除く)
        if not on_beat:
            # 8 分の裏から始まる音は次の拍の終わりまで、16 分の裏から始まる音は次の拍まで
            limit = next_beat if onset % F(1, 2) != 0 or beat != 1 else next_beat + beat
            stop = min(stop, limit)
        elif (stop - onset) % beat != 0 and stop - onset > beat:
            stop = onset + (stop - onset) // beat * beat  # 拍の頭からの長い音は拍の倍数まで書き、残りをタイで
        piece = stop - onset
        while piece not in WRITABLE:
            piece = max(v for v in WRITABLE if v < piece)
        kind, dots = WRITABLE[piece]
        pieces.append((onset, Duration(kind, dots)))
        onset += piece
    return pieces


def _groups(onset, length, pitches, staff, voice, song, articulations=(), alters=None) -> list[Group]:
    """音を (タイの鎖で) 塊にする。最後の塊以外の音にはタイを付ける"""
    chain = notate(onset, length, song)
    groups = []
    for i, (at, duration) in enumerate(chain):
        tie = i < len(chain) - 1
        notes = [Note(p, spell_pitch_class(p % 12, song.key), tie) for p in sorted(set(pitches))]
        groups.append(Group(at, staff, voice, duration, notes, tuple(articulations) if i == 0 else ()))
    return groups


# ----------------------------------------------------------------------
# 和声
# ----------------------------------------------------------------------
def chord_pitch_classes(song: Song, degree: int, seventh: bool) -> list[int]:
    scale = SCALES[song.mode]
    pcs = [(song.tonic + scale[(degree - 1 + k) % 7]) % 12 for k in (0, 2, 4, 6)[: 4 if seventh else 3]]
    if song.mode == "minor" and degree == 5:
        pcs[1] = (pcs[1] + 1) % 12  # 短調の属和音は導音を上げる
    return pcs


def voicing(pcs: list[int], low: int, size: int) -> list[int]:
    """low 以上の閉じた配置 (10 度以内) で size 音"""
    notes, pitch = [], low
    order = pcs * 3
    for pc in order:
        while pitch % 12 != pc:
            pitch += 1
        notes.append(pitch)
        if len(notes) == size:
            break
    while notes[-1] - notes[0] > 14 and len(notes) > 2:
        notes.pop()
    return notes


# ----------------------------------------------------------------------
# 旋律
# ----------------------------------------------------------------------
def beat_cells(song: Song, rng: random.Random) -> list[list[tuple[Fraction, Fraction]]]:
    """1 拍のリズムの型 [(拍の中の位置, 長さ)]"""
    if song.beat == F(3, 2):
        return [[(F(0), F(3, 2))], [(F(0), F(1)), (F(1), F(1, 2))], [(F(0), F(1, 2)), (F(1, 2), F(1, 2)), (F(1), F(1, 2))],
                [(F(0), F(1, 2)), (F(1, 2), F(1))], [(F(0), F(3, 4)), (F(3, 4), F(1, 4)), (F(1), F(1, 2))]]  # fmt: skip
    if song.beat == F(1, 2):
        return [[(F(0), F(1, 2))], [(F(0), F(1, 4)), (F(1, 4), F(1, 4))]]
    cells = [[(F(0), F(1))], [(F(0), F(1, 2)), (F(1, 2), F(1, 2))], [(F(1, 2), F(1, 2))]]
    if song.grid == F(1, 4):
        cells += [[(F(0), F(3, 4)), (F(3, 4), F(1, 4))], [(F(0), F(1, 4)), (F(1, 4), F(1, 2)), (F(3, 4), F(1, 4))],
                  [(F(0), F(1, 4)), (F(1, 4), F(1, 4)), (F(1, 2), F(1, 2))], [(F(0), F(1, 2)), (F(1, 2), F(1, 4)), (F(3, 4), F(1, 4))],
                  [(F(0), F(1, 4)), (F(1, 4), F(1, 4)), (F(1, 2), F(1, 4)), (F(3, 4), F(1, 4))]]  # fmt: skip
    if song.beat == 2:  # 2/2 は 4 分音符の型を 2 つ
        return [[(a, d) for a, d in c1] + [(a + 1, d) for a, d in c2] for c1 in cells for c2 in cells[:3]]
    return cells


def melody_rhythm(song: Song, rng: random.Random, phrase_end: bool) -> list[tuple[Fraction, Fraction]]:
    """1 小節の旋律の (位置, 長さ)。フレーズの終わりは長い音と休符"""
    beats = int(song.bar / song.beat)
    if phrase_end:
        return [(F(0), song.beat * max(1, beats - rng.choice((0, 1))))]
    cells = beat_cells(song, rng)
    notes = []
    for b in range(beats):
        cell = rng.choice(cells[:3]) if rng.random() < 0.4 else rng.choice(cells)
        notes += [(b * song.beat + a, d) for a, d in cell]
    # シンコペーション: 拍の頭の音を消して、前の裏の音を伸ばす
    if song.beat == 1 and rng.random() < song.syncopation:
        heads = [i for i, (a, _) in enumerate(notes) if i > 0 and a % 1 == 0 and notes[i - 1][0] % 1 == F(1, 2)]
        if heads:
            i = rng.choice(heads)
            onset, _ = notes[i - 1]
            notes[i - 1] = (onset, notes[i][0] + notes[i][1] - onset)
            del notes[i]
    return notes


def melody_pitches(song: Song, rhythm, chords, rng: random.Random, state: list[int]) -> list[int]:
    """順次進行の多い乱歩。拍の頭はコードの音に寄せる。state[0] は音階の上の位置 (度数 + 7 x オクターブ)"""
    scale = SCALES[song.mode]

    def pitch_of(position: int) -> int:
        return song.tonic + 12 * (position // 7) + scale[position % 7]

    pitches = []
    for onset, _ in rhythm:
        step = rng.choices((-2, -1, 0, 1, 2, 3, -3, 4, -4), weights=(8, 30, 12, 30, 8, 4, 4, 2, 2))[0]
        position = state[0] + step
        if not MELODY_RANGE[0] <= pitch_of(position) <= MELODY_RANGE[1]:
            position = state[0] - step
        pitch = pitch_of(position)
        if onset % song.beat == 0 and rng.random() < 0.7:
            chord = chords(onset)
            pitch = min((pitch + d for d in range(-3, 4)), key=lambda p: (p % 12 not in chord, abs(p - pitch)))
            position = next(
                (q for q in range(position - 2, position + 3) if pitch_of(q) == pitch),
                position,
            )
        state[0] = position
        pitches.append(pitch)
    return pitches


# ----------------------------------------------------------------------
# 伴奏型
# ----------------------------------------------------------------------
def left_hand(
    song: Song, pattern: str, pcs: list[int], rng: random.Random, measure: Measure, melody_rests: bool
) -> None:
    """1 小節の左手 (と handoff では右手の上の分散和音) を measure に足す"""
    bar, beat = song.bar, song.beat
    root = 36 + (pcs[0] - 36) % 12  # C2〜B2
    chord = voicing(pcs, 45 + (pcs[0] - 45) % 12 if rng.random() < 0.5 else 50, 3)  # A2〜D4 から
    add = measure.groups.extend
    if pattern == "block":
        add(_groups(F(0), bar, [root, *chord], 2, 1, song))
    elif pattern == "oompah":
        t = F(0)
        while t < bar:
            add(_groups(t, beat, [root], 2, 1, song))
            if t + beat < bar:
                add(_groups(t + beat, beat, chord, 2, 1, song, articulations=("staccato",) * (rng.random() < 0.6)))
            t += 2 * beat
    elif pattern in ("broken", "alberti", "handoff"):
        unit = F(1, 2) if beat != F(1, 2) else F(1, 4)
        if pattern == "alberti":
            cycle = [chord[0], chord[-1], chord[1], chord[-1]]
        else:
            third = next(p for p in range(root + 13, root + 25) if p % 12 in pcs[1:3])  # オクターブ上の 3 度か 5 度
            cycle = [root, root + 7, root + 12, third]
            cycle += cycle[-2:0:-1]
        steps = int(bar / unit)
        upper = pattern == "handoff" and melody_rests
        line = [cycle[i % len(cycle)] for i in range(steps)]
        if upper:  # 手をまたいで上がる分散和音: 後半を右手で 1 オクターブ上へ
            line = [root, root + 7, *[p for k in range(4) for p in voicing(pcs, root + 12 * (1 + k), 3)]][:steps]
        for i, pitch in enumerate(line):
            staff = 1 if upper and pitch >= 60 else 2
            group = _groups(i * unit, unit, [pitch], staff, 1, song)[0]
            if i == 0:
                group.slur_start = 1
            if i == len(line) - 1:
                group.slur_stop = 1
            add([group])
    elif pattern == "octave":
        unit = beat if rng.random() < 0.5 else min(beat, F(1, 2))
        t = F(0)
        while t < bar:
            add(_groups(t, unit, [root, root + 12], 2, 1, song))
            t += unit
    elif pattern == "pop8":
        hits = [F(0), F(3, 2), F(2), F(7, 2)] if song.meter == (4, 4) else [F(0)]
        for i, t in enumerate(hits):
            length = (hits[i + 1] if i + 1 < len(hits) else bar) - t
            add(_groups(t, length, [root] if i % 2 == 0 else chord, 2, 1, song))
    elif pattern == "two_voice":
        add(_groups(F(0), bar, [root], 2, 1, song))
        t = beat if beat <= 1 else F(1, 2)
        unit = min(beat, F(1, 2)) if beat != 2 else F(1)
        while t < bar:
            add(_groups(t, unit, chord, 2, 2, song))
            t += unit


def right_hand(song: Song, rhythm, pitches, chords, rng: random.Random, measure: Measure, octave: bool) -> None:
    for (onset, length), pitch in zip(rhythm, pitches):
        notes = [pitch]
        if octave:
            notes.append(pitch - 12)
        elif onset % song.beat == 0 and rng.random() < 0.35:
            below = [p for p in range(pitch - 9, pitch) if p % 12 in chords(onset)]
            notes += below[-rng.choice((1, 2)) :] if below else []
        written = length
        articulations = ()
        if rng.random() < song.detached and length <= song.beat:
            articulations = ("staccato",)  # 短く切る音は音価を詰めずにスタッカートで書く
        elif onset % song.beat != 0 and rng.random() < 0.05:
            articulations = ("accent",)
        measure.groups.extend(_groups(onset, written, notes, 1, 1, song, articulations))


def inner_voice(song: Song, pcs: list[int], melody_low: int, measure: Measure) -> None:
    """上の段の伸ばす内声 (声部 2): 旋律より下のコードの音を 2 分音符か小節いっぱいで"""
    pitch = max(p for p in range(melody_low - 12, melody_low - 2) if p % 12 in pcs[:3]) if melody_low > 62 else None
    if pitch is None:
        return
    half = song.bar / 2 if song.bar >= 2 and song.meter[0] % 2 == 0 else song.bar
    t = F(0)
    while t < song.bar:
        measure.groups.extend(_groups(t, half, [pitch], 1, 2, song))
        t += half


def _slur_phrases(bars: list[Measure], song: Song) -> None:
    """旋律 (上の段の声部 1) を休符で区切り、スタッカートのない 2 音以上のまとまりにスラーを付ける"""
    run: list[Group] = []
    end = None
    for i, measure in enumerate(bars):
        for g in sorted((g for g in measure.groups if (g.staff, g.voice) == (1, 1)), key=lambda g: g.onset):
            onset = i * song.bar + g.onset
            if end is not None and onset > end:
                _slur(run)
                run = []
            run.append(g)
            end = onset + g.duration.quarters
    _slur(run)


def _slur(run: list[Group]) -> None:
    # スタッカートを含むまとまりと、もうスラーのある所 (手をまたぐ分散和音) には付けない
    if len(run) >= 2 and not any("staccato" in g.articulations or g.slur_start or g.slur_stop for g in run):
        run[0].slur_start += 1
        run[-1].slur_stop += 1


# ----------------------------------------------------------------------
# 曲
# ----------------------------------------------------------------------
def generate(rng: random.Random) -> Song:
    song = new_song(rng)
    phrase = 4 if song.bar >= 3 else 8
    section_count = rng.choice((2, 3, 4))
    forms = {2: "AB", 3: rng.choice(("ABA", "AAB")), 4: rng.choice(("AABA", "ABAB", "ABCB"))}
    form = forms[section_count]
    progression = list(rng.choice(PROGRESSIONS[song.mode]))
    sections: dict[str, list[Measure]] = {}
    melody_state = [7 * 5 + rng.randint(4, 8)]  # 音階の上の位置 (度数 + 7 x オクターブ)。5 オクターブ目 = C4
    for label in form:
        if label in sections:
            song.measures += [_copy(m) for m in sections[label]]
            continue
        pattern = rng.choice(LH_PATTERNS)
        level = rng.choice(DYNAMIC_LEVELS[1:5])
        octave = label != "A" and rng.random() < 0.3
        bars = []
        for i in range(phrase * rng.choice((1, 2))):
            measure = Measure(song.meter, song.key, ("G2", "F4"), song.bar, swing="8th" if song.swing else "none")
            degree = progression[i % len(progression)]
            pcs = chord_pitch_classes(song, degree, rng.random() < 0.3)
            phrase_end = i % phrase == phrase - 1
            rhythm = melody_rhythm(song, rng, phrase_end)
            pitches = melody_pitches(song, rhythm, lambda _t, pcs=pcs: pcs, rng, melody_state)
            handoff = phrase_end and pattern == "handoff"  # フレーズの終わりは旋律を休み、分散和音を右手へ渡す
            if not handoff:
                right_hand(song, rhythm, pitches, lambda _t, pcs=pcs: pcs, rng, measure, octave)
                if rng.random() < song.inner_voice and not octave:
                    inner_voice(song, pcs, min(pitches), measure)
            left_hand(song, pattern, pcs, rng, measure, handoff)
            if pattern in ("block", "broken", "alberti", "two_voice", "handoff", "octave") and rng.random() < 0.9:
                measure.directions += [(F(0), "pedal_stop"), (F(0), "pedal_start")]
            if i == 0:
                measure.directions.append((F(0), f"dyn_{level}"))
            bars.append(measure)
        if rng.random() < 0.4 and len(bars) >= 2:
            bars[-1].directions.append((F(0), "wedge_crescendo"))
        if rng.random() < song.slur_phrases:
            for start in range(0, len(bars), 2):
                _slur_phrases(bars[start : start + 2], song)
        sections[label] = bars
        song.measures += bars
    _finish(song)
    return song


def _copy(measure: Measure) -> Measure:
    groups = [
        Group(g.onset, g.staff, g.voice, g.duration, [Note(n.pitch, n.alter, n.tie) for n in g.notes], g.articulations,
              g.cross, g.slur_stop, g.slur_start)
        for g in measure.groups
    ]  # fmt: skip
    return Measure(measure.time_signature, measure.key, measure.clefs, measure.length, measure.ottavas, measure.swing,
                   measure.double_bar, groups, list(measure.directions))  # fmt: skip


def _finish(song: Song) -> None:
    measures = song.measures
    beats, unit = song.meter
    compound = unit == 8 and beats % 3 == 0
    name = (
        f"metronome_quarter._{metronome_bpm(song.qpm / 1.5)}"
        if compound
        else f"metronome_quarter_{metronome_bpm(song.qpm)}"
    )
    measures[0].directions.insert(0, (F(0), name))
    measures[0].tempos = [(F(0), song.qpm)]
    # 松葉は次の小節の頭で閉じる。ペダルは曲の頭の離すを消し、最後に離す
    for i, m in enumerate(measures):
        if any(n == "wedge_crescendo" for _, n in m.directions) and i + 1 < len(measures):
            measures[i + 1].directions.insert(0, (F(0), "wedge_stop"))
    measures[0].directions = [d for d in measures[0].directions if d[1] != "pedal_stop"]
    if any(n == "pedal_start" for m in measures for _, n in m.directions):
        last = measures[-1]
        last.directions.append((last.length - min(last.length, F(1, 4)), "pedal_stop"))
    for m in measures:
        m.groups.sort(key=lambda g: (g.onset, g.staff, g.voice))
    readable_staves(measures)


def valid(song: Song, tokenizer: ScoreTokenizer) -> list[np.ndarray] | None:
    """トークンにして戻し、音の数が変わらないことを確かめる。だめなら None"""
    try:
        for m in song.measures:
            busy = {}
            for g in sorted(m.groups, key=lambda g: g.onset):
                key = (g.staff, g.voice)
                if g.onset < busy.get(key, 0) or g.end > m.length:
                    return None
                busy[key] = g.end
        patches = tokenizer.encode(song.measures)
        if max(len(p) for p in patches) > tokenizer.config.max_patch_tokens:
            return None
        decoded, _ = tokenizer.decode(patches)
    except Exception:  # noqa: BLE001 - 作れなかった曲は捨てる
        return None
    before = sum(len(g.notes) for m in song.measures for g in m.groups)
    after = sum(len(g.notes) for m in decoded for g in m.groups)
    return [np.asarray(p[1:], dtype=np.int16) for p in patches] if before == after else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--songs", type=int, default=20000)
    parser.add_argument("--out-dir", default="data/piano_score/generated")
    parser.add_argument("--samples", default=None, help="見本の MusicXML を書くフォルダ")
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-percent", type=float, default=1.0)
    args = parser.parse_args()

    from piano_score.musicxml import write_musicxml

    tokenizer = ScoreTokenizer()
    rng = random.Random(args.seed)
    tokens, lengths, seconds, counts, ids, failed = [], [], [], [], [], 0
    samples = Path(args.samples) if args.samples else None
    while len(ids) < args.songs:
        song = generate(rng)
        patches = valid(song, tokenizer)
        if patches is None:
            failed += 1
            continue
        song_id = f"synth{len(ids):06d}"
        if samples is not None and len(ids) < args.sample_count:
            write_musicxml(song.measures, samples / f"{song_id}.musicxml", title=f"{song_id} {song.meter} {song.mode}")
        tokens += patches
        lengths += [len(p) for p in patches]
        seconds.append(tokenizer.nominal_starts(song.measures).astype(np.float32))
        counts.append(len(patches))
        ids.append(song_id)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "tokens.npy", np.concatenate(tokens))
    np.savez(out / "measures.npz", token_offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
             seconds=np.concatenate(seconds))  # fmt: skip
    is_val = np.array([zlib.crc32(i.encode()) % 10000 < args.val_percent * 100 for i in ids])
    np.savez(out / "songs.npz", measure_offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
             ids=np.asarray(ids), is_val=is_val)  # fmt: skip
    meta = {"tokenizer": asdict(tokenizer.config), "visible_only": True, "vocab_size": tokenizer.vocab_size,
            "songs": len(ids), "measures": len(lengths), "failed": failed, "seed": args.seed}  # fmt: skip
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(out / "beat_songs.npz", songs=np.arange(len(ids)), spans=np.zeros(len(ids)), min_span=0.0)
    print(f"{len(ids)} 曲 ({failed} 曲は作り直し)、{len(lengths)} 小節 -> {out}")


if __name__ == "__main__":
    main()
