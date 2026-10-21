"""手順で作る合成の楽譜 (拍の上の演奏 -> 楽譜の学習データ)。

学習の楽譜コーパスは書き方の丁寧さがまちまちで (半分以上の曲にスラーがない、同じ伴奏を場所ごとに違う形で書く)、
モデルの楽譜もそれを引き継ぐ。そこで「人ならこう書く」書き方の決まりで楽譜を組み立て、描き出し (COVER_RENDER) と
組にして学習に混ぜる。和声や旋律の自然さには凝らず、次のものが表れるようにする。
    - 旋律の音高と時間の連続性: 順次進行の多い乱歩、2・4 小節のフレーズ、フレーズの終わりは長い音か休符
    - 重なる音と独立したリズム: 上の段の伸ばす内声 (声部 2、1 音か 2〜3 音の和音。同じコードの間は小節線をまたいで
      タイでつなぎ、ときどき 8 分の和音 + 休符で区切る)、下の段の伸ばす低音と刻み (声部 1・2)
    - 同時に鳴らす和音のまとまり: コードから片手で届く配置 (10 度以内) の和音。コードは曲ごとに 7 度・9 度を入れる割合を
      変え (Δ9・m9・7(b9) など)、右手の和音は 3 度・7 度・9 度で組む。コードは 1 小節に 2 つ・1 つ・2 小節に 1 つ
    - 右手の書き方を区間ごとに選ぶ: 旋律だけ / 和音つきの旋律 (音ごとに下に 2 音、8 分にスタッカート) / オクターブ /
      旋律 (4 分と 2 分にテヌート) + 下の声部の裏拍の和音 (8 分休符 + 8 分の和音、符尾は下)
    - 分散和音・オクターブなどの反復: 区間ごとに伴奏型を選び、同じ区間を繰り返すときは同じ書き方。広い分散和音
      (根音 -> 10 度 -> 5 度 -> 7 度を 8 分で、コードの区間ごとに折り返す) も入れる
    - 動くベースライン: 4 分音符で次のコードの根音へ順次進行でつなぎ、最後は半音で近づく (ときどきオクターブの跳躍)
    - 旋律はフレーズの頭を拍の休符から入ることがあり、拍の頭の音へ下から半音で近づく経過音も入れる
    - 声部の切り替わり: 内声を出し入れする頻度を曲ごとに変える
書き方の決まり (人が書くならこう書く):
    - 手をまたいで上がる分散和音 (左手から右手へ) と、レガートのまとまりにはスラー
    - 短く切る音は 16 分 + 休符ではなく、拍の音符にスタッカート (ズンチャの「チャ」など)
    - シンコペーションは、拍の区切り (4/4 の小節の中央、16 分の裏から始まる音の次の拍) をタイで見せる
    - 区間の頭に強弱記号、盛り上がる区間の前に松葉、コードの変わり目でペダルを踏み替える
    - 拍子は 4/4 を中心に 3/4・6/8・12/8・2/4・2/2・5/4・7/8 など。8 分主体の曲の一部は Swing。5/8・7/8・5/4 は
      曲ごとにまとまり (7/8 なら 2+2+3 など) を決め、旋律のリズム・伴奏の低音・分散和音の折り返し・タイの区切りを
      まとまりに合わせ、まとまりの頭にときどきアクセントを付ける (連桁は書き出しが音の並びから同じまとまりを選ぶ)
    - 両手 (と同じ段の別の声部) で同じ鍵盤を同時に鳴らさない: 旋律の一番上の音と低音を残し、他の声部の同じ音を消す
書き出す前に staves.readable_staves で 8va と音部記号を選び、綴りは調号に合わせ (tidy.spell_pitch_class)、半音で
進む調の外の音は進む向きで綴る (tidy.chromatic_spelling)。

    python -m piano_perf2score.synth_score --songs 20000 --out-dir data/piano_score/generated \\
        --samples outputs/synth_score_samples     # キャッシュ (prepare と同じ形) と見本の MusicXML
"""

from __future__ import annotations

import argparse
import json
import random
import zlib
from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from piano_score.score import Duration, Group, Measure, Note, metronome_bpm
from piano_score.tokenizer import ScoreTokenizer

from .playability import remove_unisons
from .staves import readable_staves
from .tidy import _in_key, chromatic_spelling, spell_pitch_class

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
LOWEST_RIGHT = 55  # 右手の旋律の下に付ける和音の一番下 (G3。これより下は濁るので付けない)
LH_PATTERNS = (
    "block", "oompah", "broken", "alberti", "octave", "pop8", "two_voice", "handoff", "bassline", "bassline", "wide", "wide",
)  # fmt: skip
# 右手の書き方: 旋律 (拍の頭にときどき和音) / 和音つきの旋律 / オクターブ / 旋律 + 下の声部の裏拍の和音
TEXTURES = ("plain", "plain", "chordal", "octave", "offbeat", "offbeat")
INNER_STYLES = ("note", "pad", "pad_punct")  # 内声: 1 音を伸ばす / 和音を伸ばす / 和音を伸ばして最後に 8 分で区切る
DYNAMIC_LEVELS = ("pp", "p", "mp", "mf", "f", "ff")
# 拍のまとまり (4 分音符単位の長さ) の候補。ここにない拍子は拍子で決まる (複合拍子は付点 4 分、x/4 は 4 分、2/2 は 2 分)
GROUPINGS = {
    (5, 8): ((F(1), F(3, 2)), (F(3, 2), F(1))),
    (7, 8): ((F(1), F(1), F(3, 2)), (F(1), F(3, 2), F(1)), (F(3, 2), F(1), F(1))),
    (5, 4): ((F(3), F(2)), (F(2), F(3))),
    (6, 4): ((F(3), F(3)),),
}
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
    grouping: tuple[Fraction, ...] = ()  # 拍のまとまりの長さ (4 分音符単位)
    tension: float = 0.3  # コードに 7 度を入れる割合 (9 度はその 6 割)
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
    def groups(self) -> list[tuple[Fraction, Fraction]]:
        """拍のまとまり [(小節の中の位置, 長さ)]"""
        starts = [sum(self.grouping[:i], F(0)) for i in range(len(self.grouping))]
        return list(zip(starts, self.grouping))

    @property
    def pulses(self) -> list[tuple[Fraction, Fraction]]:
        """旋律のリズムの型を選ぶ単位 [(位置, 長さ)]。2 分以上のまとまり (5/4 の 3+2、2/2) は 4 分音符に分ける"""
        result = []
        for start, length in self.groups:
            if length in (F(1, 2), F(1), F(3, 2)):
                result.append((start, length))
            else:
                result += [(start + k, F(1)) for k in range(int(length))]
        return result

    @property
    def pulse_starts(self) -> set[Fraction]:
        return {start for start, _ in self.pulses}

    @property
    def irregular(self) -> bool:
        """長さの違うまとまりのある拍子 (5/8・7/8・5/4)"""
        return len(set(self.grouping)) > 1

    @property
    def tonic(self) -> int:
        major = (7 * self.key) % 12
        return major if self.mode == "major" else (major + 9) % 12


def _choose(rng: random.Random, weighted) -> object:
    items, weights = zip(*weighted)
    return rng.choices(items, weights=weights)[0]


def default_grouping(meter: tuple[int, int]) -> tuple[Fraction, ...]:
    beats, unit = meter
    if unit == 8 and beats % 3 == 0:
        return (F(3, 2),) * (beats // 3)
    if unit == 2:
        return (F(2),) * beats
    return (F(4, unit),) * beats


def new_song(rng: random.Random) -> Song:
    meter = _choose(rng, METERS)
    beat_unit = meter[1]
    compound = beat_unit == 8 and meter[0] % 3 == 0
    grouping = rng.choice(GROUPINGS[meter]) if meter in GROUPINGS else default_grouping(meter)
    grid = F(1, 2) if compound or rng.random() < 0.45 else F(1, 4)
    swing = not compound and grid == F(1, 2) and meter in ((4, 4), (2, 4), (2, 2)) and rng.random() < 0.3
    return Song(
        meter=meter, key=rng.randint(-6, 6), mode="major" if rng.random() < 0.65 else "minor",
        qpm=rng.uniform(64, 176) * (1.5 if compound else 1.0), grid=grid, swing=swing,
        detached=rng.choice((0.0, 0.0, 0.15, 0.4)), syncopation=rng.uniform(0.0, 0.5),
        inner_voice=rng.choice((0.0, 0.0, 0.3, 0.7)), slur_phrases=rng.choice((0.3, 0.6, 0.9)), grouping=grouping,
        tension=rng.choice((0.0, 0.3, 0.7)),
    )  # fmt: skip


# ----------------------------------------------------------------------
# 音価の書き方
# ----------------------------------------------------------------------
def notate(onset: Fraction, length: Fraction, song: Song) -> list[tuple[Fraction, Duration]]:
    """小節の中の onset から length の音を、書き方の決まりでタイの鎖 [(位置, 音価)] にする"""
    pieces = []
    bar = song.bar
    starts = [start for start, _ in song.pulses] + [bar]
    end = min(onset + length, bar)
    while onset < end:
        stop = end
        index = bisect_right(starts, onset) - 1
        pulse_start, next_pulse = starts[index], starts[index + 1]
        if song.meter == (4, 4) and onset < 2 < stop and onset != 0 and not (onset == 1 and stop == 3):
            stop = F(2)  # 4/4 の小節の中央はまたがない (頭からの音と 4 分-2 分-4 分のシンコペーションは除く)
        if onset != pulse_start:
            # 4 分の拍の 8 分の裏から始まる音は次の拍の終わりまで、それ以外 (16 分の裏・まとまりの途中) は次の拍まで
            eighth_offbeat = (onset - pulse_start) % F(1, 2) == 0 and next_pulse - pulse_start == 1
            limit = starts[min(index + 2, len(starts) - 1)] if eighth_offbeat else next_pulse
            stop = min(stop, limit)
        elif stop - onset > next_pulse - pulse_start and stop not in starts:
            stop = max(x for x in starts if x <= stop)  # 拍の頭からの長い音は拍の区切りまで書き、残りをタイで
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
def chord_pitch_classes(song: Song, degree: int, seventh: bool, ninth: bool = False) -> list[int]:
    """度数のコードの音名 [根音, 3 度, 5 度 (, 7 度 (, 9 度))]。9 度を入れるときは 7 度も入れる"""
    scale = SCALES[song.mode]
    steps = (0, 2, 4) + ((6,) if seventh or ninth else ()) + ((8,) if ninth else ())
    pcs = [(song.tonic + scale[(degree - 1 + k) % 7]) % 12 for k in steps]
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


def color_voicing(pcs: list[int], top: int, size: int) -> list[int]:
    """top より下に置く右手の和音 (size 音、1 オクターブ以内)。7 度・9 度があれば 3 度・7 度・9 度、なければ 3 度・5 度・根音"""
    order = (
        [pcs[1], pcs[3], pcs[4]]
        if len(pcs) >= 5
        else [pcs[1], pcs[3], pcs[2]]
        if len(pcs) == 4
        else [pcs[1], pcs[2], pcs[0]]
    )
    notes = []
    for pc in order[:size]:
        pitch = top - 1
        while pitch % 12 != pc:
            pitch -= 1
        notes.append(pitch)
    notes = [p for p in notes if top - p <= 12]
    return sorted(notes)


# ----------------------------------------------------------------------
# 旋律
# ----------------------------------------------------------------------
def beat_cells(song: Song, length: Fraction) -> list[list[tuple[Fraction, Fraction]]]:
    """長さ length の拍のリズムの型 [(拍の中の位置, 長さ)]"""
    if length == F(3, 2):
        return [[(F(0), F(3, 2))], [(F(0), F(1)), (F(1), F(1, 2))], [(F(0), F(1, 2)), (F(1, 2), F(1, 2)), (F(1), F(1, 2))],
                [(F(0), F(1, 2)), (F(1, 2), F(1))], [(F(0), F(3, 4)), (F(3, 4), F(1, 4)), (F(1), F(1, 2))]]  # fmt: skip
    if length == F(1, 2):
        return [[(F(0), F(1, 2))], [(F(0), F(1, 4)), (F(1, 4), F(1, 4))]]
    cells = [[(F(0), F(1))], [(F(0), F(1, 2)), (F(1, 2), F(1, 2))], [(F(1, 2), F(1, 2))]]
    if song.grid == F(1, 4):
        cells += [[(F(0), F(3, 4)), (F(3, 4), F(1, 4))], [(F(0), F(1, 4)), (F(1, 4), F(1, 2)), (F(3, 4), F(1, 4))],
                  [(F(0), F(1, 4)), (F(1, 4), F(1, 4)), (F(1, 2), F(1, 2))], [(F(0), F(1, 2)), (F(1, 2), F(1, 4)), (F(3, 4), F(1, 4))],
                  [(F(0), F(1, 4)), (F(1, 4), F(1, 4)), (F(1, 2), F(1, 4)), (F(3, 4), F(1, 4))]]  # fmt: skip
    return cells


def melody_rhythm(
    song: Song, rng: random.Random, phrase_end: bool, phrase_start: bool = False, simple: bool = False
) -> list[tuple[Fraction, Fraction]]:
    """1 小節の旋律の (位置, 長さ)。フレーズの終わりは長い音と休符。フレーズの頭は拍の休符から入ることがある
    (4 分休符 + 4 分音符の並びなど)。simple なら拍の音符 (4 分と、ときどき 2 分や 8 分 2 つ) で歌う
    (下の声部で裏拍の和音を刻むとき)"""
    pulses = song.pulses
    if phrase_end:
        held = song.bar if len(pulses) < 2 or rng.random() < 0.5 else song.bar - pulses[-1][1]
        return [(F(0), held)]
    notes = []
    rest_start = phrase_start and len(pulses) >= 2 and rng.random() < 0.35
    plain = rest_start and rng.random() < 0.5  # 休符のあとは拍の音符だけで歌う
    skip = False
    for k, (start, length) in enumerate(pulses[1 if rest_start else 0 :], start=1 if rest_start else 0):
        if skip:
            skip = False
            continue
        cells = beat_cells(song, length)
        if simple and length == 1 and k + 1 < len(pulses) and pulses[k + 1][1] == 1 and rng.random() < 0.25:
            notes.append((start, F(2)))  # 2 拍を 2 分音符で
            skip = True
            continue
        if plain or (simple and rng.random() < 0.65):
            cell = cells[0]
        elif simple:
            cell = cells[1] if len(cells) > 1 else cells[0]
        else:
            cell = rng.choice(cells[:3]) if rng.random() < 0.4 else rng.choice(cells)
        notes += [(start + a, d) for a, d in cell]
    # シンコペーション: 4 分の拍の頭の音を消して、前の裏の音を伸ばす
    if not simple and all(length == 1 for _, length in pulses) and rng.random() < song.syncopation:
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
        if onset in song.pulse_starts and rng.random() < 0.7:
            chord = chords(onset)[:3]
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
    song: Song,
    pattern: str,
    spans: list[tuple[Fraction, Fraction, list[int], list[int]]],
    rng: random.Random,
    measure: Measure,
    melody_rests: bool,
    bass: dict | None = None,
) -> None:
    """1 小節の左手 (と handoff では右手の上の分散和音) を measure に足す。spans は小節の中のコードの区間
    [(位置, 長さ, コードの音名, 次のコードの音名)] で、伴奏型は区間ごとに頭から組み立てる (1 小節に 2 つのコード)。
    bass はベースラインの続き (次の小節の頭の音) を小節の間で受け渡す"""
    bass = {} if bass is None else bass
    octave_unit = song.beat if rng.random() < 0.5 else min(song.beat, F(1, 2))
    staccato = ("staccato",) * (rng.random() < 0.6)
    for span_start, bar, pcs, next_pcs in spans:

        def put(t, length, pitches, staff=2, voice=1, articulations=(), span_start=span_start):
            groups = _groups(span_start + t, length, pitches, staff, voice, song, articulations)
            measure.groups.extend(groups)
            return groups

        groups = [(s - span_start, n) for s, n in song.groups if span_start <= s < span_start + bar]
        pulses = [(s - span_start, n) for s, n in song.pulses if span_start <= s < span_start + bar]
        root = 36 + (pcs[0] - 36) % 12  # C2〜B2
        chord = voicing(pcs[:3], 45 + (pcs[0] - 45) % 12 if rng.random() < 0.5 else 50, 3)  # A2〜D4 から
        if pattern == "block":
            put(F(0), bar, [root, *chord])
        elif pattern == "oompah":
            if song.irregular:
                # まとまりごとに、頭で低音、残りで和音 (7/8 の 2+2+3 なら 低-和 低-和 低-和和)
                for start, length in groups:
                    head = F(1, 2) if song.meter[1] == 8 else F(1)
                    put(start, head, [root])
                    put(start + head, length - head, chord, articulations=staccato)
            else:
                for k, (start, length) in enumerate(groups):
                    low = k == 0 or (len(groups) != 3 and k % 2 == 0)  # 3 拍子は 低-和-和
                    put(start, length, [root] if low else chord, articulations=() if low else staccato)
        elif pattern in ("broken", "alberti", "handoff", "wide"):
            unit = F(1, 2)
            if pattern == "alberti":
                cycle = [chord[0], chord[-1], chord[1], chord[-1]]
            elif pattern == "wide":
                # 広い分散和音: 根音 -> 10 度 -> 5 度 (+8 度) -> 7 度か 10 度。上の音が高くなりすぎないよう根音は B♭1〜A2
                root = 34 + (pcs[0] - 34) % 12
                tenth = next(p for p in range(root + 13, root + 25) if p % 12 == pcs[1])
                fifth = next(p for p in range(root + 13, root + 25) if p % 12 == pcs[2])
                top = next((p for p in range(tenth + 1, tenth + 13) if len(pcs) > 3 and p % 12 == pcs[3]), tenth)
                cycle = [root, tenth, fifth, top]
            else:
                third = next(p for p in range(root + 13, root + 25) if p % 12 in pcs[1:3])  # オクターブ上の 3 度か 5 度
                cycle = [root, root + 7, root + 12, third]
                cycle += cycle[-2:0:-1]
            steps = int(bar / unit)
            upper = pattern == "handoff" and melody_rests and len(spans) == 1
            line = [cycle[i % len(cycle)] for i in range(steps)]
            if song.irregular:  # まとまりの頭で分散和音を折り返す
                line = [cycle[i % len(cycle)] for _, length in groups for i in range(int(length / unit))]
            if upper:  # 手をまたいで上がる分散和音: 後半を右手で 1 オクターブ上へ
                line = [root, root + 7, *[p for k in range(4) for p in voicing(pcs[:3], root + 12 * (1 + k), 3)]]
                line = line[:steps]
            for i, pitch in enumerate(line):
                group = put(i * unit, unit, [pitch], staff=1 if upper and pitch >= 60 else 2)[0]
                if pattern != "wide" and i == 0:
                    group.slur_start = 1
                if pattern != "wide" and i == len(line) - 1:
                    group.slur_stop = 1
        elif pattern == "octave":
            t = F(0)
            while t < bar:
                put(t, octave_unit, [root, root + 12])
                t += octave_unit
        elif pattern == "pop8":
            hits = {F(4): [F(0), F(3, 2), F(2), F(7, 2)], F(2): [F(0), F(3, 2)]}.get(bar, [F(0)])
            for i, t in enumerate(hits):
                length = (hits[i + 1] if i + 1 < len(hits) else bar) - t
                put(t, length, [root] if i % 2 == 0 else chord)
        elif pattern == "bassline":
            units = groups if song.irregular else pulses  # まとまり (7/8 の 2+2+3) か拍ごとに 1 音
            steps = len(units)
            start = bass.get("next")
            if start is None or start % 12 != pcs[0] or not 36 <= start <= 55:  # 音域から外れたら戻す
                start = 40 + (pcs[0] - 40) % 12  # E2〜D#3
            goal = next_pcs[0]
            target = min((p for p in range(start - 6, start + 7) if p % 12 == goal), key=lambda p: abs(p - start))
            scale = [p for p in range(24, 72) if _in_key(p % 12, song.key)]
            if target == start:  # 同じコードが続くときは、コードの音を上がって戻る
                tones = [p for p in range(start + 1, start + 13) if p % 12 in pcs[1:3]]
                line = ([start, *tones, start + 12, *tones[::-1]] * 2)[:steps]
            else:
                line = [start]
                for k in range(1, steps):
                    current = line[-1]
                    if k == steps - 1 and rng.random() < 0.5:
                        line.append(target - 1 if target >= current else target + 1)  # 次の根音へ半音で近づく
                        continue
                    direction = (target > current) - (target < current) or rng.choice((1, -1))
                    ordered = scale if direction > 0 else scale[::-1]
                    line.append(next(p for p in ordered if (p - current) * direction > 0))
            if steps >= 3 and rng.random() < 0.25:
                k = rng.randrange(1, steps - 1)
                line[k] += 12 if line[k] < 48 else -12  # オクターブの跳躍 (次の音への半音の進みは崩さない)
            bass["next"] = target
            for (start, length), pitch in zip(units, line):
                put(start, length, [pitch])
        elif pattern == "two_voice":
            put(F(0), bar, [root])
            beat = song.beat
            t = beat if beat <= 1 else F(1, 2)
            unit = min(beat, F(1, 2)) if beat != 2 else F(1)
            while t < bar:
                put(t, unit, chord, voice=2)
                t += unit


def right_hand(song: Song, rhythm, pitches, chords, rng: random.Random, measure: Measure, texture: str) -> None:
    """旋律 (上の段の声部 1)。texture: plain (拍の頭にときどき和音) / chordal (音ごとに下に 2 音の和音) /
    octave (オクターブで重ねる) / offbeat (旋律だけ。下の声部で裏拍の和音を刻む)"""
    for (onset, length), pitch in zip(rhythm, pitches):
        notes = [pitch]
        if texture == "octave":
            notes.append(pitch - 12)
        elif texture == "chordal" and length >= F(1, 2):
            notes += [p for p in color_voicing(chords(onset), pitch, 2) if p >= LOWEST_RIGHT]
        elif texture == "plain" and onset in song.pulse_starts and rng.random() < 0.35:
            below = [p for p in range(pitch - 9, pitch) if p % 12 in chords(onset)[:3]]
            notes += below[-rng.choice((1, 2)) :] if below else []
        articulations = ()
        group_head = song.irregular and onset in {start for start, _ in song.groups[1:]}
        if texture == "offbeat" and length >= 1 and rng.random() < 0.6:
            articulations = ("tenuto",)  # 裏拍の和音の上で歌う旋律の拍の音
        elif texture == "chordal" and length == F(1, 2) and rng.random() < 0.35:
            articulations = ("staccato",)
        elif texture == "chordal" and length >= 1 and rng.random() < 0.3:
            articulations = ("tenuto",)
        elif rng.random() < song.detached and length <= max(song.grouping):
            articulations = ("staccato",)  # 短く切る音は音価を詰めずにスタッカートで書く
        elif group_head and rng.random() < 0.25:
            articulations = ("accent",)  # 変拍子のまとまりの頭を見せる
        elif onset not in song.pulse_starts and rng.random() < 0.05:
            articulations = ("accent",)
        measure.groups.extend(_groups(onset, length, notes, 1, 1, song, articulations))


def offbeat_chords(song: Song, chords, melody_low: int, measure: Measure, rng: random.Random) -> None:
    """上の段の下の声部 (声部 2) で裏拍の和音を刻む: 4 分の拍は 8 分休符 + 8 分の和音、付点 4 分の拍は裏の 8 分 2 つ"""
    top = min(melody_low, 72)
    if top <= 62:
        return
    staccato = ("staccato",) * (rng.random() < 0.4)
    for start, length in song.pulses:
        offsets = {F(1): [F(1, 2)], F(3, 2): [F(1, 2), F(1)]}.get(length, [])
        for offset in offsets:
            notes = [p for p in color_voicing(chords(start + offset), top, rng.choice((2, 3))) if p >= LOWEST_RIGHT]
            if notes:
                measure.groups.extend(_groups(start + offset, F(1, 2), notes, 1, 2, song, staccato))


def inner_voice(
    song: Song, pcs: list[int], melody_low: int, measure: Measure, style: str, held: list[Group] | None
) -> list[Group] | None:
    """上の段の内声 (声部 2)。旋律より下のコードの音を、style で 1 音か和音にして伸ばす。前の小節の終わりまで伸ばした
    和音 held と同じ音で始まるときは、小節線をまたいでタイでつなぐ。小節の終わりまで伸ばした塊を返す"""
    if melody_low <= 62:
        return None
    below = [p for p in range(melody_low - 10, melody_low - 1) if p % 12 in pcs[:3]]
    if not below:
        return None
    pitches = below[-1:] if style == "note" else below[-3:]
    bar, beat = song.bar, song.beat
    if style == "note":
        half = bar / 2 if bar >= 2 and song.meter[0] % 2 == 0 else bar
        spans = [(t * half, half) for t in range(int(bar / half))]
    elif style == "pad_punct" and beat == 1 and bar >= 3:
        spans = [(F(0), bar - 1), (bar - 1, F(1, 2))]  # 伸ばして、最後の拍の 8 分の和音で区切る (後は休符)
    else:
        spans = [(F(0), bar)]
    groups = []
    for onset, length in spans:
        groups += _groups(onset, length, pitches, 1, 2, song)
    if held and sorted(n.pitch for n in held[-1].notes) == sorted(pitches):
        for note in held[-1].notes:
            note.tie = True
    measure.groups.extend(groups)
    return groups if groups[-1].end == bar else None


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
        texture = rng.choice(TEXTURES) if label != "A" or rng.random() < 0.6 else "plain"
        inner_style = rng.choice(INNER_STYLES)
        # コードは 1 小節に 2 つ (半分ずつ)、1 つ、2 小節に 1 つ (内声の和音が小節線をまたいでタイでつながる)
        even = not song.irregular and len(song.groups) % 2 == 0 and song.bar >= 2
        rate = rng.choices(("half", "bar", "slow"), weights=(0.35 if even else 0.0, 0.4, 0.25))[0]
        count = phrase * rng.choice((1, 2))
        per_bar = 2 if rate == "half" else 1
        slots = []
        for k in range(count * per_bar + 1):
            degree = progression[(k // 2 if rate == "slow" else k) % len(progression)]
            ninth = rng.random() < song.tension * 0.6
            slots.append(chord_pitch_classes(song, degree, rng.random() < song.tension, ninth))
        if rate == "slow":
            slots = [slots[k - k % 2] for k in range(len(slots))]
        bars, held, bass = [], None, {}
        for i in range(count):
            measure = Measure(song.meter, song.key, ("G2", "F4"), song.bar, swing="8th" if song.swing else "none")
            half = song.bar / per_bar
            spans = [(half * j, half, slots[i * per_bar + j], slots[i * per_bar + j + 1]) for j in range(per_bar)]

            def chords(t, spans=spans):
                return next(pcs for start, length, pcs, _ in spans if start <= t < start + length)

            phrase_end = i % phrase == phrase - 1
            rhythm = melody_rhythm(song, rng, phrase_end, phrase_start=i % phrase == 0, simple=texture == "offbeat")
            pitches = melody_pitches(song, rhythm, chords, rng, melody_state)
            for k in range(len(rhythm) - 1):
                # 拍の頭の音へ下から半音で近づく経過音 (裏の音を置き換える)
                on, nxt = rhythm[k][0], rhythm[k + 1][0]
                approach = pitches[k + 1] - 1
                if (
                    on not in song.pulse_starts
                    and nxt in song.pulse_starts
                    and rng.random() < 0.1
                    and not _in_key(approach % 12, song.key)
                ):
                    pitches[k] = approach
            handoff = phrase_end and pattern == "handoff"  # フレーズの終わりは旋律を休み、分散和音を右手へ渡す
            held_next = None
            if not handoff:
                right_hand(song, rhythm, pitches, chords, rng, measure, texture)
                if texture == "offbeat" and not phrase_end:
                    offbeat_chords(song, chords, min(pitches), measure, rng)
                elif texture == "plain" and per_bar == 1 and rng.random() < song.inner_voice:
                    # 和音を 2 小節伸ばすときは、1 小節目は伸ばしきり 2 小節目の終わりで区切る
                    style = "pad" if inner_style == "pad_punct" and rate == "slow" and i % 2 == 0 else inner_style
                    held_next = inner_voice(song, spans[0][2], min(pitches), measure, style, held)
            held = held_next
            left_hand(song, pattern, spans, rng, measure, handoff, bass)
            if (
                pattern in ("block", "broken", "alberti", "two_voice", "handoff", "octave", "wide")
                and rng.random() < 0.9
            ):
                for start, _, _, _ in spans:
                    measure.directions += [(start, "pedal_stop"), (start, "pedal_start")]
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
    remove_unisons(measures)
    readable_staves(measures)
    chromatic_spelling(measures)


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
