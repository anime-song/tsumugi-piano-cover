"""楽譜の中間表現。MusicXML の読み書き (musicxml.py) とトークン化 (tokenizer.py) の間で使う。

時刻はすべて 4 分音符 = 1 の Fraction で、小節の先頭からの位置で持つ。反復記号は展開済みで、小節は演奏順に並ぶ。
段 (staff) は 1 = 上段 (右手), 2 = 下段 (左手)。声部 (voice) は段ごとに 1 から振り直した番号。
休符・符尾・連桁・臨時記号の表示・連符括弧は持たない (書き出すときに規則で付ける)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

# 音符の種類と長さ (4 分音符 = 1)
NOTE_TYPES = {
    "breve": Fraction(8),
    "whole": Fraction(4),
    "half": Fraction(2),
    "quarter": Fraction(1),
    "eighth": Fraction(1, 2),
    "16th": Fraction(1, 4),
    "32nd": Fraction(1, 8),
    "64th": Fraction(1, 16),
    "128th": Fraction(1, 32),
}
# 連符の比 (actual, normal)。3:2 と 6:4 は長さが同じでも書き方 (括弧の数字) が違うので分ける
TUPLETS = (
    (3, 2),
    (6, 4),
    (5, 4),
    (7, 4),
    (7, 8),
    (9, 8),
    (2, 3),
    (4, 3),
    (5, 2),
    (5, 3),
    (10, 8),
    (11, 8),
    (12, 8),
    (13, 8),
)
# 連符に使う音符の種類と付点
TUPLET_TYPES = ("half", "quarter", "eighth", "16th", "32nd", "64th")
TUPLET_DOTS = (0, 1)
# 装飾音: 種類 × 斜線 (acciaccatura) の有無
GRACE_TYPES = ("quarter", "eighth", "16th", "32nd")

DYNAMICS = (
    "pppp",
    "ppp",
    "pp",
    "p",
    "mp",
    "mf",
    "f",
    "ff",
    "fff",
    "ffff",
    "fp",
    "sf",
    "sfz",
    "sffz",
    "fz",
    "rf",
    "rfz",
    "sfp",
)
# 同じ位置では stop を start より先に並べる (ペダルの踏み替えは stop -> start)
WEDGES = ("stop", "crescendo", "diminuendo")
PEDALS = ("stop", "start")
# 音部記号は「記号 + 線 (+ オクターブ移動)」の名前で持つ
CLEFS = ("G2", "F4", "G2+1", "G2-1", "F4-1", "G2+2", "F4+1", "F4-2", "F3", "F5", "C3", "C4", "G1")
DEFAULT_CLEFS = ("G2", "F4")
# 和音 (塊) ごとの奏法記号と装飾記号 (名前は MusicXML の要素名)。tremoloN は符尾の斜線 N 本のトレモロ
ORNAMENTS = ("trill-mark", "mordent", "inverted-mordent", "turn", "inverted-turn")
TREMOLOS = ("tremolo1", "tremolo2", "tremolo3", "tremolo4")
ARTICULATIONS = (
    ("staccato", "staccatissimo", "accent", "strong-accent", "tenuto", "fermata", "arpeggiate") + ORNAMENTS + TREMOLOS
)
# 2 音間のトレモロ。MuseScore は 2 音とも「書いた音価の半分の長さ」(連符比 2:1) で書き出す
TREMOLO_PAIR = (2, 1)
TREMOLO_PAIR_TYPES = ("whole", "half", "quarter", "eighth", "16th")
# オクターブ記号 (段ごとの状態)。8va / 15ma は実音が記譜より高い
OTTAVAS = ("none", "8va", "15ma", "8vb", "15mb")
# スイングの指定 (曲全体の状態)。楽譜はまっすぐな 8 分 (16 分) で書き、この指定で長短を付けて弾く
SWINGS = ("none", "8th", "16th")

STEPS = "CDEFGAB"
STEP_PITCH_CLASS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
PITCH_CLASS_STEP = {v: k for k, v in STEP_PITCH_CLASS.items()}
SHARP_ORDER = "FCGDAEB"


@dataclass(frozen=True)
class Duration:
    """音価。grace は装飾音 (None / "grace" / "slash")"""

    type: str
    dots: int = 0
    tuplet: tuple[int, int] | None = None
    grace: str | None = None

    @property
    def quarters(self) -> Fraction:
        if self.grace:
            return Fraction(0)
        value = NOTE_TYPES[self.type] * (2 - Fraction(1, 2**self.dots))
        if self.tuplet:
            value = value * self.tuplet[1] / self.tuplet[0]
        return value


def all_durations() -> list[Duration]:
    """トークンにする音価の一覧"""
    durations = [Duration(t, d) for t in NOTE_TYPES for d in (0, 1, 2, 3)]
    durations += [Duration(t, d, r) for r in TUPLETS for t in TUPLET_TYPES for d in TUPLET_DOTS]
    durations += [Duration(t, 0, None, g) for g in ("grace", "slash") for t in GRACE_TYPES]
    durations += [Duration(t, d, TREMOLO_PAIR) for t in TREMOLO_PAIR_TYPES for d in (0, 1)]
    return durations


@dataclass
class Note:
    pitch: int  # MIDI 番号 (実音)
    alter: int  # 綴り (-2..2)。幹音は pitch - alter
    tie: bool = False  # 次の同じ音へタイでつながる
    glissando: bool = False  # 同じ声部の次の音へグリッサンドでつながる (波線・直線とも)


@dataclass
class Group:
    """同じ位置・同じ声部で鳴り始める和音 (1 音でもよい)。装飾音も 1 つの塊にする"""

    onset: Fraction
    staff: int
    voice: int
    duration: Duration
    notes: list[Note]
    articulations: tuple[str, ...] = ()
    cross: bool = False  # もう一方の段に書かれている (段をまたぐ声部)
    # この塊で終わるスラーと始まるスラーの数。対の相手は持たず pair_slurs の規則で決める
    slur_stop: int = 0
    slur_start: int = 0

    @property
    def end(self) -> Fraction:
        return self.onset + self.duration.quarters

    @property
    def display_staff(self) -> int:
        return 3 - self.staff if self.cross else self.staff


@dataclass
class Measure:
    time_signature: tuple[int, int]
    key: int  # 五度圏上の位置 (-7..7)
    clefs: tuple[str, str]  # 小節の頭の音部記号 (上段, 下段)
    length: Fraction
    ottavas: tuple[str, str] = ("none", "none")  # 小節の頭のオクターブ記号の状態 (上段, 下段)
    swing: str = "none"  # 小節の頭のスイングの状態
    groups: list[Group] = field(default_factory=list)
    # 小節内の指示 (位置, 名前)。名前は dyn_p / wedge_crescendo / pedal_start / clef1_G2 など
    directions: list[tuple[Fraction, str]] = field(default_factory=list)
    # テンポ (位置, 4 分音符/分)。トークンには入れない (演奏時刻の見積もりと書き出し用)
    tempos: list[tuple[Fraction, float]] = field(default_factory=list)

    @property
    def nominal_length(self) -> Fraction:
        beats, beat_type = self.time_signature
        return Fraction(4 * beats, beat_type)


def group_order(group: Group) -> tuple:
    """小節の中の塊の並び (位置 -> 段 -> 声部、同じ声部では装飾音が先)。同じ値の塊は元の順のまま (安定ソート)"""
    return (group.onset, group.staff, group.voice, group.duration.grace is None)


def pair_slurs(measures: list[Measure]) -> list[tuple[int, Group, str, int]]:
    """スラーの始まりと終わりを対にする。

    小節と塊を並びの順に見て、終わりは「同じ声部でいちばん最近開いたスラー」を閉じる。同じ声部に開いたものがなければ
    同じ段、それもなければ全体でいちばん最近開いたものを閉じる (声部や段をまたぐスラー)。同じ塊では終わりを先に見る。
    返り値は出てくる順の (小節番号, 塊, "start" / "stop", スラーの番号) で、相手のない始まりと終わりは含めない。
    読み込み (相手のないものを消す) と書き出し (MusicXML の番号を振る) で同じ規則を使う。
    """
    events: list[tuple[int, Group, str, int]] = []
    open_slurs: list[tuple[int, Group]] = []  # (スラーの番号, 始まりの塊) を開いた順に
    next_id = 0
    for mi, measure in enumerate(measures):
        for g in sorted(measure.groups, key=group_order):
            for _ in range(g.slur_stop):
                index = None
                for same in (
                    lambda s: (s.staff, s.voice) == (g.staff, g.voice),
                    lambda s: s.staff == g.staff,
                    lambda s: True,
                ):
                    index = next((i for i in range(len(open_slurs) - 1, -1, -1) if same(open_slurs[i][1])), None)
                    if index is not None:
                        break
                if index is not None:
                    slur_id, _ = open_slurs.pop(index)
                    events.append((mi, g, "stop", slur_id))
            for _ in range(g.slur_start):
                open_slurs.append((next_id, g))
                events.append((mi, g, "start", next_id))
                next_id += 1
    closed = {slur_id for _, _, kind, slur_id in events if kind == "stop"}
    return [event for event in events if event[3] in closed]


def spell(pitch: int, alter: int) -> tuple[str, int]:
    """MIDI 番号と変化記号から (幹音名, オクターブ) を返す。綴りとして成り立たなければ ValueError"""
    natural = pitch - alter
    step = PITCH_CLASS_STEP.get(natural % 12)
    if step is None:
        raise ValueError(f"綴りにならない音: pitch={pitch} alter={alter}")
    return step, natural // 12 - 1


def key_alters(key: int) -> dict[str, int]:
    """調号で変化する幹音と変化量"""
    if key >= 0:
        return {step: 1 for step in SHARP_ORDER[:key]}
    return {step: -1 for step in SHARP_ORDER[::-1][:-key]}


def measure_seconds(measures: list[Measure], default_qpm: float = 120.0) -> np.ndarray:
    """楽譜のテンポ記号どおりに弾いたときの各小節の開始時刻 (秒)。最後に曲の終わりの時刻を足した M+1 個"""
    starts = np.zeros(len(measures) + 1)
    qpm = default_qpm
    seconds = 0.0
    for i, measure in enumerate(measures):
        starts[i] = seconds
        position = Fraction(0)
        for at, tempo in sorted(measure.tempos):
            seconds += float(at - position) * 60.0 / qpm
            position, qpm = at, tempo
        seconds += float(measure.length - position) * 60.0 / qpm
    starts[-1] = seconds
    return starts


def first_onset_seconds(measures: list[Measure], starts: np.ndarray) -> float:
    """最初の音の時刻 (measure_seconds の時間軸)。音がなければ 0"""
    for i, measure in enumerate(measures):
        if measure.groups:
            onset = min(g.onset for g in measure.groups)
            span = starts[i + 1] - starts[i]
            return float(starts[i] + span * float(onset / measure.length))
    return 0.0
