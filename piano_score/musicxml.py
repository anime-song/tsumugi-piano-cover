"""MusicXML <-> 楽譜の中間表現 (score.py)。

読み込みは MuseScore が書き出す形 (1 パート 2 段、または 1 段ずつの 2 パート) を前提にする。
学習に使えない楽譜 (段の数が違う、対応していない音価、声部の中で音が重なるなど) は ScoreError にする。

書き出しでは、トークンに入れていない休符・符尾・連桁・臨時記号の表示・連符括弧を規則で付ける。
"""

from __future__ import annotations

import copy
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

from .config import ScoreTokenizerConfig
from .score import (
    ARTICULATIONS,
    CLEFS,
    DEFAULT_CLEFS,
    DYNAMIC_WORDS,
    DYNAMICS,
    METRONOME_UNITS,
    NOTE_TYPES,
    ORNAMENTS,
    STEP_PITCH_CLASS,
    TEMPO_MARKS,
    TEMPO_WORDS,
    TREMOLO_PAIR,
    TREMOLOS,
    TUPLETS,
    Duration,
    Group,
    Measure,
    Note,
    group_order,
    key_alters,
    metronome_bpm,
    metronome_quarters,
    pair_slurs,
    spell,
)


class ScoreError(Exception):
    """学習に使えない楽譜"""


# ----------------------------------------------------------------------
# 読み込み
# ----------------------------------------------------------------------
@dataclass
class _RawGroup:
    measure: int
    onset: Fraction
    raw_voice: tuple[int, str]  # (パート, voice 要素の値)
    duration: Duration
    notes: list[Note]
    display_staves: list[int]
    articulations: set[str]
    order: int
    slurs: set[tuple[str, str]] = field(default_factory=set)  # (start / stop, number)


@dataclass
class _RawMeasure:
    time_signature: tuple[int, int] | None = None
    key: int | None = None
    length: Fraction = Fraction(0)
    # (位置, 種類, 値): 種類は clef (値 = (段, 名前)) / direction / tempo
    events: list[tuple[Fraction, str, object]] = field(default_factory=list)
    forward_repeat: bool = False
    backward_repeat: int = 0  # 繰り返す回数 (0 なら反復記号なし)
    ending_start: frozenset[int] | None = None
    ending_stop: bool = False
    double_bar: bool = False  # 小節の終わりが区切りの複縦線 (反復記号のない二重線)


def _clef_name(clef: ET.Element) -> str:
    name = f"{clef.findtext('sign')}{clef.findtext('line')}"
    change = int(clef.findtext("clef-octave-change") or 0)
    if change:
        name += f"{change:+d}"
    if name not in CLEFS:
        raise ScoreError(f"対応していない音部記号 {name}")
    return name


def _ottava_name(shift: ET.Element) -> str:
    """octave-shift 要素をオクターブ記号の状態名にする。type="down" は記譜を下げる = 実音が高い (8va)"""
    kind, size = shift.get("type"), shift.get("size", "8")
    if kind == "stop":
        return "none"
    number = "8" if size == "8" else "15"
    if kind == "down":
        return number + ("va" if number == "8" else "ma")
    return number + ("vb" if number == "8" else "mb")


SWING_OFF = re.compile(r"swing\s*off|no\s*swing|remove\s*swing|without\s*swing|straight")
SWING_ON = re.compile(r"swing|shuffle")


def _swing_from_words(text: str) -> str | None:
    """演奏指示の文字からスイングの状態を読む。"Swing" / "Swing 16ths" / "Swing off" / "Straight" など。関係なければ None"""
    text = text.lower()
    if SWING_OFF.search(text):
        return "none"
    if SWING_ON.search(text):
        return "16th" if "16" in text else "8th"
    return None


def _swing_from_sound(swing: ET.Element) -> str:
    """<sound><swing> (再生用の指定) からスイングの状態を読む"""
    if swing.find("straight") is not None:
        return "none"
    return "16th" if swing.findtext("swing-type") == "16th" else "8th"


def _swing_from_metronome(metronome: ET.Element) -> str | None:
    """<metronome> の音符の等式 (♫ = 3 連符の ♩♪ など) からスイングの状態を読む。等式でなければ None"""
    if metronome.find("metronome-relation") is None:
        return None
    notes = metronome.findall("metronome-note")
    if not notes:
        return None
    if not any(n.find("metronome-tuplet") is not None for n in notes):
        return "none"  # ♫ = ♫ (まっすぐに戻す)
    return "16th" if notes[0].findtext("metronome-type") == "16th" else "8th"


# 演奏指示の文字から拾う速度と強弱の変化。1 つの文字に複数あれば全部拾う ("rit. e dim." など)
WORD_PATTERNS = (
    ("tempo_rit", re.compile(r"\brit\b|\britard|\briten")),
    ("tempo_rall", re.compile(r"\brall")),
    ("tempo_accel", re.compile(r"\baccel")),
    ("tempo_a_tempo", re.compile(r"\ba tempo\b|\btempo (i|1|primo)\b")),
    ("tempo_rubato", re.compile(r"\brubato")),
    ("tempo_allarg", re.compile(r"\ballarg")),
    ("tempo_string", re.compile(r"\bstring(\.|endo)")),
    ("tempo_meno_mosso", re.compile(r"\bmeno mosso")),
    ("tempo_piu_mosso", re.compile(r"\bpi[uù] mosso")),
    ("text_cresc", re.compile(r"\bcresc")),
    ("text_dim", re.compile(r"\bdim\b|\bdimin|\bdecresc|\bdecr\b")),
)
# 速度標語。長い語を先に並べ (Prestissimo を Presto と読まないように)、文字の中で最初に出てくるものを 1 つ取る
TEMPO_MARK_PATTERN = re.compile(
    r"\b(prestissimo|presto|larghetto|largo|allegretto|allegro|andantino|andante|adagio|lento|grave|moderato"
    r"|vivace|vivo|maestoso|slowly|slow|moderately|moderate|fast)\b"
)
TEMPO_MARK_ALIASES = {"vivo": "vivace", "slowly": "slow", "moderate": "moderately"}
NUMBER = re.compile(r"\d+(\.\d+)?")


def _words_directions(text: str) -> list[str]:
    """演奏指示の文字 (Allegro / rit. / a tempo / cresc. など) から、速度標語と速度・強弱の変化の指示を読む"""
    text = text.lower()
    names = [name for name, pattern in WORD_PATTERNS if pattern.search(text)]
    mark = TEMPO_MARK_PATTERN.search(text)
    if mark is not None:
        names.append(f"mark_{TEMPO_MARK_ALIASES.get(mark.group(1), mark.group(1))}")
    return names


def _metronome_mark(metronome: ET.Element) -> str | None:
    """<metronome> (♩ = 120 など) を metronome_{基準の音符}_{数値} にする。音符の等式や読めない数値は None"""
    units = metronome.findall("beat-unit")
    number = NUMBER.search(metronome.findtext("per-minute") or "")
    if len(units) != 1 or number is None or float(number.group()) <= 0:
        return None
    unit = (units[0].text or "").strip() + "." * len(metronome.findall("beat-unit-dot"))
    if unit not in METRONOME_UNITS:
        return None
    return f"metronome_{unit}_{metronome_bpm(float(number.group()))}"


def _one_metronome(directions: list[tuple[Fraction, str]]) -> list[tuple[Fraction, str]]:
    """同じ位置のメトロノーム記号は最後の 1 つだけ残す"""
    last = {position: name for position, name in directions if name.startswith("metronome")}
    return [(p, name) for p, name in directions if not name.startswith("metronome") or last[p] == name]


DOUBLE_BAR_STYLES = ("light-light", "light-heavy", "heavy-light", "heavy-heavy")


def _ending_numbers(text: str) -> frozenset[int]:
    numbers: set[int] = set()
    for a, b in re.findall(r"(\d+)\s*(?:-\s*(\d+))?", text or ""):
        numbers.update(range(int(a), int(b or a) + 1))
    return frozenset(numbers)


def _tolerance(divisions: int) -> Fraction:
    """丸めの誤差として許すずれ。divisions が小さい楽譜では丸めが起きないので、大きくても 1/32 拍にする"""
    return min(Fraction(3, divisions), Fraction(1, 32))


# 拍の中の位置として書ける分数の分母 (トークナイザーの既定)。これで表せる位置は丸められていないとみなす
EXACT_DENOMINATORS = (1, *ScoreTokenizerConfig().fraction_denominators)


def _snap(value: Fraction, divisions: int) -> Fraction:
    """divisions で丸められた位置 (7 連符の 23/160 など) を、丸めの誤差の範囲で書ける分数 (1/7) に寄せる。

    書ける分数ならそのまま返す。寄せる先は divisions で割り切れない分母 (割り切れるならそもそも丸められない) の中で、
    分母のいちばん小さいもの。
    """
    if (value % 1).denominator in EXACT_DENOMINATORS:
        return value
    tolerance = _tolerance(divisions)
    for denominator in EXACT_DENOMINATORS:
        if divisions % denominator == 0:
            continue
        candidate = Fraction(round(value * denominator), denominator)
        if abs(candidate - value) <= tolerance:
            return candidate
    return value


def _duration(note: ET.Element, grace: ET.Element | None) -> Duration | None:
    """音符の種類・付点・連符・装飾音から音価を作る。種類が書かれていない (小節全体の休符など) なら None"""
    note_type = note.findtext("type")
    if note_type not in NOTE_TYPES:
        return None
    modification = note.find("time-modification")
    tuplet = None
    if modification is not None:
        tuplet = (int(modification.findtext("actual-notes")), int(modification.findtext("normal-notes")))
    grace_kind = None if grace is None else ("slash" if grace.get("slash") == "yes" else "grace")
    return Duration(note_type, len(note.findall("dot")), tuplet, grace_kind)


def _parse_part(
    part: ET.Element, part_index: int, staff_offset: int, raw_measures: list[_RawMeasure]
) -> list[_RawGroup]:
    groups: list[_RawGroup] = []
    divisions = 1
    order = 0
    for mi, measure in enumerate(part.findall("measure")):
        raw = raw_measures[mi]
        cursor = Fraction(0)
        max_position = Fraction(0)
        last_onset = Fraction(0)
        last_group: _RawGroup | None = None
        voice_end: dict[str, Fraction] = {}  # 声部ごとの、音価どおりに積み上げた次の音の位置
        for el in measure:
            tag = el.tag
            if tag == "attributes":
                if el.findtext("divisions"):
                    divisions = int(float(el.findtext("divisions")))
                key = el.find("key")
                if key is not None and key.findtext("fifths") is not None:
                    fifths = int(key.findtext("fifths"))
                    if cursor > 0 and fifths != raw.key:
                        raise ScoreError("小節の途中で調号が変わる")
                    raw.key = fifths
                time = el.find("time")
                if time is not None:
                    if time.find("senza-misura") is not None:
                        raise ScoreError("拍子なし")
                    try:
                        signature = (int(time.findtext("beats")), int(time.findtext("beat-type")))
                    except (TypeError, ValueError):
                        raise ScoreError(f"対応していない拍子 {time.findtext('beats')}/{time.findtext('beat-type')}")
                    if cursor > 0 and signature != raw.time_signature:
                        raise ScoreError("小節の途中で拍子が変わる")
                    raw.time_signature = signature
                for clef in el.findall("clef"):
                    staff = staff_offset + int(clef.get("number", "1"))
                    raw.events.append((cursor, "clef", (staff, _clef_name(clef))))
                if el.find("transpose") is not None:
                    raise ScoreError("移調楽器")
            elif tag == "backup":
                cursor -= Fraction(int(el.findtext("duration")), divisions)
            elif tag == "forward":
                cursor += Fraction(int(el.findtext("duration")), divisions)
                max_position = max(max_position, cursor)
            elif tag == "direction":
                position = _snap(cursor + Fraction(int(el.findtext("offset") or 0), divisions), divisions)
                for direction_type in el.findall("direction-type"):
                    for child in direction_type:
                        if child.tag == "dynamics":
                            for mark in child:
                                if mark.tag in DYNAMICS:
                                    raw.events.append((position, "direction", f"dyn_{mark.tag}"))
                        elif child.tag == "wedge" and child.get("type") in ("crescendo", "diminuendo", "stop"):
                            raw.events.append((position, "direction", f"wedge_{child.get('type')}"))
                        elif child.tag == "octave-shift":
                            staff = staff_offset + int(el.findtext("staff") or 1)
                            raw.events.append((position, "ottava", (staff, _ottava_name(child))))
                        elif child.tag in ("words", "metronome"):
                            if child.tag == "words":
                                swing = _swing_from_words(child.text or "")
                                for name in _words_directions(child.text or ""):
                                    raw.events.append((position, "direction", name))
                            else:
                                swing = _swing_from_metronome(child)
                                mark = _metronome_mark(child) if swing is None else None
                                if mark is not None:
                                    raw.events.append((position, "direction", mark))
                            if swing is not None:
                                raw.events.append((position, "swing", swing))
                        elif child.tag == "pedal":
                            kind = child.get("type")
                            if kind in ("stop", "change"):
                                raw.events.append((position, "direction", "pedal_stop"))
                            if kind in ("start", "change"):
                                raw.events.append((position, "direction", "pedal_start"))
                sound = el.find("sound")
                if sound is not None and sound.get("tempo"):
                    raw.events.append((position, "tempo", float(sound.get("tempo"))))
                if sound is not None and sound.find("swing") is not None:
                    raw.events.append((position, "swing", _swing_from_sound(sound.find("swing"))))
            elif tag == "sound" and el.get("tempo"):
                raw.events.append((cursor, "tempo", float(el.get("tempo"))))
            elif tag == "barline" and part_index == 0:
                repeat = el.find("repeat")
                # 反復記号のない二重線 (終止線の形も含む) は区切りの複縦線。小節の頭に書いたものは前の小節の終わり
                if repeat is None and el.findtext("bar-style") in DOUBLE_BAR_STYLES:
                    location = el.get("location", "right")
                    if location == "right":
                        raw.double_bar = True
                    elif location == "left" and mi > 0:
                        raw_measures[mi - 1].double_bar = True
                if repeat is not None:
                    if repeat.get("direction") == "forward":
                        raw.forward_repeat = True
                    else:
                        raw.backward_repeat = int(repeat.get("times") or 2)
                ending = el.find("ending")
                if ending is not None:
                    if ending.get("type") == "start":
                        raw.ending_start = _ending_numbers(ending.get("number"))
                    else:
                        raw.ending_stop = True
            elif tag == "note":
                is_chord = el.find("chord") is not None
                grace = el.find("grace")
                is_rest = el.find("rest") is not None
                length = Fraction(int(el.findtext("duration") or 0), divisions)
                voice = el.findtext("voice") or "1"
                duration = _duration(el, grace)
                # 7 連符などは divisions で割り切れず、duration が丸められている。数 tick のずれなら
                # 声部ごとに音価どおりの長さで位置を積み上げる (backup / forward はファイルの値のまま)
                tolerance = _tolerance(divisions)
                if is_chord:
                    onset = last_onset
                else:
                    expected = voice_end.get(voice)
                    if expected is not None and abs(cursor - expected) <= tolerance:
                        onset = expected
                    else:
                        onset = _snap(cursor, divisions)
                    if grace is None:
                        cursor += length
                        max_position = max(max_position, cursor)
                        nominal = duration.quarters if duration is not None else length
                        if abs(nominal - length) > tolerance:
                            if not is_rest:
                                raise ScoreError("duration が音符の種類と合わない")
                            nominal = length
                        voice_end[voice] = onset + nominal
                last_onset = onset
                if is_rest:
                    last_group = None
                    continue
                if el.find("pitch") is None:
                    raise ScoreError("音高のない音符 (打楽器)")
                step = el.findtext("pitch/step")
                alter_text = el.findtext("pitch/alter") or "0"
                if float(alter_text) != int(float(alter_text)):
                    raise ScoreError("微分音")
                alter = int(float(alter_text))
                pitch = (int(el.findtext("pitch/octave")) + 1) * 12 + STEP_PITCH_CLASS[step] + alter
                if abs(alter) > 2:
                    raise ScoreError("変化記号が大きすぎる")
                if duration is None:
                    raise ScoreError(f"対応していない音符の種類 {el.findtext('type')}")
                articulations = set()
                notations = el.find("notations")
                if notations is not None:
                    for child in notations.iter():
                        if child.tag in ARTICULATIONS:
                            articulations.add(child.tag)
                        elif child.tag == "tremolo" and child.get("type", "single") in ("single", "start"):
                            # 2 音間のトレモロは始まりの音にだけ付ける (終わりは次の 2:1 の音)
                            articulations.add(f"tremolo{min(max(int(child.text or 3), 1), 4)}")
                slurs = set()
                if notations is not None:
                    slurs = {
                        (s.get("type"), s.get("number", "1"))
                        for s in notations.findall("slur")
                        if s.get("type") in ("start", "stop")
                    }
                tie = any(t.get("type") == "start" for t in el.findall("tie"))
                glissando = notations is not None and any(
                    child.get("type") == "start" for child in notations if child.tag in ("glissando", "slide")
                )
                note = Note(pitch, alter, tie, glissando)
                display_staff = staff_offset + int(el.findtext("staff") or 1)
                if is_chord and last_group is not None:
                    if last_group.duration != duration:
                        raise ScoreError("和音の中で音価が違う")
                    last_group.notes.append(note)
                    last_group.display_staves.append(display_staff)
                    last_group.articulations |= articulations
                    last_group.slurs |= slurs  # 和音の各音に同じスラーが書かれていても番号で 1 本にまとまる
                else:
                    last_group = _RawGroup(
                        mi, onset, (part_index, el.findtext("voice") or "1"), duration, [note], [display_staff],
                        articulations, order, slurs,
                    )  # fmt: skip
                    groups.append(last_group)
                    order += 1
        # 長さも丸めの誤差を含むので、音価どおりに積み上げた終わりと比べて寄せる (拍子どおりの長さに近ければそれにする)
        tolerance = _tolerance(divisions)
        length = _snap(max([max_position, *voice_end.values()]), divisions)
        if raw.time_signature is not None:
            nominal = Fraction(4 * raw.time_signature[0], raw.time_signature[1])
            if abs(length - nominal) <= tolerance:
                length = nominal
        raw.length = max(raw.length, length)
    return groups


def _unfold(raw_measures: list[_RawMeasure]) -> list[int]:
    """反復記号とカッコ (1 番・2 番) を展開した演奏順の小節番号"""
    order: list[int] = []
    repeat_start = 0
    passes: dict[int, int] = {}
    current_pass = 1
    i = 0
    while i < len(raw_measures):
        if len(order) > 4 * len(raw_measures):
            raise ScoreError("反復記号を展開できない")
        raw = raw_measures[i]
        if raw.ending_start is not None and current_pass not in raw.ending_start:
            # このカッコを飛ばす (終わりの小節の次へ)
            j = i
            while j < len(raw_measures) - 1 and not raw_measures[j].ending_stop and not raw_measures[j].backward_repeat:
                j += 1
            i = j + 1
            continue
        if raw.forward_repeat:
            repeat_start = i
        order.append(i)
        if raw.backward_repeat:
            count = passes.get(i, 1)
            if count < raw.backward_repeat:
                passes[i] = count + 1
                current_pass = count + 1
                i = repeat_start
                continue
            passes[i] = 1
            current_pass = 1
            repeat_start = i + 1
        elif raw.ending_stop:
            current_pass = 1
        i += 1
    return order


def _clean_group(group: _RawGroup, length: Fraction) -> _RawGroup | None:
    """同じ音の重複と 88 鍵の外の音、小節の最後の後打ちの装飾音を捨てる。音が残らなければ None"""
    if group.duration.grace and 0 < length <= group.onset:
        return None
    kept: dict[int, tuple[Note, int]] = {}
    for note, staff in zip(group.notes, group.display_staves):
        if note.pitch not in kept and 21 <= note.pitch <= 108:
            kept[note.pitch] = (note, staff)
    if not kept:
        return None
    group.notes = [note for note, _ in kept.values()]
    group.display_staves = [staff for _, staff in kept.values()]
    return group


def parse_xml(path: str | Path) -> ET.Element:
    try:
        return ET.parse(path).getroot()
    except ET.ParseError as e:
        raise ScoreError(f"XML として読めない: {e}")


def score_title(root: ET.Element) -> str:
    """曲名 (work-title / movement-title / いちばん大きいクレジット文字)。なければ空"""
    for path in ("work/work-title", "movement-title"):
        text = (root.findtext(path) or "").strip()
        if text:
            return text
    credits = [
        (float(w.get("font-size") or 0), (w.text or "").strip())
        for w in root.iter("credit-words")
        if (w.text or "").strip()
    ]
    return max(credits)[1] if credits else ""


def read_musicxml(source: str | Path | ET.Element, unfold: bool = True) -> list[Measure]:
    """MusicXML (パスか読み込み済みの要素) を読む。unfold=True なら反復記号を展開した演奏順の小節にする"""
    root = source if isinstance(source, ET.Element) else parse_xml(source)
    if root.tag != "score-partwise":
        raise ScoreError(f"{root.tag} には対応していない")
    parts = root.findall("part")
    staves = [int(p.findtext("measure/attributes/staves") or 1) for p in parts]
    if staves == [2]:
        offsets = [0]
    elif staves == [1, 1]:
        offsets = [0, 1]
    else:
        raise ScoreError(f"段の構成 {staves}")
    counts = {len(p.findall("measure")) for p in parts}
    if len(counts) != 1:
        raise ScoreError("パートごとに小節数が違う")
    raw_measures = [_RawMeasure() for _ in range(counts.pop())]
    if not raw_measures:
        raise ScoreError("小節がない")
    raw_groups: list[_RawGroup] = []
    for index, (part, offset) in enumerate(zip(parts, offsets)):
        raw_groups += _parse_part(part, index, offset, raw_measures)
    raw_groups = [g for g in map(lambda g: _clean_group(g, raw_measures[g.measure].length), raw_groups) if g]

    # 声部: MuseScore の決まり (1〜4 は上段、5〜8 は下段) で本来の段を決め、段の中で 1 から振り直す。
    # 別のソフトの書き出しで 9 割以上がもう一方の段に書かれている声部だけ、その段を本来の段にする
    # (多数決にすると、反復を展開して書き出したときにほぼ半々の声部の段が入れ替わる)
    staff_counts: dict[tuple[int, str], Counter] = defaultdict(Counter)
    for g in raw_groups:
        staff_counts[g.raw_voice][g.display_staves[0]] += 1

    def preferred_staff(voice: tuple[int, str]) -> int:
        if len(parts) == 2:
            return voice[0] + 1
        return 2 if voice[1].isdigit() and int(voice[1]) > 4 else 1

    def home_staff(voice: tuple[int, str], counts: Counter) -> int:
        other = 3 - preferred_staff(voice)
        if len(parts) == 1 and counts[other] >= 0.9 * sum(counts.values()):
            return other
        return preferred_staff(voice)

    home = {voice: home_staff(voice, counts) for voice, counts in staff_counts.items()}
    # 段を移したせいで声部が 5 つ以上になる段 (上段の声部が全部下段に書かれているなど) は、移した声部を戻す
    for staff in (1, 2):
        if sum(h == staff for h in home.values()) > 4:
            home.update({v: 3 - staff for v, h in home.items() if h == staff and preferred_staff(v) != staff})
    voice_index: dict[tuple[int, str], int] = {}
    for staff in (1, 2):
        voices = sorted(
            (v for v in home if home[v] == staff), key=lambda v: (v[0], int(v[1]) if v[1].isdigit() else 99)
        )
        if len(voices) > 4:
            raise ScoreError("1 段に声部が 5 つ以上")
        for i, voice in enumerate(voices, 1):
            voice_index[voice] = i

    measures: list[Measure] = []
    time_signature, key = None, 0
    clefs = list(DEFAULT_CLEFS)
    ottavas = ["none", "none"]
    swing = "none"
    carried: list[tuple[Fraction, str, object]] = []  # 小節の終わりに書かれた指示は次の小節の頭へ
    for raw in raw_measures:
        if raw.time_signature is not None:
            time_signature = raw.time_signature
        if time_signature is None:
            raise ScoreError("拍子がない")
        if raw.key is not None:
            key = raw.key
        measure = Measure(time_signature, key, tuple(clefs), raw.length)
        measure.double_bar = raw.double_bar
        if measure.length == 0:
            measure.length = measure.nominal_length
        events = [(Fraction(0), kind, value) for _, kind, value in carried] + raw.events
        carried = []
        header_clefs = list(clefs)
        header_ottavas = list(ottavas)
        header_swing = swing
        # 小節の途中のオクターブ記号 (段ごと) とスイング (段 0 として扱う) の変化。同じ位置は最後の状態
        changes: dict[tuple[Fraction, int], str] = {}
        for position, kind, value in sorted(events, key=lambda e: e[0]):
            if position >= measure.length:
                carried.append((position, kind, value))
            elif kind == "clef":
                staff, name = value
                if position == 0:
                    header_clefs[staff - 1] = name
                elif name != clefs[staff - 1]:
                    measure.directions.append((position, f"clef{staff}_{name}"))
                clefs[staff - 1] = name
            elif kind == "ottava":
                staff, name = value
                if position == 0:
                    header_ottavas[staff - 1] = name
                else:
                    changes[(position, staff)] = name
                ottavas[staff - 1] = name
            elif kind == "swing":
                if position == 0:
                    header_swing = value
                else:
                    changes[(position, 0)] = value
                swing = value
            elif kind == "direction":
                measure.directions.append((position, value))
            elif kind == "tempo":
                measure.tempos.append((position, value))
        measure.clefs = tuple(header_clefs)
        measure.ottavas = tuple(header_ottavas)
        measure.swing = header_swing
        state = [header_swing, *header_ottavas]
        for (position, staff), name in sorted(changes.items()):
            if name != state[staff]:
                measure.directions.append((position, f"ottava{staff}_{name}" if staff else f"swing_{name}"))
                state[staff] = name
        measure.directions = _one_metronome(sorted(set(measure.directions)))
        measures.append(measure)

    for g in raw_groups:
        staff = home[g.raw_voice]
        group = Group(
            g.onset, staff, voice_index[g.raw_voice], g.duration, sorted(g.notes, key=lambda n: n.pitch),
            tuple(a for a in ARTICULATIONS if a in g.articulations), g.display_staves[0] != staff,
            sum(kind == "stop" for kind, _ in g.slurs), sum(kind == "start" for kind, _ in g.slurs),
        )  # fmt: skip
        measures[g.measure].groups.append(group)

    for measure in measures:
        _check_voices(measure)
    if unfold:
        # 2 回目以降に出てくる小節は別のものにする (スラーの後始末で塊ごとに数を変えるため)
        seen: set[int] = set()
        unfolded = []
        for i in _unfold(raw_measures):
            unfolded.append(copy.deepcopy(measures[i]) if i in seen else measures[i])
            seen.add(i)
        measures = unfolded
    _drop_unpaired_slurs(measures)
    drop_tempo_ramps(measures)
    if measures:
        measures[-1].double_bar = False  # 最後は書き出しで終止線にする
    return measures


# 直前のメトロノーム記号からこの長さ (4 分音符単位) 以内で、テンポの変化がこの比 (対数) 未満のものは再生用の刻みとみなす
TEMPO_RAMP_QUARTERS = 8
TEMPO_RAMP_RATIO = 0.12


def drop_tempo_ramps(measures: list[Measure]) -> None:
    """再生用に細かく刻んだメトロノーム記号 (rit. や accel. でテンポを少しずつ変えて鳴らすためのもの) を消す。

    ♩=80 rit. ♩=76 ♩=72 ... と拍ごとに置かれた刻みは楽譜としては読みにくいので、最初の記号だけ残す
    (テンポの変化そのものは <sound> のテンポとして MTIME に残る)。消すと新しく刻みに見える組ができることがあるので、
    変わらなくなるまで繰り返す (書き出して読み直しても同じ結果になる)。
    """
    while True:
        marks = []  # (曲の頭からの位置, 小節番号, 小節の中の位置, 名前)
        offset = Fraction(0)
        for i, measure in enumerate(measures):
            marks += [(offset + p, i, p, name) for p, name in measure.directions if name.startswith("metronome")]
            offset += measure.length
        drop = set()
        for (at, _, _, before), (now, i, position, name) in pairwise(marks):
            change = abs(math.log(_metronome_qpm(name) / _metronome_qpm(before)))
            if now - at < TEMPO_RAMP_QUARTERS and change < TEMPO_RAMP_RATIO:
                drop.add((i, position, name))
        if not drop:
            return
        for i, position, name in drop:
            measures[i].directions = [d for d in measures[i].directions if d != (position, name)]


def _metronome_qpm(name: str) -> float:
    """メトロノーム記号の名前 (metronome_quarter._60 など) を 4 分音符単位のテンポにする"""
    unit, bpm = name.split("_", 1)[1].rsplit("_", 1)
    return float(int(bpm) * metronome_quarters(unit))


def _drop_unpaired_slurs(measures: list[Measure]) -> None:
    """相手のないスラーの始まりと終わりを消す (反復の展開で 1 番カッコに入るスラーが切れた場合など)"""
    counts: Counter = Counter()
    for _, g, kind, _ in pair_slurs(measures):
        counts[(id(g), kind)] += 1
    for measure in measures:
        for g in measure.groups:
            g.slur_start, g.slur_stop = counts[(id(g), "start")], counts[(id(g), "stop")]


def _check_voices(measure: Measure) -> None:
    """声部ごとに音が重ならず、小節からはみ出さないことを確かめ、塊を位置・段・声部の順に並べる"""
    measure.groups.sort(key=group_order)
    busy: dict[tuple[int, int], Fraction] = {}
    for g in measure.groups:
        voice = (g.staff, g.voice)
        if g.onset < busy.get(voice, 0):
            raise ScoreError("声部の中で音が重なる")
        if g.end > measure.length or (g.duration.grace and g.onset >= measure.length):
            raise ScoreError("小節からはみ出す音")
        busy[voice] = g.end


# ----------------------------------------------------------------------
# 書き出し
# ----------------------------------------------------------------------
@dataclass
class _Item:
    """声部の中の 1 つの音符 (塊) または休符"""

    onset: Fraction
    duration: Duration
    group: Group | None = None  # None なら休符
    whole_rest: Fraction | None = None  # 小節全体の休符 (長さ)
    forward: Fraction | None = None  # 休符で書けない隙間 (長さ)
    tremolo: tuple[str, int] | None = None  # 2 音間のトレモロ (start / stop, 斜線の数)
    hidden: bool = False  # 表示しない休符 (第 2 声部以降の最初の音の前と最後の音の後)
    beams: dict[int, str] = field(default_factory=dict)
    tuplet: str | None = None  # start / stop

    @property
    def quarters(self) -> Fraction:
        if self.whole_rest is not None:
            return self.whole_rest
        return self.forward if self.forward is not None else self.duration.quarters


def _is_compound(time_signature: tuple[int, int]) -> bool:
    beats, beat_type = time_signature
    return beat_type >= 8 and beats % 3 == 0 and beats > 3


REST_TUPLETS = ((3, 2), (5, 4), (7, 4), (9, 8))


def _rest_candidates(time_signature: tuple[int, int]) -> tuple[list[Duration], list[Duration]]:
    plain = [Duration(t) for t in ("whole", "half", "quarter", "eighth", "16th", "32nd", "64th", "128th")]
    if _is_compound(time_signature):
        plain += [Duration("quarter", 1), Duration("half", 1)]
    plain.sort(key=lambda d: -d.quarters)
    # 休符に使う連符はよくある比だけにする (珍しい比を選ぶと残りを分けられなくなる)
    tuplets = sorted((Duration(t, 0, r) for r in REST_TUPLETS for t in NOTE_TYPES), key=lambda d: -d.quarters)
    return plain, tuplets


def _is_dyadic(value: Fraction) -> bool:
    return value.denominator & (value.denominator - 1) == 0


def _writable(value: Fraction) -> bool:
    """休符を並べて書ける長さか (分母が 2^k に 3, 5, 7, 9 のどれかを掛けた数)"""
    denominator = value.denominator
    for factor in (1, 3, 5, 7, 9):
        if denominator % factor == 0 and _is_dyadic(Fraction(1, denominator // factor)):
            return True
    return False


def _rests(start: Fraction, end: Fraction, time_signature: tuple[int, int]) -> list[_Item]:
    """start から end までの休符を、拍の区切りに合う大きい順に並べる。

    連符が絡む (位置が 2 の累乗の分数でない) 空きは 4 分音符の拍ごとに区切ってから、連符の休符を優先して分ける。
    """
    if not (_is_dyadic(start) and _is_dyadic(end)):
        cuts = [start, *range(math.floor(start) + 1, math.ceil(end)), end]
        return [
            item for a, b in zip(cuts, cuts[1:]) for item in _rest_segment(Fraction(a), Fraction(b), time_signature)
        ]
    return _rest_segment(start, end, time_signature)


def _rest_segment(start: Fraction, end: Fraction, time_signature: tuple[int, int]) -> list[_Item]:
    items = []
    plain, tuplets = _rest_candidates(time_signature)
    position = start
    while position < end:
        dyadic = _is_dyadic(position) and _is_dyadic(end - position)
        grid = math.lcm(position.denominator, end.denominator)
        for duration in plain + tuplets if dyadic else tuplets + plain:
            d = duration.quarters
            rest = end - position - d
            if rest >= 0 and position % d == 0 and grid % d.denominator == 0 and (rest == 0 or _writable(rest)):
                break
        else:
            # 書ける休符にならない隙間 (元の楽譜の誤差や生成の誤り) は見えない forward で埋める
            items.append(_Item(position, Duration("128th"), forward=end - position))
            break
        items.append(_Item(position, duration))
        position += d
    return items


def _beat_span(time_signature: tuple[int, int]) -> Fraction:
    """連桁でまとめる範囲"""
    beat_type = time_signature[1]
    if _is_compound(time_signature):
        return Fraction(3, 2)
    if beat_type <= 2:
        return Fraction(2)
    return Fraction(1)


def _beam_levels(duration: Duration) -> int:
    return {"eighth": 1, "16th": 2, "32nd": 3, "64th": 4, "128th": 5}.get(duration.type, 0)


def _set_beams(items: list[_Item], time_signature: tuple[int, int]) -> None:
    span = _beat_span(time_signature)
    runs: list[list[_Item]] = []
    run: list[_Item] = []
    for item in items:
        beamable = (
            item.group is not None
            and item.duration.grace is None
            and item.duration.tuplet != TREMOLO_PAIR
            and _beam_levels(item.duration) > 0
        )
        window = item.onset // span
        same_window = beamable and (item.onset + item.quarters) <= (window + 1) * span
        if run and (not same_window or run[-1].onset // span != window):
            runs.append(run)
            run = []
        if same_window:
            run.append(item)
    runs.append(run)
    for run in runs:
        if len(run) < 2:
            continue
        levels = [_beam_levels(item.duration) for item in run]
        for i, item in enumerate(run):
            for level in range(1, levels[i] + 1):
                before = i > 0 and levels[i - 1] >= level
                after = i < len(run) - 1 and levels[i + 1] >= level
                if level == 1 or before or after:
                    item.beams[level] = "continue" if before and after else "end" if before else "begin"
                else:
                    item.beams[level] = "backward hook" if i > 0 else "forward hook"


def _set_tuplets(items: list[_Item]) -> None:
    """同じ比の連符が続く間を 1 つの括弧にする。括弧の長さは normal-notes × 中でいちばん短い音符の種類"""
    open_item: _Item | None = None
    ratio = None
    total = Fraction(0)
    shortest = Fraction(0)
    previous: _Item | None = None
    for item in items:
        if item.duration.grace or item.forward is not None:
            continue
        tuplet = item.duration.tuplet if item.whole_rest is None else None
        if tuplet == TREMOLO_PAIR:
            tuplet = None
        if open_item is not None and tuplet != ratio:
            previous.tuplet = "stop" if previous is not open_item else None
            open_item = None
        if tuplet:
            if open_item is None:
                open_item, ratio, total, shortest = item, tuplet, Fraction(0), NOTE_TYPES[item.duration.type]
                item.tuplet = "start"
            total += item.quarters
            shortest = min(shortest, NOTE_TYPES[item.duration.type])
            if total >= tuplet[1] * shortest:
                item.tuplet = "stop" if item is not open_item else None
                open_item = None
        previous = item
    if open_item is not None and previous is not open_item:
        previous.tuplet = "stop"
    elif open_item is not None:
        open_item.tuplet = None


def _set_tremolos(items: list[_Item]) -> None:
    """連符比 2:1 の音を 2 音間のトレモロにする。tremoloN の付いた音 (斜線 N 本) が組の始まりで、それ以外は終わり"""
    start: _Item | None = None
    for item in items:
        if item.group is None or item.duration.tuplet != TREMOLO_PAIR:
            continue
        count = next((int(a[-1]) for a in item.group.articulations if a in TREMOLOS), None)
        if count is not None:
            item.tremolo = ("start", count)
            start = item
        else:  # 始まりが前の小節にある組もある
            item.tremolo = ("stop", start.tremolo[1] if start else 3)
            start = None


def _voice_items(measure: Measure) -> dict[tuple[int, int], list[_Item]]:
    """段・声部ごとに、塊の間を休符で埋めた列を作る。音のない段は声部 1 に小節全体の休符を置く"""
    by_voice: dict[tuple[int, int], list[Group]] = defaultdict(list)
    for g in sorted(measure.groups, key=group_order):
        by_voice[(g.staff, g.voice)].append(g)
    result: dict[tuple[int, int], list[_Item]] = {}
    for staff in (1, 2):
        if not any(s == staff for s, _ in by_voice):
            result[(staff, 1)] = [_Item(Fraction(0), Duration("whole"), whole_rest=measure.length)]
    for voice, groups in by_voice.items():
        # 第 2 声部以降は、最初の音の前と最後の音の後の休符を表示しない (段の間が休符で埋まって読みにくくなる)
        hide = voice[1] > 1
        items: list[_Item] = []
        position = Fraction(0)
        for g in groups:
            if g.onset > position:
                rests = _rests(position, g.onset, measure.time_signature)
                for rest in rests:
                    rest.hidden = hide and not items
                items += rests
            items.append(_Item(g.onset, g.duration, g))
            position = max(position, g.end)
        if position < measure.length:
            rests = _rests(position, measure.length, measure.time_signature)
            for rest in rests:
                rest.hidden = hide
            items += rests
        _set_beams(items, measure.time_signature)
        _set_tuplets(items)
        _set_tremolos(items)
        result[voice] = items
    return dict(sorted(result.items()))


def _tie_stops(measures: list[Measure]) -> set[tuple[int, int, int]]:
    """タイで前の音から続いている音 (小節番号, id(塊), pitch)"""
    stops = set()
    for mi, measure in enumerate(measures):
        for g in measure.groups:
            for note in g.notes:
                if not note.tie:
                    continue
                if g.end < measure.length:
                    candidates = [(mi, h) for h in measure.groups if h.onset == g.end]
                elif mi + 1 < len(measures):
                    candidates = [(mi + 1, h) for h in measures[mi + 1].groups if h.onset == 0]
                else:
                    continue
                # 同じ声部を優先し、なければ同じ段の別の声部の同じ音につなぐ
                candidates.sort(key=lambda c: (c[1].staff, c[1].voice) != (g.staff, g.voice))
                for target_measure, h in candidates:
                    if h.staff == g.staff and h.duration.grace is None and any(n.pitch == note.pitch for n in h.notes):
                        stops.add((target_measure, id(h), note.pitch))
                        break
    return stops


def _glissando_stops(measures: list[Measure]) -> dict[tuple[int, int, int], str]:
    """グリッサンドの行き先の音 (小節番号, id(塊), pitch) -> 線の番号。行き先は同じ声部の次の塊の、いちばん近い音"""
    stops = {}
    for mi, measure in enumerate(measures):
        for g in measure.groups:
            starts = [(i, n) for i, n in enumerate(g.notes) if n.glissando]
            if not starts:
                continue
            same_voice = lambda h: (h.staff, h.voice) == (g.staff, g.voice)  # noqa: E731
            later = [(mi, h) for h in measure.groups if same_voice(h) and h.onset >= g.end and h is not g]
            if not later and mi + 1 < len(measures):
                later = [(mi + 1, h) for h in measures[mi + 1].groups if same_voice(h)]
            if not later:
                continue
            target_measure, target = min(later, key=lambda c: (c[0], c[1].onset))
            for i, note in starts:
                nearest = min(target.notes, key=lambda n: abs(n.pitch - note.pitch))
                stops[(target_measure, id(target), nearest.pitch)] = str(i + 1)
    return stops


def _slur_marks(measures: list[Measure]) -> dict[int, list[tuple[str, int]]]:
    """塊ごとに書くスラー id(塊) -> [(start / stop, MusicXML の番号)]。同時に開いているスラーには別の番号を振る"""
    marks: dict[int, list[tuple[str, int]]] = defaultdict(list)
    numbers: dict[int, int] = {}
    for _, g, kind, slur_id in pair_slurs(measures):
        if kind == "start":
            used = set(numbers.values())
            numbers[slur_id] = next(n for n in range(1, 17) if n not in used)
            marks[id(g)].append(("start", numbers[slur_id]))
        else:
            marks[id(g)].append(("stop", numbers.pop(slur_id)))
    return marks


def _divisions(measures: list[Measure], items: list[dict[tuple[int, int], list[_Item]]]) -> int:
    denominators = {1}
    for measure, voices in zip(measures, items):
        denominators.add(measure.length.denominator)
        denominators.update(position.denominator for position, _ in measure.directions)
        denominators.update(position.denominator for position, _ in measure.tempos)
        for voice_items in voices.values():
            for item in voice_items:
                denominators.add(item.onset.denominator)
                denominators.add(item.quarters.denominator)
    return math.lcm(*denominators)


def _sub(parent: ET.Element, tag: str, text: object = None, **attributes: str) -> ET.Element:
    element = ET.SubElement(parent, tag, attributes)
    if text is not None:
        element.text = str(text)
    return element


def _clef_element(parent: ET.Element, staff: int, name: str) -> None:
    clef = _sub(parent, "clef", number=str(staff))
    _sub(clef, "sign", name[0])
    _sub(clef, "line", name[1])
    if len(name) > 2:
        _sub(clef, "clef-octave-change", int(name[2:]))


ACCIDENTAL_NAMES = {-2: "flat-flat", -1: "flat", 0: "natural", 1: "sharp", 2: "double-sharp"}


def _accidentals(measure: Measure, tie_stops: set, measure_index: int) -> dict[tuple[int, int], bool]:
    """臨時記号を付ける音 (id(塊), pitch)。段ごとに、調号とその小節で前に出た同じ幹音・オクターブの変化を見る"""
    shown = {}
    for staff in (1, 2):
        state: dict[tuple[str, int], int] = {}
        alters = key_alters(measure.key)
        groups = [g for g in measure.groups if g.display_staff == staff]
        for g in sorted(groups, key=group_order):
            for note in g.notes:
                step, octave = spell(note.pitch, note.alter)
                current = state.get((step, octave), alters.get(step, 0))
                continued = (measure_index, id(g), note.pitch) in tie_stops
                shown[(id(g), note.pitch)] = note.alter != current and not continued
                state[(step, octave)] = note.alter
    return shown


def write_musicxml(measures: list[Measure], path: str | Path, title: str | None = None) -> None:
    items = [_voice_items(m) for m in measures]
    divisions = _divisions(measures, items)
    tie_stops = _tie_stops(measures)
    glissando_stops = _glissando_stops(measures)
    slur_marks = _slur_marks(measures)

    def ticks(value: Fraction) -> int:
        return int(value * divisions)

    root = ET.Element("score-partwise", version="3.1")
    if title:
        _sub(_sub(root, "work"), "work-title", title)
    part_list = _sub(root, "part-list")
    score_part = _sub(part_list, "score-part", id="P1")
    _sub(score_part, "part-name", "Piano")
    part = _sub(root, "part", id="P1")

    previous: Measure | None = None
    clefs = [None, None]
    ottavas = ["none", "none"]
    swing = "none"
    for mi, (measure, voices) in enumerate(zip(measures, items)):
        element = _sub(part, "measure", number=str(mi + 1))
        if mi == 0 and measure.length != measure.nominal_length:
            element.set("implicit", "yes")
        changed_clefs = [s for s in (1, 2) if clefs[s - 1] != measure.clefs[s - 1]]
        if (
            previous is None
            or previous.key != measure.key
            or previous.time_signature != measure.time_signature
            or changed_clefs
        ):
            attributes = _sub(element, "attributes")
            if previous is None:
                _sub(attributes, "divisions", divisions)
            if previous is None or previous.key != measure.key:
                _sub(_sub(attributes, "key"), "fifths", measure.key)
            if previous is None or previous.time_signature != measure.time_signature:
                time = _sub(attributes, "time")
                _sub(time, "beats", measure.time_signature[0])
                _sub(time, "beat-type", measure.time_signature[1])
            if previous is None:
                _sub(attributes, "staves", 2)
            for staff in changed_clefs:
                _clef_element(attributes, staff, measure.clefs[staff - 1])
        clefs = list(measure.clefs)

        for position, tempo in measure.tempos:
            if position == 0:
                _sub(element, "sound", tempo=f"{tempo:g}")

        accidentals = _accidentals(measure, tie_stops, mi)
        voices_per_staff = Counter(s for (s, _), v in voices.items() if any(i.group for i in v))
        cursor = 0
        for (staff, voice), voice_items in voices.items():
            if cursor:
                _sub(_sub(element, "backup"), "duration", cursor)
            cursor = 0
            for item in voice_items:
                _write_item(
                    element, item, staff, voice, ticks, accidentals, tie_stops, glissando_stops, slur_marks, mi,
                    voices_per_staff,
                )  # fmt: skip
                cursor += ticks(item.quarters)

        # 指示 (強弱・松葉・ペダル・音部記号とオクターブ記号の変更・テンポ) は backup / forward で位置を合わせて書く。
        # オクターブ記号は状態で持っているので、次の小節の頭で状態が変わるときはこの小節の終わりで止める
        extra = [(p, name) for p, name in measure.directions] + [(p, t) for p, t in measure.tempos if p > 0]
        extra += [(Fraction(0), f"ottava{s + 1}_{name}") for s, name in enumerate(measure.ottavas)]
        extra.append((Fraction(0), f"swing_{measure.swing}"))
        if mi + 1 < len(measures):
            end_state = list(measure.ottavas)
            for _, name in sorted(measure.directions):
                if name.startswith("ottava"):
                    end_state[int(name[6]) - 1] = name.split("_", 1)[1]
            following = measures[mi + 1].ottavas
            extra += [(measure.length, f"ottava{s}_none") for s in (1, 2) if end_state[s - 1] != following[s - 1]]
        # 同じ位置ではペダルや松葉の終わりを始まりより先に書く (後に書くと始めた直後に止めたことになる)
        for position, value in sorted(extra, key=lambda e: (e[0], not str(e[1]).endswith("_stop"))):
            target = ticks(position)
            if target < cursor:
                _sub(_sub(element, "backup"), "duration", cursor - target)
            elif target > cursor:
                _sub(_sub(element, "forward"), "duration", target - cursor)
            cursor = target
            if isinstance(value, float):
                _sub(element, "sound", tempo=f"{value:g}")
            elif value.startswith("clef"):
                staff = int(value[4])
                _clef_element(_sub(element, "attributes"), staff, value.split("_", 1)[1])
                clefs[staff - 1] = value.split("_", 1)[1]
            elif value.startswith("ottava"):
                staff, name = int(value[6]), value.split("_", 1)[1]
                if name != ottavas[staff - 1]:
                    _write_ottava(element, staff, ottavas[staff - 1], name)
                    ottavas[staff - 1] = name
            elif value.startswith("swing"):
                name = value.split("_", 1)[1]
                if name != swing:
                    _write_swing(element, name)
                    swing = name
            else:
                _write_direction(element, value)
        if cursor < ticks(measure.length):
            _sub(_sub(element, "forward"), "duration", ticks(measure.length) - cursor)
        if mi == len(measures) - 1 or measure.double_bar:
            barline = _sub(element, "barline", location="right")
            _sub(barline, "bar-style", "light-heavy" if mi == len(measures) - 1 else "light-light")
        previous = measure

    ET.indent(root)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _write_swing(parent: ET.Element, name: str) -> None:
    """スイングの指定を、表示用の文字 (Swing / Swing 16ths / Straight) と再生用の <sound><swing> で書く"""
    direction = _sub(parent, "direction", placement="above")
    text = {"none": "Straight", "8th": "Swing", "16th": "Swing 16ths"}[name]
    _sub(_sub(direction, "direction-type"), "words", text, **{"font-weight": "bold"})
    swing = _sub(_sub(direction, "sound"), "swing")
    if name == "none":
        _sub(swing, "straight")
    else:
        _sub(swing, "first", 2)
        _sub(swing, "second", 1)
        _sub(swing, "swing-type", "eighth" if name == "8th" else "16th")


def _write_ottava(parent: ET.Element, staff: int, before: str, after: str) -> None:
    """オクターブ記号の状態を before から after に変える (効いているものを止めてから始める)"""
    for kind, name in (("stop", before), ("start", after)):
        if name == "none":
            continue
        size = "8" if name.startswith("8") else "15"
        high = name.endswith(("va", "ma"))
        direction = _sub(parent, "direction", placement="above" if high else "below")
        shift_type = "stop" if kind == "stop" else ("down" if high else "up")
        _sub(_sub(direction, "direction-type"), "octave-shift", type=shift_type, size=size)
        _sub(direction, "staff", staff)


def _write_direction(parent: ET.Element, name: str) -> None:
    kind, value = name.split("_", 1)
    if kind == "metronome":
        _write_metronome(parent, *value.rsplit("_", 1))
        return
    direction = _sub(parent, "direction", placement="above" if kind in ("tempo", "mark") else "below")
    direction_type = _sub(direction, "direction-type")
    if kind == "mark":
        _sub(direction_type, "words", TEMPO_MARKS[value], **{"font-weight": "bold"})
        staff = 1
    elif kind in ("tempo", "text"):
        text = TEMPO_WORDS[value] if kind == "tempo" else DYNAMIC_WORDS[value]
        _sub(direction_type, "words", text, **{"font-style": "italic"})
        staff = 1
    elif kind == "dyn":
        _sub(_sub(direction_type, "dynamics"), value)
        staff = 1
    elif kind == "wedge":
        _sub(direction_type, "wedge", type=value)
        staff = 1
    else:
        _sub(direction_type, "pedal", type=value, line="yes")
        staff = 2
    _sub(direction, "staff", staff)


def _write_metronome(parent: ET.Element, unit: str, bpm: str) -> None:
    """メトロノーム記号 (♩ = 120 など)。再生用に 4 分音符単位のテンポも <sound> に書く"""
    direction = _sub(parent, "direction", placement="above")
    metronome = _sub(_sub(direction, "direction-type"), "metronome")
    _sub(metronome, "beat-unit", unit.rstrip("."))
    if unit.endswith("."):
        _sub(metronome, "beat-unit-dot")
    _sub(metronome, "per-minute", bpm)
    _sub(direction, "staff", 1)
    _sub(direction, "sound", tempo=f"{float(int(bpm) * metronome_quarters(unit)):g}")


def _write_item(
    parent: ET.Element,
    item: _Item,
    staff: int,
    voice: int,
    ticks,
    accidentals: dict,
    tie_stops: set,
    glissando_stops: dict,
    slur_marks: dict,
    measure_index: int,
    voices_per_staff: Counter,
) -> None:
    voice_number = str((staff - 1) * 4 + voice)
    if item.forward is not None:
        forward = _sub(parent, "forward")
        _sub(forward, "duration", ticks(item.forward))
        _sub(forward, "voice", voice_number)
        _sub(forward, "staff", staff)
        return
    if item.group is None:
        note = _sub(parent, "note")
        if item.whole_rest is not None:
            _sub(note, "rest", measure="yes")
            _sub(note, "duration", ticks(item.whole_rest))
            _sub(note, "voice", voice_number)
        else:
            if item.hidden:
                note.set("print-object", "no")
            _sub(note, "rest")
            _sub(note, "duration", ticks(item.quarters))
            _sub(note, "voice", voice_number)
            _write_duration(note, item.duration)
        _sub(note, "staff", staff)
        _write_tuplet(note, item)
        return

    g = item.group
    duration = g.duration
    for i, n in enumerate(g.notes):
        note = _sub(parent, "note")
        if duration.grace:
            _sub(note, "grace", **({"slash": "yes"} if duration.grace == "slash" else {}))
        if i > 0:
            _sub(note, "chord")
        step, octave = spell(n.pitch, n.alter)
        pitch = _sub(note, "pitch")
        _sub(pitch, "step", step)
        if n.alter:
            _sub(pitch, "alter", n.alter)
        _sub(pitch, "octave", octave)
        if not duration.grace:
            _sub(note, "duration", ticks(duration.quarters))
        stop = (measure_index, id(g), n.pitch) in tie_stops
        if stop:
            _sub(note, "tie", type="stop")
        if n.tie:
            _sub(note, "tie", type="start")
        _sub(note, "voice", voice_number)
        _write_duration(note, duration, accidental=n.alter if accidentals.get((id(g), n.pitch)) else None)
        if voices_per_staff[staff] > 1:
            _sub(note, "stem", "up" if voice % 2 else "down")
        _sub(note, "staff", g.display_staff)
        if i == 0:
            for level, value in sorted(item.beams.items()):
                _sub(note, "beam", value, number=str(level))
        notations = ET.Element("notations")
        if stop:
            _sub(notations, "tied", type="stop")
        if n.tie:
            _sub(notations, "tied", type="start")
        glissando_stop = glissando_stops.get((measure_index, id(g), n.pitch))
        if glissando_stop:
            _sub(notations, "glissando", type="stop", number=glissando_stop, **{"line-type": "wavy"})
        if n.glissando:
            _sub(notations, "glissando", "gliss.", type="start", number=str(i + 1), **{"line-type": "wavy"})
        if i == 0:
            for kind, number in slur_marks.get(id(g), []):
                _sub(notations, "slur", type=kind, number=str(number))
            _write_tuplet(notations, item, standalone=False)
            marks = [a for a in g.articulations if a not in ("fermata", "arpeggiate", *ORNAMENTS, *TREMOLOS)]
            ornaments = [a for a in g.articulations if a in ORNAMENTS]
            tremolo = next((int(a[-1]) for a in g.articulations if a in TREMOLOS), None)
            if ornaments or item.tremolo or tremolo:
                ornaments_element = _sub(notations, "ornaments")
                for mark in ornaments:
                    _sub(ornaments_element, mark)
                if item.tremolo:
                    _sub(ornaments_element, "tremolo", item.tremolo[1], type=item.tremolo[0])
                elif tremolo:
                    _sub(ornaments_element, "tremolo", tremolo, type="single")
            if marks:
                articulations = _sub(notations, "articulations")
                for mark in marks:
                    _sub(articulations, mark)
            if "fermata" in g.articulations:
                _sub(notations, "fermata", type="upright")
        if "arpeggiate" in g.articulations:
            _sub(notations, "arpeggiate")
        if len(notations):
            note.append(notations)


def _write_duration(note: ET.Element, duration: Duration, accidental: int | None = None) -> None:
    _sub(note, "type", duration.type)
    for _ in range(duration.dots):
        _sub(note, "dot")
    if accidental is not None:
        _sub(note, "accidental", ACCIDENTAL_NAMES[accidental])
    if duration.tuplet:
        modification = _sub(note, "time-modification")
        _sub(modification, "actual-notes", duration.tuplet[0])
        _sub(modification, "normal-notes", duration.tuplet[1])


def _write_tuplet(parent: ET.Element, item: _Item, standalone: bool = True) -> None:
    if item.tuplet is None:
        return
    notations = _sub(parent, "notations") if standalone else parent
    _sub(notations, "tuplet", type=item.tuplet)
