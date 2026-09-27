"""楽譜の中間表現 (score.py) <-> 小節ごとのトークン列の変換。

1 小節 (= Global の 1 パッチ) のトークン列の文法:
    measure   := MTIME TS KEY CLEF1 CLEF2 OTTAVA* [SWING] [MLEN position] onset* (EOM | EOS)
    onset     := position CLEF* OTTAVA* [SWING] DIRECTION* group*  position は小節の中で昇順
    position  := BEAT [FRAC]                                BEAT = 4 分音符単位の整数部、FRAC = 拍の中の分数
    group     := SV [CROSS] DUR ART* SLUR_STOP* SLUR_START* (PITCH [TIE] [GLISS])+
                                                           同じ位置・同じ声部の和音。音高は昇順
MTIME は演奏上の小節の開始時刻。最初の小節は「最初の音から小節の頭まで何秒さかのぼるか」、それ以降は
「前の小節の (トークンから復元した) 開始時刻から何秒後か」。復元した時刻との差を取るので、ビンの丸め誤差が積もらない。
ヘッダ (拍子・調・音部記号) は毎小節入れる。オクターブ記号は段ごとの状態として、ヘッダでは効いている段だけ入れ、
小節の途中では状態が変わる位置に入れる (none で終わり)。スイングの指定も同じく曲全体の状態として入れる。
MLEN は小節の長さが拍子と違うとき (弱起など) だけ入れる。
同じ位置の中は、音部記号の変更 -> 強弱などの指示 -> 塊 (段・声部の順、同じ声部では装飾音が先) の順に並べる。
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from .config import ScoreTokenizerConfig
from .musicxml import ScoreError
from .score import (
    ARTICULATIONS,
    CLEFS,
    DYNAMICS,
    OTTAVAS,
    PEDALS,
    STEP_PITCH_CLASS,
    STEPS,
    SWINGS,
    WEDGES,
    Group,
    Measure,
    Note,
    all_durations,
    first_onset_seconds,
    group_order,
    measure_seconds,
    spell,
)

PAD, EOM, EOS, MLEN, CROSS, TIE, GLISS, SLUR_STOP, SLUR_START = range(9)
SPECIAL_NAMES = ("PAD", "EOM", "EOS", "MLEN", "CROSS", "TIE", "GLISS", "SLUR_STOP", "SLUR_START")

# 損失をトークンの種類別に見るためのグループ
TOKEN_GROUPS = (
    "end", "mtime", "header", "position", "direction", "voice", "duration", "articulation", "slur", "pitch", "tie",
)  # fmt: skip

TIME_SIGNATURE_BEATS = range(1, 25)
TIME_SIGNATURE_BEAT_TYPES = (1, 2, 4, 8, 16, 32)
DIRECTIONS = (
    tuple(f"dyn_{d}" for d in DYNAMICS) + tuple(f"wedge_{w}" for w in WEDGES) + tuple(f"pedal_{p}" for p in PEDALS)
)


class ScoreTokenizer:
    def __init__(self, config: ScoreTokenizerConfig | None = None) -> None:
        self.config = config or ScoreTokenizerConfig()
        c = self.config
        self.fractions = sorted({Fraction(n, d) for d in c.fraction_denominators for n in range(1, d)})
        self.durations = all_durations()
        self.pitches = [(p, a) for p in range(c.pitch_min, c.pitch_max + 1) for a in range(-2, 3) if _spellable(p, a)]
        self.mtime_values = np.concatenate(
            [[0.0], np.geomspace(c.mtime_min_seconds, c.mtime_max_seconds, c.num_mtime_bins - 1)]
        )

        # (種類, 値, グループ) の一覧から語彙を作る
        entries: list[tuple[str, object, str]] = [("special", name, "end") for name in SPECIAL_NAMES]
        entries += [("mtime", i, "mtime") for i in range(c.num_mtime_bins)]
        entries += [("ts", (b, t), "header") for b in TIME_SIGNATURE_BEATS for t in TIME_SIGNATURE_BEAT_TYPES]
        entries += [("key", k, "header") for k in range(-7, 8)]
        entries += [("clef", (s, name), "header") for s in (1, 2) for name in CLEFS]
        entries += [("ottava", (s, name), "direction") for s in (1, 2) for name in OTTAVAS]
        entries += [("swing", name, "direction") for name in SWINGS]
        entries += [("beat", b, "position") for b in range(c.max_measure_quarters + 1)]
        entries += [("frac", f, "position") for f in self.fractions]
        entries += [("direction", d, "direction") for d in DIRECTIONS]
        entries += [("sv", (s, v), "voice") for s in (1, 2) for v in range(1, 5)]
        entries += [("dur", d, "duration") for d in self.durations]
        entries += [("art", a, "articulation") for a in ARTICULATIONS]
        entries += [("pitch", p, "pitch") for p in self.pitches]
        self.kinds = [e[0] for e in entries]
        self.values = [e[1] for e in entries]
        self.ids = {(kind, value): i for i, (kind, value, _) in enumerate(entries)}
        self.vocab_size = len(entries)
        group = np.array([TOKEN_GROUPS.index(e[2]) for e in entries], dtype=np.int64)
        group[[MLEN]] = TOKEN_GROUPS.index("header")
        group[[CROSS]] = TOKEN_GROUPS.index("voice")
        group[[TIE]] = TOKEN_GROUPS.index("tie")
        group[[GLISS]] = TOKEN_GROUPS.index("articulation")
        group[[SLUR_STOP, SLUR_START]] = TOKEN_GROUPS.index("slur")
        group[[PAD]] = -1
        self.token_group = group
        self.group_names = TOKEN_GROUPS

    def token(self, kind: str, value: object) -> int:
        try:
            return self.ids[(kind, value)]
        except KeyError:
            raise ScoreError(f"語彙にない {kind} {value}") from None

    def describe(self, token: int) -> str:
        kind, value = self.kinds[token], self.values[token]
        return str(value) if kind == "special" else f"{kind}:{value}"

    # ------------------------------------------------------------------
    # 小節の開始時刻
    # ------------------------------------------------------------------
    def mtime_bin(self, seconds: float) -> int:
        if seconds < self.config.mtime_min_seconds / 2:
            return 0
        logs = np.log(self.mtime_values[1:])
        return int(np.argmin(np.abs(logs - np.log(seconds)))) + 1

    def nominal_starts(self, measures: list[Measure]) -> np.ndarray:
        """テンポ記号どおりに弾いたときの各小節の開始時刻 (最初の音 = 0 秒)"""
        starts = measure_seconds(measures)
        return starts[:-1] - first_onset_seconds(measures, starts)

    def mtime_bins(self, starts: np.ndarray) -> list[int]:
        """各小節の開始時刻 (最初の音 = 0 秒) を MTIME のビンにする。前の小節は丸めた後の時刻から測る"""
        bins = []
        decoded = 0.0
        for k, start in enumerate(starts):
            target = -start if k == 0 else start - decoded
            b = self.mtime_bin(max(float(target), 0.0))
            decoded = -self.mtime_values[b] if k == 0 else decoded + self.mtime_values[b]
            bins.append(b)
        return bins

    def decode_starts(self, bins: list[int]) -> np.ndarray:
        values = self.mtime_values[np.asarray(bins, dtype=np.int64)]
        if len(values):
            values[0] = -values[0]
        return np.cumsum(values)

    # ------------------------------------------------------------------
    # 移調 (学習時のデータの水増し)
    # ------------------------------------------------------------------
    def transposition(self, semitones: int, keys: set[int]) -> np.ndarray | None:
        """semitones だけ移調したときの、各トークンの移調後の番号 [vocab_size]。

        音程の綴り (長 2 度か減 3 度かなど) は、曲に出てくる調 keys が五度圏で ±7 に収まるように選ぶ。
        収まらなければ None。移調すると綴れない音や 88 鍵の外に出る音は -1 にする (その曲にあれば使わない)。
        """
        fifths = semitones * 7 % 12
        for shift in sorted({fifths, fifths - 12}, key=abs):
            if all(-7 <= k + shift <= 7 for k in keys):
                break
        else:
            return None
        table = np.arange(self.vocab_size, dtype=np.int64)
        for key in range(-7, 8):
            table[self.ids[("key", key)]] = self.ids.get(("key", key + shift), -1)
        letter_shift = shift * 4 % 7  # 五度 = 幹音 4 つ分
        for pitch, alter in self.pitches:
            step, _ = spell(pitch, alter)
            new_pitch = pitch + semitones
            new_step = STEPS[(STEPS.index(step) + letter_shift) % 7]
            new_alter = (new_pitch - STEP_PITCH_CLASS[new_step] + 6) % 12 - 6
            table[self.ids[("pitch", (pitch, alter))]] = self.ids.get(("pitch", (new_pitch, new_alter)), -1)
        return table

    # ------------------------------------------------------------------
    # 楽譜 -> トークン
    # ------------------------------------------------------------------
    def encode(self, measures: list[Measure], starts: np.ndarray | None = None) -> list[list[int]]:
        """小節ごとのトークン列 (終端トークン込み)。starts は演奏上の各小節の開始時刻で、省略するとテンポ記号から見積もる"""
        if starts is None:
            starts = self.nominal_starts(measures)
        bins = self.mtime_bins(starts)
        return [self.encode_measure(m, b, i == len(measures) - 1) for i, (m, b) in enumerate(zip(measures, bins))]

    def position_tokens(self, value: Fraction) -> list[int]:
        beat = value.numerator // value.denominator
        tokens = [self.token("beat", beat)]
        if value != beat:
            tokens.append(self.token("frac", value - beat))
        return tokens

    def encode_measure(self, measure: Measure, mtime_bin: int, last: bool) -> list[int]:
        tokens = [
            self.token("mtime", mtime_bin),
            self.token("ts", measure.time_signature),
            self.token("key", measure.key),
            self.token("clef", (1, measure.clefs[0])),
            self.token("clef", (2, measure.clefs[1])),
        ]
        tokens += [self.token("ottava", (s + 1, name)) for s, name in enumerate(measure.ottavas) if name != "none"]
        if measure.swing != "none":
            tokens.append(self.token("swing", measure.swing))
        if measure.length != measure.nominal_length:
            tokens += [MLEN, *self.position_tokens(measure.length)]
        directions: dict[Fraction, list[int]] = {}
        for position, name in measure.directions:
            if name.startswith("clef"):
                token = self.token("clef", (int(name[4]), name.split("_", 1)[1]))
            elif name.startswith("ottava"):
                token = self.token("ottava", (int(name[6]), name.split("_", 1)[1]))
            elif name.startswith("swing"):
                token = self.token("swing", name.split("_", 1)[1])
            else:
                token = self.token("direction", name)
            directions.setdefault(position, []).append(token)
        groups: dict[Fraction, list[Group]] = {}
        for g in measure.groups:
            groups.setdefault(g.onset, []).append(g)
        for position in sorted(set(directions) | set(groups)):
            tokens += self.position_tokens(position)
            tokens += sorted(set(directions.get(position, [])))
            # 同じ声部の中の順 (装飾音 -> 本体) は読み込んだ順のまま保つ (安定ソート)
            for g in sorted(groups.get(position, []), key=group_order):
                tokens.append(self.token("sv", (g.staff, g.voice)))
                if g.cross:
                    tokens.append(CROSS)
                tokens.append(self.token("dur", g.duration))
                tokens += [self.token("art", a) for a in ARTICULATIONS if a in g.articulations]
                tokens += [SLUR_STOP] * g.slur_stop + [SLUR_START] * g.slur_start
                for note in sorted(g.notes, key=lambda n: n.pitch):
                    tokens.append(self.token("pitch", (note.pitch, note.alter)))
                    if note.tie:
                        tokens.append(TIE)
                    if note.glissando:
                        tokens.append(GLISS)
        tokens.append(EOS if last else EOM)
        return tokens

    # ------------------------------------------------------------------
    # トークン -> 楽譜
    # ------------------------------------------------------------------
    def decode(self, patches: list[list[int]]) -> tuple[list[Measure], np.ndarray]:
        """小節ごとのトークン列を楽譜と各小節の開始時刻 (秒) に戻す。文法から外れたトークンは読み飛ばす"""
        measures: list[Measure] = []
        bins: list[int] = []
        previous = Measure((4, 4), 0, ("G2", "F4"), Fraction(4))
        for sequence in patches:
            measure, mtime = self._decode_measure(sequence, previous)
            measures.append(measure)
            bins.append(mtime)
            previous = measure
            if EOS in sequence:
                break
        return measures, self.decode_starts(bins)

    def _decode_measure(self, sequence: list[int], previous: Measure) -> tuple[Measure, int]:
        mtime = 0
        time_signature, key, clefs = previous.time_signature, previous.key, list(previous.clefs)
        ottavas = ["none", "none"]
        swing = "none"
        length: Fraction | None = None
        position: Fraction | None = None
        mode = "header"  # header / length / onset
        directions: list[tuple[Fraction, str]] = []
        groups: list[Group] = []
        group: Group | None = None
        for token in sequence:
            if token in (PAD, EOM, EOS):
                break
            kind, value = self.kinds[token], self.values[token]
            if kind == "mtime":
                mtime = value
            elif kind == "ts":
                time_signature = value
            elif kind == "key":
                key = value
            elif token == MLEN:
                mode = "length"
                length = None
            elif kind == "clef":
                if position is None:
                    clefs[value[0] - 1] = value[1]
                else:
                    directions.append((position, f"clef{value[0]}_{value[1]}"))
            elif kind == "ottava":
                if position is None:
                    ottavas[value[0] - 1] = value[1]
                else:
                    directions.append((position, f"ottava{value[0]}_{value[1]}"))
            elif kind == "swing":
                if position is None:
                    swing = value
                else:
                    directions.append((position, f"swing_{value}"))
            elif kind == "beat":
                if mode == "length" and length is None:
                    length = Fraction(value)
                else:
                    mode = "onset"
                    position = Fraction(value)
                group = None
            elif kind == "frac":
                if mode == "length" and length is not None:
                    length += value
                    mode = "onset"
                elif position is not None:
                    position += value
            elif kind == "direction" and position is not None:
                directions.append((position, value))
            elif kind == "sv" and position is not None:
                group = Group(position, value[0], value[1], None, [])  # type: ignore[arg-type]
                groups.append(group)
            elif token == CROSS and group is not None:
                group.cross = True
            elif kind == "dur" and group is not None and group.duration is None:
                group.duration = value
            elif kind == "art" and group is not None:
                group.articulations = (*group.articulations, value)
            elif token == SLUR_STOP and group is not None:
                group.slur_stop += 1
            elif token == SLUR_START and group is not None:
                group.slur_start += 1
            elif kind == "pitch" and group is not None and group.duration is not None:
                if all(n.pitch != value[0] for n in group.notes):
                    group.notes.append(Note(value[0], value[1]))
            elif token == TIE and group is not None and group.notes:
                group.notes[-1].tie = True
            elif token == GLISS and group is not None and group.notes:
                group.notes[-1].glissando = True
        measure = Measure(time_signature, key, (clefs[0], clefs[1]), Fraction(0), (ottavas[0], ottavas[1]), swing)
        measure.length = length if length is not None else measure.nominal_length
        measure.directions = sorted(set(directions))
        # 声部の中で前の音に重なる塊と、小節からはみ出す塊は捨てる (生成が文法から外れた場合の保険)
        busy: dict[tuple[int, int], Fraction] = {}
        for g in sorted((g for g in groups if g.duration is not None and g.notes), key=group_order):
            voice = (g.staff, g.voice)
            if g.onset < busy.get(voice, 0) or g.end > measure.length or g.onset >= measure.length:
                continue
            busy[voice] = g.end
            g.articulations = tuple(a for a in ARTICULATIONS if a in g.articulations)
            g.notes.sort(key=lambda n: n.pitch)
            measure.groups.append(g)
        return measure, mtime


def _spellable(pitch: int, alter: int) -> bool:
    try:
        spell(pitch, alter)
        return True
    except ValueError:
        return False
