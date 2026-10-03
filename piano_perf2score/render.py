"""楽譜 (score.py の Measure の列) から合成演奏 (MIDI の音とペダル) を描き出す。演奏 -> 楽譜モデルの 2 段目の入力。

楽譜をそのまま MIDI にすると、ピアニストの弾き方と食い違う書き方がある。次のように直して鳴らす。
- 同じ鍵の重なり: 別の声部・段が同時に同じ音を始めたら 1 回だけ打鍵して長いほうまでのばす。鳴っている間に別の声部が
  同じ音を始めたら、そこで打ち直して遅いほうの終わりまでのばす。描き出した後も同じ音高は重ねない
- タイは 1 つの音にまとめる (つなぐ先は musicxml._tie_stops と同じ規則)
- 装飾音は主音の直前 (または拍の上で主音を遅らせて) 短く弾く。トリル・モルデント・ターンは隣の音 (調号から決める) と
  交互に弾き、トレモロは斜線の本数の細かさで刻む (2 音間のトレモロは、書いた音価が半分なので 2 音分の長さで交互に)
- アルペジオは低い音から少しずつずらし (両手にあれば 1 本につなぐ)、グリッサンドは間の白鍵で埋める
- フェルマータはその音の終わりで時間を止めてのばし、スイングは 8 分 (16 分) を長短にする
- オクターブ記号は、読み込んだ音高がもともと実音なので何もしない

楽譜に書いていない演奏の要素 (テンポの揺れ・rit. などの速度の変化・強弱・アーティキュレーション・タイミングのずれ・
ペダル) は乱数で付ける。ペダル記号のない曲にも確率でペダルを足し、自動採譜の誤り (音の抜け・余計な音・オクターブの誤り)
も確率で混ぜる。小節の開始時刻 (MTIME の正解) は、最初の音ではなくテンポから決まる小節線の時刻。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from piano_score.musicxml import _tie_stops
from piano_score.score import (
    STEP_PITCH_CLASS,
    STEPS,
    TREMOLO_PAIR,
    Group,
    Measure,
    key_alters,
    measure_seconds,
    metronome_quarters,
    pair_slurs,
    spell,
)

DYNAMIC_LEVELS = {
    "pppp": 22, "ppp": 30, "pp": 40, "p": 50, "mp": 60, "mf": 70, "f": 82, "ff": 94, "fff": 106, "ffff": 116,
}  # fmt: skip
# その位置だけ強くする記号 (強弱の流れは変えない)。fp / sfp は強く弾いてすぐ p
ACCENT_DYNAMICS = {"sf": 18, "sfz": 20, "sffz": 24, "fz": 18, "rf": 14, "rfz": 16, "fp": 16, "sfp": 18}
SLOWING = ("tempo_rit", "tempo_rall", "tempo_allarg")
QUICKENING = ("tempo_accel", "tempo_string")


@dataclass(frozen=True)
class RenderConfig:
    # テンポ
    tempo_scale: tuple[float, float] = (0.8, 1.25)  # 曲全体の倍率 (対数一様。事前学習の data.py の倍率とは別)
    tempo_wander: float = 0.03  # 拍ごとのテンポの揺れ (対数、AR(1) の雑音の標準偏差)
    tempo_wander_keep: float = 0.9  # AR(1) の係数
    rit_ratio: tuple[float, float] = (0.6, 0.85)  # rit. で行き着く速さの比
    rit_beats: tuple[float, float] = (3.0, 10.0)  # rit. / accel. にかける拍数
    final_rit_prob: float = 0.7  # 曲の最後の 2 小節を遅くする確率
    fermata_ratio: tuple[float, float] = (1.5, 2.5)  # フェルマータの音の長さの倍率
    swing_ratio: tuple[float, float] = (1.5, 2.6)  # スイングの長短の比
    # タイミング
    onset_jitter: float = 0.008  # 同じ位置の音にまとめてかけるずれ (秒、標準偏差)
    note_jitter: float = 0.005  # 音ごとのずれ
    melody_lead: tuple[float, float] = (0.0, 0.02)  # 上段の最も高い声部を先に出す秒数
    arpeggio_gap: tuple[float, float] = (0.015, 0.045)
    grace_seconds: tuple[float, float] = (0.045, 0.09)
    ornament_rate: tuple[float, float] = (9.0, 15.0)  # トリルなどの 1 秒あたりの音数
    # 長さ (楽譜どおりの長さに対する比)
    staccato: tuple[float, float] = (0.35, 0.55)
    staccatissimo: tuple[float, float] = (0.18, 0.3)
    detached: tuple[float, float] = (0.82, 0.97)  # スラーもテヌートもない音
    legato_overlap: tuple[float, float] = (0.0, 0.04)  # スラーの中で次の音に重ねる秒数
    # 強さ
    velocity_spread: tuple[float, float] = (0.6, 1.2)  # 強弱記号の差をどれだけ付けるか
    velocity_shift: tuple[float, float] = (-10.0, 8.0)
    velocity_noise: float = 5.0
    melody_boost: tuple[float, float] = (3.0, 10.0)
    # ペダル
    auto_pedal_prob: float = 0.75  # ペダル記号のない曲にペダルを足す確率
    # 自動採譜の誤り
    noise_prob: float = 0.5  # 誤りを混ぜる曲の割合
    drop_rate: tuple[float, float] = (0.0, 0.03)
    extra_rate: tuple[float, float] = (0.0, 0.02)
    octave_error_rate: float = 0.003


@dataclass
class Performance:
    notes: np.ndarray  # [N, 4] 打鍵の時刻・離鍵の時刻 (秒)・音高・ベロシティ。打鍵の順
    pedal: np.ndarray  # [K, 2] 時刻・踏んでいるか (1 / 0)。時刻の順
    measure_starts: np.ndarray  # [M] 各小節の小節線の時刻 (秒)。最初の音 = 0


@dataclass
class _Note:
    on: float  # 曲の頭からの位置 (4 分音符単位)
    end: float
    pitch: int
    group: Group
    measure: int
    alter: int
    top: bool  # 上段の塊のいちばん高い音
    gliss_to: int | None = None
    ornament: str | None = None
    cut: bool = False  # 後の同じ音の打ち直しで切った音


def measure_qpm(measures: list[Measure], seconds: np.ndarray | None = None) -> np.ndarray:
    """小節ごとのテンポ (4 分音符/分)。seconds (各小節の開始時刻、prepare.py のキャッシュ) があればそこから求める"""
    if seconds is None:
        seconds = measure_seconds(measures)
    lengths = np.array([float(m.length) for m in measures])
    spans = np.diff(np.asarray(seconds, dtype=np.float64))
    qpm = np.full(len(measures), 120.0)
    n = min(len(spans), len(measures))
    qpm[:n] = np.where(spans[:n] > 0, lengths[:n] * 60.0 / np.maximum(spans[:n], 1e-6), 120.0)
    if len(spans) < len(measures) and len(measures) > 1:
        qpm[len(spans) :] = qpm[len(spans) - 1]
    return np.clip(qpm, 20.0, 400.0)


def render(
    measures: list[Measure],
    rng: np.random.Generator,
    config: RenderConfig = RenderConfig(),
    qpm: np.ndarray | None = None,
) -> Performance:
    """楽譜を合成演奏にする。qpm は小節ごとのテンポ (measure_qpm)。なければテンポ記号から求める"""
    c = config
    if qpm is None:
        qpm = measure_qpm(measures)
    offsets = np.concatenate([[0.0], np.cumsum([float(m.length) for m in measures])])
    timeline = _Timeline(measures, offsets, qpm, rng, c)

    notes = _score_notes(measures, offsets)
    graces = _grace_notes(measures, offsets)
    legato = _slurred_groups(measures)
    song = {
        "detached": rng.uniform(*c.detached),
        "lead": rng.uniform(*c.melody_lead),
        "spread": rng.uniform(*c.velocity_spread),
        "shift": rng.uniform(*c.velocity_shift),
        "boost": rng.uniform(*c.melody_boost),
        "rate": rng.uniform(*c.ornament_rate),
        "arpeggio": rng.uniform(*c.arpeggio_gap),
    }
    dynamics = _Dynamics(measures, offsets, rng)

    # 位置ごとのずれ (同じ位置の音はまとめてずらす)
    cluster_shift: dict[float, float] = {}

    def shift_at(position: float) -> float:
        if position not in cluster_shift:
            cluster_shift[position] = rng.normal(0.0, c.onset_jitter)
        return cluster_shift[position]

    # アルペジオ: 同じ位置でアルペジオのある塊の音を、低い音から順にずらす (両手のものは 1 本につなぐ)
    rolls: dict[float, list[int]] = {}
    for n in notes:
        if "arpeggiate" in n.group.articulations:
            rolls.setdefault(n.on, []).append(n.pitch)
    roll_offset = {
        (on, p): k * song["arpeggio"] for on, pitches in rolls.items() for k, p in enumerate(sorted(set(pitches)))
    }

    out: list[list[float]] = []  # [on, off, pitch, velocity]
    for n in notes:
        on = timeline.time(n.on) + shift_at(n.on) + rng.normal(0.0, c.note_jitter)
        on += roll_offset.get((n.on, n.pitch), 0.0)
        if n.top and n.group.staff == 1:
            on -= song["lead"]
        written_end = timeline.time(n.end)
        span = max(written_end - timeline.time(n.on), 0.02)
        arts = n.group.articulations
        if n.cut:
            off = written_end - 0.01
        elif "staccatissimo" in arts:
            off = on + span * rng.uniform(*c.staccatissimo)
        elif "staccato" in arts:
            off = on + span * rng.uniform(*c.staccato)
        elif id(n.group) in legato or "tenuto" in arts:
            off = written_end + (rng.uniform(*c.legato_overlap) if id(n.group) in legato else 0.0)
        else:
            off = on + span * song["detached"] * rng.uniform(0.97, 1.03)
        off = max(off, on + 0.03)
        velocity = dynamics.velocity(n, song, rng)
        if n.ornament is not None:
            out += _ornament(n, on, off, velocity, measures, song["rate"], rng)
        elif n.gliss_to is not None:
            out += _glissando(n.pitch, n.gliss_to, on, off, velocity)
        else:
            out.append([on, off, n.pitch, velocity])

    # 装飾音: 斜線ありは主に拍の前、斜線なしは主に拍の上 (主音を遅らせる)
    for main_on, main_pitch_list, items in graces:
        if not items:
            continue
        slash = items[0][1]
        before = rng.random() < (0.8 if slash else 0.4)
        d = rng.uniform(*c.grace_seconds)
        t = timeline.time(main_on) + shift_at(main_on)
        k = len(items)
        velocity = float(np.clip(66.0 + (dynamics.level(main_on) - 66.0) * song["spread"] + song["shift"] - 8, 8, 127))
        for i, (pitch, _) in enumerate(items):
            start = t - (k - i) * d if before else t + i * d
            out.append([start, start + d * 1.1, pitch, velocity])
        if not before:
            for row in out:
                if abs(row[0] - t) < 0.06 and row[2] in main_pitch_list:
                    row[0] = max(row[0], t + k * d)
                    row[1] = max(row[1], row[0] + 0.03)

    performance = np.array(out, dtype=np.float64).reshape(-1, 4)
    pedal = _pedal(measures, offsets, timeline, notes, rng, c)
    if rng.random() < c.noise_prob:
        performance = _transcription_errors(performance, rng, c)
    performance = _resolve_overlaps(performance)
    starts = np.array([timeline.time(offsets[i]) for i in range(len(measures))])

    # 最初の音を 0 秒にする
    origin = performance[:, 0].min() if len(performance) else 0.0
    performance[:, :2] -= origin
    if len(pedal):
        pedal[:, 0] -= origin
    return Performance(performance, pedal, starts - origin)


# ----------------------------------------------------------------------
# 楽譜の音 -> 打鍵する音 (タイ・同じ鍵の重なり・トレモロ・グリッサンド・装飾記号)
# ----------------------------------------------------------------------
def _score_notes(measures: list[Measure], offsets: np.ndarray) -> list[_Note]:
    continued = _tie_stops(measures)
    groups = [(mi, g) for mi, m in enumerate(measures) for g in sorted(m.groups, key=_group_key) if not g.duration.grace]
    tremolo_pairs = _tremolo_pairs(measures)
    gliss_targets = _glissando_targets(measures)
    notes: list[_Note] = []
    active: dict[int, _Note] = {}  # 音高 -> タイでのばしている最中の音
    skip: set[int] = set()
    for mi, g in groups:
        if id(g) in skip:
            continue
        base = offsets[mi]
        on, end = base + float(g.onset), base + float(g.end)
        top_pitch = max(n.pitch for n in g.notes)
        ornament = next((a for a in g.articulations if a in _ORNAMENT_STEPS), None)
        tremolo = next((int(a[-1]) for a in g.articulations if a.startswith("tremolo")), 0)
        if id(g) in tremolo_pairs:
            # 2 音間のトレモロ: 書いた音価は半分なので、2 つ目の塊の終わりまでを交互に刻む
            mj, h = tremolo_pairs[id(g)]
            skip.add(id(h))
            span_end = offsets[mj] + float(h.end)
            step = 0.5 / 2 ** (max(tremolo, 1) - 1)
            k, t = 0, on
            while t < span_end - 1e-9:
                which = g if k % 2 == 0 else h
                for n in which.notes:
                    notes.append(_Note(t, min(t + step, span_end), n.pitch, which, mi, n.alter, n.pitch == top_pitch))
                t += step
                k += 1
            continue
        if tremolo:
            # 1 音のトレモロ: 斜線 1 本で 8 分、2 本で 16 分 ... に刻む
            step = 0.5 / 2 ** (tremolo - 1)
            t = on
            while t < end - 1e-9:
                for n in g.notes:
                    notes.append(_Note(t, min(t + step, end), n.pitch, g, mi, n.alter, n.pitch == top_pitch))
                t += step
            continue
        for n in g.notes:
            previous = active.get(n.pitch)
            if (mi, id(g), n.pitch) in continued and previous is not None and abs(previous.end - on) < 1e-6:
                previous.end = end  # タイで続く音は打鍵しない
            else:
                note = _Note(on, end, n.pitch, g, mi, n.alter, n.pitch == top_pitch)
                if n.pitch == top_pitch and ornament:
                    note.ornament = ornament
                if n.glissando:
                    note.gliss_to = gliss_targets.get((mi, id(g), n.pitch))
                notes.append(note)
                previous = note
            if n.tie:
                active[n.pitch] = previous
            else:
                active.pop(n.pitch, None)
    return _merge_same_keys(notes)


def _group_key(g: Group) -> tuple:
    return (g.onset, g.staff, g.voice, g.duration.grace is None)


def _merge_same_keys(notes: list[_Note]) -> list[_Note]:
    """同じ鍵の重なりを直す。同時に始まるものは 1 つにまとめ、鳴っている間に始まるものはそこで打ち直す"""
    by_pitch: dict[int, list[_Note]] = {}
    for n in notes:
        by_pitch.setdefault(n.pitch, []).append(n)
    kept: list[_Note] = []
    for same in by_pitch.values():
        same.sort(key=lambda n: (n.on, -n.end))
        current = same[0]
        for n in same[1:]:
            if abs(n.on - current.on) < 1e-6:
                current.end = max(current.end, n.end)  # 同時に始まる同じ音は 1 回だけ打鍵する
                continue
            if n.on < current.end - 1e-6:
                n.end = max(n.end, current.end)  # 打ち直して、遅いほうの終わりまでのばす
                current.end = n.on
                current.cut = True
            kept.append(current)
            current = n
        kept.append(current)
    kept.sort(key=lambda n: (n.on, n.pitch))
    return kept


def _tremolo_pairs(measures: list[Measure]) -> dict[int, tuple[int, Group]]:
    """2 音間のトレモロの始まりの塊 -> (小節番号, 終わりの塊)。終わりは同じ声部ですぐ後に続く 2:1 の塊"""
    pairs = {}
    for mi, m in enumerate(measures):
        for g in m.groups:
            if g.duration.tuplet != TREMOLO_PAIR or not any(a.startswith("tremolo") for a in g.articulations):
                continue
            candidates = [(mi, h) for h in m.groups if h.onset == g.end]
            if g.end >= m.length and mi + 1 < len(measures):
                candidates += [(mi + 1, h) for h in measures[mi + 1].groups if h.onset == 0]
            for mj, h in candidates:
                if (h.staff, h.voice) == (g.staff, g.voice) and h.duration.tuplet == TREMOLO_PAIR and h is not g:
                    pairs[id(g)] = (mj, h)
                    break
    return pairs


def _glissando_targets(measures: list[Measure]) -> dict[tuple[int, int, int], int]:
    """グリッサンドの始まりの音 -> 行き先の音高 (同じ声部の次の塊の、いちばん近い音)"""
    targets = {}
    flat = [(mi, g) for mi, m in enumerate(measures) for g in sorted(m.groups, key=_group_key) if not g.duration.grace]
    for i, (mi, g) in enumerate(flat):
        for n in g.notes:
            if not n.glissando:
                continue
            following = next(((mj, h) for mj, h in flat[i + 1 :] if (h.staff, h.voice) == (g.staff, g.voice)), None)
            if following is not None:
                targets[(mi, id(g), n.pitch)] = min((h.pitch for h in following[1].notes), key=lambda p: abs(p - n.pitch))
    return targets


def _grace_notes(measures: list[Measure], offsets: np.ndarray) -> list[tuple[float, list[int], list[tuple[int, bool]]]]:
    """装飾音を、同じ位置・同じ声部の主音ごとにまとめる。(主音の位置, 主音の音高, [(装飾音の音高, 斜線あり)])"""
    result = []
    for mi, m in enumerate(measures):
        voices: dict[tuple[Fraction, int, int], list[Group]] = {}
        for g in sorted(m.groups, key=_group_key):
            voices.setdefault((g.onset, g.staff, g.voice), []).append(g)
        for (onset, _, _), groups in voices.items():
            graces = [g for g in groups if g.duration.grace]
            mains = [g for g in groups if not g.duration.grace]
            if not graces:
                continue
            items = [(max(n.pitch for n in g.notes), g.duration.grace == "slash") for g in graces]
            main_pitches = [n.pitch for g in mains for n in g.notes]
            result.append((offsets[mi] + float(onset), main_pitches, items))
    return result


def _slurred_groups(measures: list[Measure]) -> set[int]:
    """スラーの中の塊 (終わりの塊は除く)。同じ声部で始まりから終わりの手前まで"""
    events = pair_slurs(measures)
    starts: dict[int, tuple[int, Group]] = {}
    spans = []
    for mi, g, kind, slur_id in events:
        if kind == "start":
            starts[slur_id] = (mi, g)
        elif slur_id in starts:
            spans.append((starts.pop(slur_id), (mi, g)))
    inside: set[int] = set()
    for (m0, g0), (m1, g1) in spans:
        for mi in range(m0, m1 + 1):
            for g in measures[mi].groups:
                if (g.staff, g.voice) != (g0.staff, g0.voice):
                    continue
                after_start = mi > m0 or g.onset >= g0.onset
                before_end = mi < m1 or g.onset < g1.onset
                if after_start and before_end:
                    inside.add(id(g))
    return inside


# ----------------------------------------------------------------------
# 装飾記号とグリッサンドの展開
# ----------------------------------------------------------------------
# 主音から見た音の並び (+1 = 上の隣の音、-1 = 下の隣の音)。最後の音が残りの長さを受け持つ
_ORNAMENT_STEPS = {
    "mordent": (0, -1, 0),
    "inverted-mordent": (0, 1, 0),
    "turn": (1, 0, -1, 0),
    "inverted-turn": (-1, 0, 1, 0),
    "trill-mark": None,  # 長さに合わせて主音と上の音を交互に
}


def _neighbor(pitch: int, alter: int, direction: int, key: int) -> int:
    """調号に従った上 (direction=1) / 下 (-1) の隣の音"""
    try:
        step, octave = spell(pitch, alter)
    except ValueError:
        return pitch + direction * 2
    index = STEPS.index(step) + direction
    octave += index // 7
    step = STEPS[index % 7]
    return (octave + 1) * 12 + STEP_PITCH_CLASS[step] + key_alters(key).get(step, 0)


def _ornament(
    n: _Note, on: float, off: float, velocity: float, measures: list[Measure], rate: float, rng: np.random.Generator
) -> list[list[float]]:
    key = measures[n.measure].key
    upper, lower = _neighbor(n.pitch, n.alter, 1, key), _neighbor(n.pitch, n.alter, -1, key)
    d = 1.0 / rate
    span = off - on
    if n.ornament == "trill-mark":
        count = max(3, int(span / d))
        start_upper = rng.random() < 0.3
        pitches = [(upper if (k % 2 == 0) == start_upper else n.pitch) for k in range(count)]
        if pitches[-1] != n.pitch:
            pitches.append(n.pitch)
        d = span / len(pitches)
    else:
        pitches = [{0: n.pitch, 1: upper, -1: lower}[s] for s in _ORNAMENT_STEPS[n.ornament]]
        d = min(d, span / len(pitches))
    rows = []
    for k, p in enumerate(pitches):
        start = on + k * d
        end = off if k == len(pitches) - 1 else start + d * 1.05
        rows.append([start, end, p, velocity * (1.0 if k == 0 else 0.85)])
    return rows


def _glissando(start: int, target: int, on: float, off: float, velocity: float) -> list[list[float]]:
    """始まりの音から行き先の手前までを白鍵で埋める (始まりの音の長さの中に並べる)"""
    direction = 1 if target > start else -1
    keys = [p for p in range(start + direction, target, direction) if p % 12 in (0, 2, 4, 5, 7, 9, 11)]
    pitches = [start, *keys]
    d = (off - on) / len(pitches)
    return [[on + k * d, on + (k + 1) * d * 1.02, p, velocity * (1.0 if k == 0 else 0.8)] for k, p in enumerate(pitches)]


# ----------------------------------------------------------------------
# テンポ (楽譜の位置 -> 秒)
# ----------------------------------------------------------------------
class _Timeline:
    """楽譜の位置 (曲の頭からの 4 分音符単位) を秒にする。拍ごとのテンポに、全体の倍率・揺れ・rit. などの変化をかけ、
    フェルマータの位置では時間を足す。スイングは位置を拍の中で長短にずらしてから秒にする"""

    def __init__(
        self, measures: list[Measure], offsets: np.ndarray, qpm: np.ndarray, rng: np.random.Generator, c: RenderConfig
    ) -> None:
        self.offsets = offsets
        scale = math.exp(rng.uniform(math.log(c.tempo_scale[0]), math.log(c.tempo_scale[1])))
        self.swing = _swing_spans(measures, offsets)
        self.swing_ratio = rng.uniform(*c.swing_ratio)

        # 拍 (4 分音符) ごとの区切り
        knots = [0.0]
        for i, m in enumerate(measures):
            position = 1.0
            while position < float(m.length) - 1e-9:
                knots.append(offsets[i] + position)
                position += 1.0
            knots.append(offsets[i + 1])
        knots = np.array(knots)
        segment_measure = np.searchsorted(offsets, knots[:-1], side="right") - 1
        base = qpm[np.clip(segment_measure, 0, len(qpm) - 1)] * scale

        # 速度の変化の指示を拍の位置に並べる
        changes = sorted(
            (offsets[i] + float(p), name)
            for i, m in enumerate(measures)
            for p, name in m.directions
            if name.startswith(("tempo_", "mark_", "metronome_"))
        )
        factor = np.ones(len(base))
        level, target, remaining, wander = 1.0, 1.0, 0, 0.0
        change_index = 0
        for k in range(len(base)):
            while change_index < len(changes) and changes[change_index][0] <= knots[k] + 1e-9:
                name = changes[change_index][1]
                beats = int(rng.uniform(*c.rit_beats))
                if name in SLOWING:
                    target, remaining = level * rng.uniform(*c.rit_ratio), beats
                elif name in QUICKENING:
                    target, remaining = level / rng.uniform(*c.rit_ratio), beats
                elif name == "tempo_meno_mosso":
                    level = target = level * rng.uniform(0.8, 0.92)
                elif name == "tempo_piu_mosso":
                    level = target = level * rng.uniform(1.08, 1.25)
                else:  # a tempo・速度標語・メトロノーム記号で元に戻す (テンポそのものは qpm に入っている)
                    level, target, remaining = 1.0, 1.0, 0
                change_index += 1
            if remaining > 0:
                level += (target - level) / remaining
                remaining -= 1
            wander = c.tempo_wander_keep * wander + rng.normal(0.0, c.tempo_wander)
            factor[k] = level * math.exp(wander)
        # 曲の最後の 2 小節を遅くする
        if len(measures) >= 2 and rng.random() < c.final_rit_prob:
            final = knots[:-1] >= offsets[-3]
            ramp = np.linspace(1.0, rng.uniform(*c.rit_ratio), final.sum()) if final.any() else []
            factor[final] *= ramp
        seconds = np.diff(knots) * 60.0 / (base * factor)
        self.knots = knots
        self.times = np.concatenate([[0.0], np.cumsum(seconds)])

        # フェルマータ: その塊の終わりで、音の長さを伸ばした分だけ時間を足す
        pauses = []
        for i, m in enumerate(measures):
            for g in m.groups:
                if "fermata" in g.articulations and not g.duration.grace:
                    end = offsets[i] + float(g.end)
                    length = self._base_time(end) - self._base_time(offsets[i] + float(g.onset))
                    pauses.append((end, length * (rng.uniform(*c.fermata_ratio) - 1.0) + rng.uniform(0.05, 0.3)))
        pauses.sort()
        merged: dict[float, float] = {}
        for at, extra in pauses:
            merged[at] = max(merged.get(at, 0.0), extra)  # 同じ位置のフェルマータ (和音・両手) は 1 回分
        self.pause_at = np.array(sorted(merged), dtype=np.float64)
        self.pause_cum = np.cumsum([merged[k] for k in sorted(merged)]) if merged else np.zeros(0)

    def _base_time(self, position: float) -> float:
        return float(np.interp(position, self.knots, self.times))

    def time(self, position: float) -> float:
        position = self._swung(position)
        t = self._base_time(position)
        if len(self.pause_at):
            k = np.searchsorted(self.pause_at, position + 1e-9, side="right")
            if k:
                t += self.pause_cum[k - 1]
        return t

    def _swung(self, position: float) -> float:
        """スイングの範囲では、拍 (8 分のスイングなら 4 分音符、16 分なら 8 分音符) の中の前半を長くする"""
        for start, end, unit in self.swing:
            if start <= position < end:
                beat = math.floor((position - start) / unit) * unit + start
                x = (position - beat) / unit
                long = self.swing_ratio / (1.0 + self.swing_ratio)
                y = x * 2 * long if x <= 0.5 else long + (x - 0.5) * 2 * (1 - long)
                return beat + y * unit
        return position


def _swing_spans(measures: list[Measure], offsets: np.ndarray) -> list[tuple[float, float, float]]:
    """スイングの範囲 (始まり, 終わり, 拍の長さ)"""
    spans = []
    state, since = "none", 0.0
    for i, m in enumerate(measures):
        events = [(Fraction(0), m.swing)] + [(p, n.split("_", 1)[1]) for p, n in m.directions if n.startswith("swing_")]
        for p, value in events:
            at = offsets[i] + float(p)
            if value != state:
                if state != "none":
                    spans.append((since, at, 1.0 if state == "8th" else 0.5))
                state, since = value, at
    if state != "none":
        spans.append((since, offsets[-1], 1.0 if state == "8th" else 0.5))
    return spans


# ----------------------------------------------------------------------
# 強さ
# ----------------------------------------------------------------------
class _Dynamics:
    """強弱記号・松葉・cresc. / dim. から、位置ごとの強さ (MIDI のベロシティの目安) の折れ線を作る"""

    def __init__(self, measures: list[Measure], offsets: np.ndarray, rng: np.random.Generator) -> None:
        events = sorted(
            (offsets[i] + float(p), name) for i, m in enumerate(measures) for p, name in m.directions
            if name.startswith(("dyn_", "wedge_", "text_cresc", "text_dim"))
        )  # fmt: skip
        level = rng.uniform(58.0, 76.0)  # 強弱記号がないところは mf 前後
        knots = [(0.0, level)]
        self.accents: dict[float, float] = {}
        ramp_start: tuple[float, float, int] | None = None  # (位置, 強さ, +1 / -1)
        for at, name in events:
            if name.startswith("dyn_"):
                mark = name[4:]
                if mark in DYNAMIC_LEVELS:
                    new = DYNAMIC_LEVELS[mark] + rng.normal(0.0, 3.0)
                    if ramp_start is not None:  # 松葉の行き先
                        knots.append((at, new))
                        ramp_start = None
                    else:
                        knots += [(at - 1e-6, level), (at, new)]
                    level = new
                elif mark in ACCENT_DYNAMICS:
                    self.accents[at] = ACCENT_DYNAMICS[mark]
                    if mark in ("fp", "sfp"):
                        knots += [(at - 1e-6, level), (at, DYNAMIC_LEVELS["p"])]
                        level = DYNAMIC_LEVELS["p"]
            elif name in ("wedge_crescendo", "text_cresc", "wedge_diminuendo", "text_dim"):
                if ramp_start is not None:
                    level = self._close(knots, ramp_start, at)
                ramp_start = (at, level, 1 if "cresc" in name else -1)
                knots.append((at, level))
            elif name == "wedge_stop" and ramp_start is not None:
                level = self._close(knots, ramp_start, at)
                ramp_start = None
        if ramp_start is not None:
            level = self._close(knots, ramp_start, ramp_start[0] + 8.0)
        knots.append((offsets[-1] + 1.0, level))
        knots.sort()
        self.positions = np.array([k[0] for k in knots])
        self.values = np.array([k[1] for k in knots])

    @staticmethod
    def _close(knots: list, start: tuple[float, float, int], at: float) -> float:
        """行き先の強弱記号のない松葉は、長さに応じて ±8〜16 だけ変える"""
        position, level, sign = start
        change = min(16.0, 8.0 + 2.0 * (at - position))
        new = float(np.clip(level + sign * change, 20, 120))
        knots.append((at, new))
        return new

    def level(self, position: float) -> float:
        return float(np.interp(position, self.positions, self.values))

    def velocity(self, n: _Note, song: dict, rng: np.random.Generator) -> float:
        v = 66.0 + (self.level(n.on) - 66.0) * song["spread"] + song["shift"]
        arts = n.group.articulations
        if "strong-accent" in arts:
            v += 16
        elif "accent" in arts:
            v += 10
        v += self.accents.get(n.on, 0.0)
        if n.top and n.group.staff == 1:
            v += song["boost"]
        elif n.group.staff == 2:
            v -= 3
        v += rng.normal(0.0, 5.0)
        return float(np.clip(v, 8, 127))


# ----------------------------------------------------------------------
# ペダル
# ----------------------------------------------------------------------
def _pedal(
    measures: list[Measure],
    offsets: np.ndarray,
    timeline: _Timeline,
    notes: list[_Note],
    rng: np.random.Generator,
    c: RenderConfig,
) -> np.ndarray:
    """ペダル記号どおりに踏む。記号がなければ確率で、下段の最も低い音が変わるところ (和声の変わり目の目安) で踏み替える"""
    marks = [(offsets[i] + float(p), name) for i, m in enumerate(measures) for p, name in m.directions]
    marks = [(at, name) for at, name in marks if name in ("pedal_start", "pedal_stop")]
    events: list[tuple[float, int]] = []
    if marks:
        for at, name in sorted(marks, key=lambda x: (x[0], x[1] == "pedal_start")):
            t = timeline.time(at)
            if name == "pedal_stop":
                events.append((t + rng.uniform(0.0, 0.03), 0))
            else:
                events.append((t + rng.uniform(0.04, 0.12), 1))
    elif rng.random() < c.auto_pedal_prob:
        # 拍の頭にある下段の音で、最も低い音が前と変わるところ。スタッカートの多い小節では踏まない
        bass: dict[float, int] = {}
        for n in notes:
            if n.group.staff == 2 and abs(n.on - round(n.on)) < 1e-6:
                bass[n.on] = min(bass.get(n.on, 999), n.pitch)
        staccato_measures = set()
        for i, m in enumerate(measures):
            total = sum(len(g.notes) for g in m.groups)
            short = sum(len(g.notes) for g in m.groups if {"staccato", "staccatissimo"} & set(g.articulations))
            if total and short / total > 0.4:
                staccato_measures.add(i)
        previous = None
        down = False
        for at in sorted(bass):
            measure = int(np.searchsorted(offsets, at, side="right") - 1)
            if measure in staccato_measures:
                if down:
                    events.append((timeline.time(at) + rng.uniform(0.0, 0.03), 0))
                    down = False
                previous = None
                continue
            if previous is not None and bass[at] % 12 == previous % 12 and rng.random() < 0.8:
                continue
            t = timeline.time(at)
            if down:
                events.append((t + rng.uniform(0.0, 0.03), 0))
            events.append((t + rng.uniform(0.05, 0.14), 1))
            down = True
            previous = bass[at]
        if down:
            events.append((timeline.time(offsets[-1]) + rng.uniform(0.2, 1.0), 0))
    events.sort()
    return np.array(events, dtype=np.float64).reshape(-1, 2)


# ----------------------------------------------------------------------
# 自動採譜の誤りと、同じ鍵の重なりの後始末
# ----------------------------------------------------------------------
def _transcription_errors(notes: np.ndarray, rng: np.random.Generator, c: RenderConfig) -> np.ndarray:
    """音の抜け (弱く短い音ほど抜けやすい)・余計な音 (強い音の倍音あたり)・オクターブの誤りを混ぜる"""
    if not len(notes):
        return notes
    drop = rng.uniform(*c.drop_rate)
    weakness = np.clip((80 - notes[:, 3]) / 60, 0.2, 1.5) * np.clip(0.3 / np.maximum(notes[:, 1] - notes[:, 0], 0.05), 0.5, 2)
    keep = rng.random(len(notes)) >= drop * weakness
    notes = notes[keep]
    octave = rng.random(len(notes)) < c.octave_error_rate
    notes[octave, 2] += rng.choice([-12, 12], octave.sum())
    extra_rate = rng.uniform(*c.extra_rate)
    loud = np.flatnonzero((notes[:, 3] > 60) & (rng.random(len(notes)) < extra_rate))
    if len(loud):
        extra = notes[loud].copy()
        extra[:, 2] += rng.choice([12, 19, 24, -12], len(loud))
        extra[:, 0] += rng.uniform(-0.01, 0.02, len(loud))
        extra[:, 1] = extra[:, 0] + rng.uniform(0.04, 0.2, len(loud))
        extra[:, 3] = rng.uniform(15, 40, len(loud))
        notes = np.concatenate([notes, extra])
    return notes


def _resolve_overlaps(notes: np.ndarray) -> np.ndarray:
    """88 鍵の外の音を捨て、同じ音高の音が重ならないようにする。ほぼ同時 (20ms 未満) の同じ音は 1 つにまとめ、
    それ以外は前の音を次の打鍵の少し手前で切る"""
    notes = notes[(notes[:, 2] >= 21) & (notes[:, 2] <= 108)]
    notes = notes[np.lexsort((notes[:, 0], notes[:, 2]))]
    keep = np.ones(len(notes), dtype=bool)
    last = -1
    for i in range(len(notes)):
        if last >= 0 and notes[i, 2] == notes[last, 2]:
            if notes[i, 0] - notes[last, 0] < 0.02:
                notes[last, 1] = max(notes[last, 1], notes[i, 1])
                notes[last, 3] = max(notes[last, 3], notes[i, 3])
                keep[i] = False
                continue
            notes[last, 1] = min(notes[last, 1], notes[i, 0] - 0.005)
        last = i
    notes = notes[keep]
    notes[:, 1] = np.maximum(notes[:, 1], notes[:, 0] + 0.01)
    return notes[np.argsort(notes[:, 0], kind="stable")]
