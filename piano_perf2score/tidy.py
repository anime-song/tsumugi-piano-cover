"""楽譜の記譜の不整合を直す後処理 (拍の上の演奏 -> 楽譜で、V2 のカバーを楽譜にしたもの向け)。

モデルは小節ごとに書くので、小節をまたぐ整合が崩れることがある。書き出す前に次を直す。
    - 冒頭の音のない小節 (カバーの冒頭の無音を書いたもの。弱起の短い小節のことが多い) を消し、指示を次の小節の頭へ移す
    - 行き先のないタイ (次の位置の同じ段に同じ音がない) を消す
    - ペダル: 踏んだまま踏む指示は踏み替え (離す + 踏むを同じ位置に置き、書き出しで踏み替えの記号になる) にし、
      離したまま離す指示は消す。曲の終わりで踏んだままなら最後に離す
    - 今と同じ強弱記号の繰り返しを消す
    - 調号と綴り: 調号が同じ小節の並びごとに、音名の数え上げから調号を選び直す (はっきり合うときだけ)。
      そのうえで音名ごとに綴りを 1 つに決める。重変化記号と、調号にない白鍵の異名 (Cb・Fb・E#・B#) は使わない
    - ただし調の外の音が同じ声部の次の音へ半音で進むときは、進む向きで綴る (chromatic_spelling)。上がる音は進む先の
      1 つ下の幹音に # (ニ長調の E-E#-F#、F#-F##-G#。この形だけ重変化記号も使う)、下がる音は 1 つ上の幹音に b
      (黒鍵のときだけ。A-Ab は Bbb-Ab と書かない)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from itertools import groupby, pairwise

from piano_score.score import STEP_PITCH_CLASS, STEPS, Measure, spell

# 五度の並びの上の位置 (C = 0、G = 1、F = -1)。綴りの位置は 幹音の位置 + 7 x 変化記号
FIFTHS = {"F": -1, "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5}
KEY_MARGIN = 0.03  # 調号を選び直すのは、音名の当てはまりがこの割合以上よくなるときだけ


@dataclass
class TidyStats:
    rests: int = 0  # 消した冒頭の音のない小節
    ties: int = 0  # 消した行き先のないタイ
    pedals: int = 0  # 直したペダルの指示
    dynamics: int = 0  # 消した繰り返しの強弱記号
    keys: int = 0  # 調号を選び直した小節の並び
    spellings: int = 0  # 綴りを変えた音
    chromatic: int = 0  # 半音の進みの向きで綴り直した音


def _drop_leading_rests(measures: list[Measure]) -> int:
    dropped = 0
    while len(measures) > 1 and not measures[0].groups:
        first, following = measures[0], measures[1]
        following.directions = [(Fraction(0), name) for _, name in first.directions] + following.directions
        if not any(position == 0 for position, _ in following.tempos):
            following.tempos = [(Fraction(0), tempo) for _, tempo in first.tempos[-1:]] + following.tempos
        del measures[0]
        dropped += 1
    return dropped


def _drop_dangling_ties(measures: list[Measure]) -> int:
    dropped = 0
    for mi, measure in enumerate(measures):
        for g in measure.groups:
            for note in g.notes:
                if not note.tie:
                    continue
                if g.end < measure.length:
                    candidates = [h for h in measure.groups if h.onset == g.end]
                elif mi + 1 < len(measures):
                    candidates = [h for h in measures[mi + 1].groups if h.onset == 0]
                else:
                    candidates = []
                if not any(
                    h.staff == g.staff and h.duration.grace is None and any(n.pitch == note.pitch for n in h.notes)
                    for h in candidates
                ):
                    note.tie = False
                    dropped += 1
    return dropped


def _fix_pedals(measures: list[Measure]) -> int:
    fixed = 0
    down = False
    for measure in measures:
        others = [d for d in measure.directions if not d[1].startswith("pedal_")]
        pedals = [d for d in measure.directions if d[1].startswith("pedal_")]
        pedals.sort(key=lambda d: (d[0], d[1] == "pedal_start"))
        kept = []
        for position, name in pedals:
            if name == "pedal_stop" and not down:
                fixed += 1
                continue
            if name == "pedal_start" and down:
                kept.append((position, "pedal_stop"))
                fixed += 1
            kept.append((position, name))
            down = name == "pedal_start"
        measure.directions = others + kept
    if down:
        measures[-1].directions.append((measures[-1].length, "pedal_stop"))
        fixed += 1
    return fixed


def _drop_repeated_dynamics(measures: list[Measure]) -> int:
    dropped = 0
    current = None
    for measure in measures:
        kept = []
        for position, name in sorted(measure.directions, key=lambda d: d[0]):
            if name.startswith("dyn_"):
                if name == current:
                    dropped += 1
                    continue
                current = name
            kept.append((position, name))
        measure.directions = kept
    return dropped


def _diatonic(position: int, key: int) -> bool:
    return key - 1 <= position <= key + 5


def _position(pitch_class: int, alter: int) -> int | None:
    """綴りの五度の並びの上の位置。その変化記号では綴れなければ None"""
    step = next((s for s, pc in STEP_PITCH_CLASS.items() if pc == (pitch_class - alter) % 12), None)
    return None if step is None else FIFTHS[step] + 7 * alter


def _spellings(pitch_class: int, key: int) -> list[int]:
    """使ってよい綴り (変化記号) の候補。重変化記号は使わず、白鍵の異名は調号にあるときだけ"""
    white = pitch_class in STEP_PITCH_CLASS.values()
    return [
        alter
        for alter in (-1, 0, 1)
        if (position := _position(pitch_class, alter)) is not None
        and (alter == 0 or not white or _diatonic(position, key))
    ]


def spell_pitch_class(pitch_class: int, key: int, preferred: int | None = None) -> int:
    """調号 key での音名の綴り (変化記号)。調の音ならその綴り、そうでなければ preferred (使えるとき)、
    それもなければ調の中心に近い綴り (同じ近さなら #、短調の導音)"""
    candidates = _spellings(pitch_class, key)
    diatonic = [a for a in candidates if _diatonic(_position(pitch_class, a), key)]
    if diatonic:
        return diatonic[0]
    if preferred in candidates:
        return preferred
    return min(candidates, key=lambda a: (abs(_position(pitch_class, a) - key - 2), -a))


def _in_key(pitch_class: int, key: int) -> bool:
    return any((position := _position(pitch_class, a)) is not None and _diatonic(position, key) for a in (-1, 0, 1))


def chromatic_spelling(measures: list[Measure]) -> int:
    """調の外の音 (1 音だけの塊) が、同じ段・声部の次の塊 (1 音) へ半音で進むとき、進む向きで綴り直し、
    綴りを変えた音の数を返す。タイでつながる前の音も同じ綴りにそろえる"""
    lines: dict[tuple[int, int], list] = defaultdict(list)
    offset = Fraction(0)
    for m in measures:
        for g in m.groups:
            if g.duration.grace is None:
                lines[(g.staff, g.voice)].append((offset + g.onset, g, m.key))
        offset += m.length
    changed = 0
    for line in lines.values():
        line.sort(key=lambda item: item[0])
        for (_, g, key), (_, h, _) in pairwise(line):
            if len(g.notes) != 1 or len(h.notes) != 1:
                continue
            note, target = g.notes[0], h.notes[0]
            direction = target.pitch - note.pitch
            if abs(direction) != 1 or _in_key(note.pitch % 12, key):
                continue
            step, octave = spell(target.pitch, target.alter)
            index = STEPS.index(step) - direction  # 上がるなら 1 つ下の幹音、下がるなら 1 つ上の幹音
            natural = 12 * (octave + index // 7 + 1) + STEP_PITCH_CLASS[STEPS[index % 7]]
            alter = note.pitch - natural
            # 上がる音は # と ## (E#・F##) まで、下がる音は黒鍵の b だけ (A-Ab を Bbb-Ab、E-Eb を Fb-Eb とは書かない)
            allowed = (
                alter in (1, 2) if direction > 0 else alter == -1 and (note.pitch % 12) not in STEP_PITCH_CLASS.values()
            )
            if allowed and alter != note.alter:
                note.alter = alter
                changed += 1
        for (_, g, _), (_, h, _) in reversed(list(pairwise(line))):
            for note in g.notes:
                tied = next((n for n in h.notes if n.pitch == note.pitch), None) if note.tie else None
                if tied is not None:
                    note.alter = tied.alter
    return changed


def _choose_key(counts: Counter, model_key: int) -> int:
    if abs(model_key) == 7:  # 嬰ハ長調・変ハ長調 (7 つ) は、同じ音の 5 つの調号にする
        return model_key - 12 if model_key > 0 else model_key + 12
    total = sum(counts.values())

    def fit(key: int) -> int:
        return sum(c for pc, c in counts.items() if any(_diatonic(_position(pc, a), key) for a in _spellings(pc, key)))

    # C# 長調と Db 長調のように同じ音の調号は、変化記号の少ないほう (同じ数なら元の調号に近いほう)
    best = max(range(-6, 7), key=lambda k: (fit(k), -abs(k), -abs(k - model_key)))
    if fit(best) - fit(model_key) < KEY_MARGIN * total:
        return model_key
    return best


def _respell(measures: list[Measure], stats: TidyStats) -> None:
    for model_key, run in groupby(measures, key=lambda m: m.key):
        run = list(run)
        notes = [n for m in run for g in m.groups for n in g.notes]
        counts = Counter(n.pitch % 12 for n in notes)
        key = _choose_key(counts, model_key)
        if key != model_key:
            stats.keys += 1
            for m in run:
                m.key = key
        votes = Counter((n.pitch % 12, n.alter) for n in notes)
        spelling = {}
        for pc in counts:
            # 調号を変えていなければ、モデルの綴りの多いほうを使う
            voted = [a for a in _spellings(pc, key) if votes[(pc, a)]]
            preferred = max(voted, key=lambda a: votes[(pc, a)]) if voted and key == model_key else None
            spelling[pc] = spell_pitch_class(pc, key, preferred)
        for n in notes:
            if n.alter != spelling[n.pitch % 12]:
                n.alter = spelling[n.pitch % 12]
                stats.spellings += 1


def tidy_score(measures: list[Measure]) -> TidyStats:
    """measures をその場で直し、直した数を返す"""
    stats = TidyStats()
    if not measures:
        return stats
    stats.rests = _drop_leading_rests(measures)
    stats.ties = _drop_dangling_ties(measures)
    stats.pedals = _fix_pedals(measures)
    stats.dynamics = _drop_repeated_dynamics(measures)
    _respell(measures, stats)
    stats.chromatic = chromatic_spelling(measures)
    return stats
