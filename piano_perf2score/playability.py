"""楽譜を両手で弾ける形に直す後処理 (拍の上の演奏 -> 楽譜で、V2 のカバーを楽譜にしたもの向け)。

V2 のカバーには、片手では届かない広い和音 (10 度を超える) や 6 音以上の和音が、人の楽譜の 4 倍ほどある
(生成 2.8%、学習の楽譜 0.7%)。カバーの採譜の癖をカバーモデルが学んだもので、楽譜にもそのまま出る。
そこで段ごとに、同時に打鍵する和音を次の順で直す。
    1. 和音の中のオクターブの重複 (同じ音名) を、外側 (一番上と一番下) を残して内側から消す
    2. 広い和音の端の音 (上の段なら一番下、下の段なら一番上) を、もう一方の段の同じ位置・同じ音価の和音に移す。
       そういう和音がなければ、もう一方の段の声部 1 がその間空いているときだけ新しい塊として置く
       (移した先も片手で弾けるときだけ)
    3. それでも無理なら、上の段は下から、下の段は上から音を消す (上の段の一番上 = メロディと、下の段の一番下 = 低音は残す)
タイでつながる音 (タイの始まり・続き) は、つながりを壊さないよう動かさない。
両手 (と同じ段の別の声部) で同じ鍵盤を同時に鳴らす音は、旋律の一番上の音と低音を残して消す (remove_unisons)。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction

from piano_score.musicxml import _tie_stops
from piano_score.score import NOTE_TYPES, Duration, Group, Measure, Note

from .tidy import _drop_dangling_ties

MAX_SPAN = 14  # 片手で届く広さ (半音、10 度の長 10 度まで)
MAX_NOTES = 5
# upper_to_right: 左手の和音のこの音 (D3) 以上を右手に渡す候補にする。左手がこの時間 (4 分音符単位) 以内に
# 1 オクターブ (HAND_SPAN) を超えて跳ぶ (低音と和音を続けて弾く) ときだけ渡す
RIGHT_SPLIT = 50
LEAP_WINDOW = Fraction(1, 2)
HAND_SPAN = 12
HELD_BEFORE = Fraction(1)  # 右手がこれ以上前から伸ばしている音は、渡す和音の手前で切ってよい


@dataclass
class PlayableStats:
    chords: int = 0  # 直す前に片手で無理だった和音
    deduplicated: int = 0
    moved: int = 0
    dropped: int = 0


def _voice_free(measure: Measure, staff: int, voice: int, start, end) -> bool:
    return not any(
        g.staff == staff and g.voice == voice and g.duration.grace is None and g.onset < end and start < g.end
        for g in measure.groups
    )


def _playable(pitches: list[int]) -> bool:
    return not pitches or (max(pitches) - min(pitches) <= MAX_SPAN and len(pitches) <= MAX_NOTES)


def make_playable(measures: list[Measure]) -> PlayableStats:
    """measures をその場で直し、直した数を返す"""
    stats = PlayableStats()
    tie_stops = _tie_stops(measures)
    for mi, measure in enumerate(measures):
        groups = [g for g in measure.groups if g.duration.grace is None]
        for onset in sorted({g.onset for g in groups}):
            for staff in (1, 2):
                here = [g for g in groups if g.onset == onset and g.staff == staff]
                pitches = [n.pitch for g in here for n in g.notes]
                if _playable(pitches):
                    continue
                stats.chords += 1

                def fixed(g, n, mi=mi) -> bool:
                    return n.tie or (mi, id(g), n.pitch) in tie_stops

                # 1. オクターブの重複を内側から消す
                for g in here:
                    while not _playable([n.pitch for h in here for n in h.notes]):
                        all_pitches = sorted(n.pitch for h in here for n in h.notes)
                        outer = {all_pitches[0], all_pitches[-1]}
                        classes = [n.pitch % 12 for h in here for n in h.notes]
                        victim = next(
                            (n for n in g.notes if n.pitch not in outer and classes.count(n.pitch % 12) > 1
                             and not fixed(g, n) and len(g.notes) > 1),
                            None,
                        )  # fmt: skip
                        if victim is None:
                            break
                        g.notes.remove(victim)
                        stats.deduplicated += 1
                # 2. 端の音をもう一方の段へ移す / 3. 消す
                while not _playable([n.pitch for g in here for n in g.notes]):
                    candidates = [(g, n) for g in here for n in g.notes if not fixed(g, n) and len(g.notes) > 1]
                    if not candidates:
                        break
                    # 上の段は一番下、下の段は一番上の音
                    g, n = (min if staff == 1 else max)(candidates, key=lambda c: c[1].pitch)
                    target = next(
                        (h for h in groups if h.onset == onset and h.staff == 3 - staff and h.duration == g.duration),
                        None,
                    )
                    other = [m.pitch for h in groups if h.onset == onset and h.staff == 3 - staff for m in h.notes]
                    g.notes.remove(n)
                    if not _playable([*other, n.pitch]):
                        stats.dropped += 1
                    elif target is not None:
                        target.notes.append(n)
                        target.notes.sort(key=lambda m: m.pitch)
                        stats.moved += 1
                    elif _voice_free(measure, 3 - staff, 1, onset, g.end):
                        # 同じ音価の和音がなく、もう一方の段の声部 1 がその間空いていれば、新しい塊を置く
                        moved = replace(g, staff=3 - staff, voice=1, notes=[n], articulations=(), cross=False)
                        moved.slur_start = moved.slur_stop = 0
                        measure.groups.append(moved)
                        groups.append(moved)
                        stats.moved += 1
                    else:
                        stats.dropped += 1
    return stats


def _keep_rank(g: Group, note: Note) -> int:
    """同じ鍵盤が重なったときに残す順位: 旋律の一番上 > 低音 (下の段の声部 1 の一番下) > 上の段 > 下の段"""
    if (g.staff, g.voice) == (1, 1) and note.pitch == max(n.pitch for n in g.notes):
        return 3
    if (g.staff, g.voice) == (2, 1) and note.pitch == min(n.pitch for n in g.notes):
        return 2
    return 1 if g.staff == 1 else 0


def remove_unisons(measures: list[Measure]) -> int:
    """別の声部 (両手や上の段の内声) で同じ鍵盤が同時に鳴る音を、残す順位の低いほうから消す。消してつながらなくなった
    タイも消す。消した音の数を返す"""
    removed = 0
    for m in measures:
        groups = [g for g in m.groups if g.duration.grace is None]
        for g in groups:
            for note in list(g.notes):
                for h in groups:
                    if h is g or (h.staff, h.voice) == (g.staff, g.voice) or not (h.onset < g.end and g.onset < h.end):
                        continue
                    other = next((n for n in h.notes if n.pitch == note.pitch), None)
                    if other is None or note not in g.notes:
                        continue
                    loser, owner = (note, g) if _keep_rank(g, note) < _keep_rank(h, other) else (other, h)
                    owner.notes.remove(loser)
                    removed += 1
        m.groups = [g for g in m.groups if g.notes]
    _drop_dangling_ties(measures)
    return removed


def _writable(length: Fraction) -> Duration | None:
    """長さ length (4 分音符単位) を 1 つの音符 (付点 1 つまで) で書ければその音価"""
    for kind, quarters in NOTE_TYPES.items():
        for dots in (0, 1):
            if quarters * (2 - Fraction(1, 2**dots)) == length:
                return Duration(kind, dots)
    return None


def _upper(g: Group) -> list[Note]:
    return [n for n in g.notes if n.pitch >= RIGHT_SPLIT]


def _movable(measure: Measure, g: Group, fixed: set, leap: bool = True) -> bool:
    """左手の塊 g の D3 以上の和音を右手に渡せる形か (タイ・スラーを壊さない)。leap なら、左手が跳ぶときだけ"""
    upper = _upper(g)
    if len(upper) < 2 or g.cross or any(n.tie or (id(g), n.pitch) in fixed for n in upper):
        return False
    if len(upper) == len(g.notes) and (g.slur_start or g.slur_stop):
        return False  # 塊ごと移すとスラーの端が段をまたぐ
    if not leap:
        return True
    top = max(n.pitch for n in upper)
    lows = [n.pitch for n in g.notes if n.pitch < RIGHT_SPLIT]
    lows += [
        n.pitch for h in measure.groups
        if h.staff == 2 and h is not g and h.duration.grace is None and abs(h.onset - g.onset) <= LEAP_WINDOW
        for n in h.notes
    ]  # fmt: skip
    return any(top - low > HAND_SPAN for low in lows)


def _to_right(measure: Measure, g: Group, cut_held: bool) -> bool:
    """左手の塊 g の D3 以上の音を、右手が弾けるなら右手へ移す。cut_held なら、右手が HELD_BEFORE 以上前から伸ばして
    いる音を g の手前で切ってよい"""
    upper = _upper(g)
    right = [
        h for h in measure.groups
        if h.staff == 1 and not h.cross and h.duration.grace is None and h.onset < g.end and g.onset < h.end
    ]  # fmt: skip
    cut = []
    if cut_held and right and all(h.onset <= g.onset - HELD_BEFORE for h in right):
        cut = [h for h in right if not any(n.tie for n in h.notes) and _writable(g.onset - h.onset)]
        right, cut = ([], cut) if len(cut) == len(right) else (right, [])
    pitches = [n.pitch for h in right for n in h.notes]
    ours = [n.pitch for n in upper]
    if max(pitches + ours) - min(pitches + ours) > HAND_SPAN or len(pitches + ours) > MAX_NOTES:
        return False
    if pitches and max(ours) >= min(pitches):
        return False
    free = [v for v in (1, 2) if not any(h.voice == v for h in right)]
    target = next((h for h in right if h.onset == g.onset and h.duration == g.duration), None)
    if target is None and not free:
        return False
    for h in cut:
        h.duration = _writable(g.onset - h.onset)
    for n in upper:
        g.notes.remove(n)
    if target is not None:
        target.notes = sorted(target.notes + upper, key=lambda n: n.pitch)
    else:
        measure.groups.append(Group(g.onset, 1, free[0], g.duration, upper, g.articulations))
    return True


def upper_to_right(measures: list[Measure]) -> int:
    """左手が低音と中音域の和音を続けて弾く (LEAP_WINDOW 以内に 1 オクターブを超えて跳ぶ、または 1 つの和音の中で
    1 オクターブを超える) とき、和音の D3 以上の音を右手が弾けるなら右手へ移す。V2 のカバーの楽譜は、右手が
    空いていても低音の上の和音を左手に書くことが多い (Shiny Day の 11・13・24 小節目)。
    右手が弾けるのは、その間に右手で鳴っている音と合わせて 1 オクターブ以内・MAX_NOTES 音以内で、右手の音より
    下に収まるとき。同じ位置・同じ音価の右手の和音があればそこへ足し、なければ右手の空いている声部 (1 か 2) に置く。
    同じ小節で右手へ移した和音と同じ音の左手の和音は、右手が HELD_BEFORE 以上前から伸ばしている音をその和音の
    手前で切ってでも右手へ移す (同じ和音を左右に書き分けない)。移した音の数を返す"""
    tie_stops = _tie_stops(measures)
    moved = 0
    for mi, measure in enumerate(measures):
        fixed = {(group_id, pitch) for m, group_id, pitch in tie_stops if m == mi}
        left = sorted((g for g in measure.groups if g.staff == 2 and g.duration.grace is None), key=lambda g: g.onset)
        moved_sets = set()
        for g in left:
            pitches = frozenset(n.pitch for n in _upper(g))
            if _movable(measure, g, fixed) and _to_right(measure, g, cut_held=False):
                moved += len(pitches)
                moved_sets.add(pitches)
        for g in left:
            pitches = frozenset(n.pitch for n in _upper(g))
            if pitches in moved_sets and _movable(measure, g, fixed, leap=False) and _to_right(measure, g, True):
                moved += len(pitches)
        measure.groups = [g for g in measure.groups if g.notes]
    return moved
