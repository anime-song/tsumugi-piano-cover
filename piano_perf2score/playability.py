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

from piano_score.musicxml import _tie_stops
from piano_score.score import Group, Measure, Note

from .tidy import _drop_dangling_ties

MAX_SPAN = 14  # 片手で届く広さ (半音、10 度の長 10 度まで)
MAX_NOTES = 5


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
