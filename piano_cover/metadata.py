"""原曲 MIDI のメタデータを、生成したカバーの MIDI に写す。

カバーは原曲の時間軸の上で作るので、テンポ・拍子・調・マーカー (tsumugi のコード) は
そのまま使える。写さないと、開いた側は MIDI の既定 (120 bpm・4/4) で解釈してしまい、
速さも小節の頭も合わない。秒で持っているスコアを書き出すときは、このテンポ図が
秒 -> tick の変換にも使われるので、位置そのものがずれる。

歌詞 (lyrics) はトラックごとの情報で、tsumugi は音符のない専用のトラックに置く。カバーは
ピアノ 1 本なので、原曲の全トラックの歌詞をまとめてそのピアノトラックに置く。

notes / controls はカバーのもの、メタデータと歌詞は原曲のものを、という分担。

書き出すのは 1 回だけにする。書いてから写すと秒 -> tick の丸めが 2 回起きるので、
`score_for_cover` でメタデータを入れた Score を作ってから `dump_midi` する。
`copy_metadata` は、メタデータなしで書かれた古いテイクを 1 回だけ直すためのもの。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# 写す対象 (Score の属性名)
META_BLOCKS = ("tempos", "time_signatures", "key_signatures", "markers")


def lyrics_of(score) -> list[tuple[float, str]]:
    """全トラックの歌詞を時刻順にまとめたもの"""
    items = [(float(text.time), str(text.text)) for track in score.tracks for text in track.lyrics]
    items.sort(key=lambda pair: pair[0])
    return items


def apply_metadata(score, source) -> None:
    """source のテンポ・拍子・調・マーカー・歌詞を score に写す (どちらも秒の Score)"""
    from symusic import TextMeta

    for name in META_BLOCKS:
        items = [item.copy() for item in getattr(source, name)]
        block = getattr(score, name)
        block.clear()
        for item in items:
            block.append(item)

    # 歌詞はトラックごとの情報なので、まとめてカバーのピアノトラックに置く
    for track in score.tracks:
        track.lyrics.clear()
    lyrics = lyrics_of(source)
    if lyrics and score.tracks:
        for time, text in lyrics:
            score.tracks[0].lyrics.append(TextMeta(time, text, "second"))


def score_for_cover(tokenizer, events, source):
    """原曲のメタデータを入れた、カバーの Score を作る (書き出しは呼ぶ側で 1 回だけ)"""
    score = tokenizer.events_to_score(events)
    if source is not None:
        apply_metadata(score, source)
    return score


def copy_metadata(source: str | Path, target: str | Path) -> bool:
    """メタデータなしで書かれた古いテイクの MIDI を、原曲のメタデータつきに書き直す。

    notes / controls は target のものを使い、書き出しは一時ファイルに書いてから置き換える
    (読んでいる側や同時のダウンロードに、書きかけのファイルを見せない)。
    秒 -> tick の丸めは 1 回だけ。何度も呼ぶとその分ずれるので、呼ぶ側が「1 回だけ」を守ること。
    """
    from symusic import Score

    src = Score(str(source)).to("second")
    out = Score(str(target)).to("second")
    apply_metadata(out, src)
    tmp = Path(target).with_suffix(".mid.tmp")
    out.dump_midi(str(tmp))
    os.replace(tmp, target)
    return True


def sheetsage_beat_map(annotation: dict) -> tuple[list[tuple[float, float]], list[tuple[float, int, int]]]:
    """SheetSage2-Refined の拍と小節 (piano_ar.sheetsage の JSON) から、MIDI のテンポ (秒, qpm) と拍子 (秒, 分子, 分母) を作る。

    拍ごとにテンポを変えて、SheetSage の各拍が MIDI の拍 (拍子の分母の音符) にちょうど乗り、小節の頭が MIDI の小節の頭に
    乗るようにする (piano_cover.source._beat_rows で読むと SheetSage の拍に戻る)。MIDI は 0 秒が小節の頭なので、
    最初の小節の頭より前は、最初の拍までを最初の拍の間隔くらいの拍で埋め、弱起の拍と合わせて最初の小節と同じ拍子の
    小節に並べる (余りは先頭の短い小節にする)。最後の拍より後は最後のテンポのまま"""
    beats = np.asarray(annotation.get("beats", []), dtype=np.float64)
    bars = annotation.get("bars") or []
    if len(beats) < 2 or not bars:
        return [], []
    starts = np.array([float(b["start"]) for b in bars])
    first = int(np.searchsorted(beats, starts[0] - 1e-6))  # 最初の小節の頭の拍
    numerator, unit = max(1, int(bars[0]["beats"])), int(bars[0]["unit"])
    interval = float(beats[1] - beats[0])
    fill = max(1, round(beats[0] / interval)) if beats[0] > 1e-3 else 0
    steps = list(np.linspace(0.0, beats[0], fill, endpoint=False)) + list(beats)
    # 拍ごとの、拍子の分母の音符の長さ (四分音符を 1 とする)
    bar_of = np.clip(np.searchsorted(starts, beats + 1e-6, side="right") - 1, 0, len(bars) - 1)
    quarters = [4 / unit] * fill + [4 / int(bars[k]["unit"]) for k in bar_of]
    tempos = []
    for i, time in enumerate(steps):
        length = steps[i + 1] - time if i + 1 < len(steps) else time - steps[i - 1]
        tempos.append((float(time), quarters[i] * 60.0 / max(length, 1e-3)))

    signatures = []
    pre = fill + first  # 最初の小節の頭より前の拍の数
    head = pre % numerator
    position = 0
    if head:
        signatures.append((0.0, head, unit))
        position = head
    while position < pre:
        signatures.append((float(steps[position]), numerator, unit))
        position += numerator
    previous = None
    for k, bar in enumerate(bars):
        end = starts[k + 1] if k + 1 < len(bars) else np.inf
        count = (
            int(((beats >= starts[k] - 1e-6) & (beats < end - 1e-6)).sum()) if k + 1 < len(bars) else int(bar["beats"])
        )
        meter = (max(1, count), int(bar["unit"]))
        if meter != previous:
            signatures.append((float(starts[k]), *meter))
        previous = meter
    return tempos, signatures


def apply_sheetsage_beats(path: str | Path, annotation: dict) -> bool:
    """MIDI (tsumugi の原曲) のテンポと拍子を SheetSage2-Refined の拍にする (音などの秒の位置は変えない)。
    書き出しは一時ファイルに書いてから置き換える。拍が足りなければ何もしない"""
    from symusic import Score, Tempo, TimeSignature

    tempos, signatures = sheetsage_beat_map(annotation)
    if not tempos:
        return False
    score = Score(str(path)).to("second")
    score.tempos.clear()
    for time, qpm in tempos:
        score.tempos.append(Tempo(time, qpm=qpm, ttype="second"))
    score.time_signatures.clear()
    for time, numerator, denominator in signatures:
        score.time_signatures.append(TimeSignature(time, numerator, denominator, ttype="second"))
    tmp = Path(path).with_suffix(".mid.tmp")
    score.dump_midi(str(tmp))
    os.replace(tmp, path)
    return True
