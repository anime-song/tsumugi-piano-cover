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

    テンポは小節ごとに 1 つ。小節の頭は SheetSage の小節の頭にちょうど乗り、小節の中の拍は等間隔になる。拍ごとにテンポを
    変えると、拍の推定の 10 ms ほどの誤差がそのまま BPM の上下に見える (186 bpm で ±6 bpm) ので、小節の中では均す
    (SheetSage の拍との差は 10〜20 ms)。MIDI は 0 秒が小節の頭なので、最初の小節の頭より前は、最初の拍までを最初の拍の
    間隔くらいの拍で埋め、弱起の拍と合わせて最初の小節と同じ拍子の小節に並べる (余りは先頭の短い小節にする)。
    最後の小節は、その中の拍の間隔 (足りなければ 1 つ前の小節) のテンポ"""
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

    # MIDI の小節 (頭の秒, 拍の数, 拍子の分母)。最初の小節の頭より前の拍を、先頭の短い小節 + 最初の小節と同じ拍子の小節に
    pre = fill + first
    head = pre % numerator
    measures = [(0.0, head, unit)] if head else []
    measures += [(float(steps[position]), numerator, unit) for position in range(head, pre, numerator)]
    for k, bar in enumerate(bars):
        end = starts[k + 1] if k + 1 < len(bars) else np.inf
        count = (
            int(((beats >= starts[k] - 1e-6) & (beats < end - 1e-6)).sum()) if k + 1 < len(bars) else int(bar["beats"])
        )
        measures.append((float(starts[k]), max(1, count), int(bar["unit"])))
    # 先頭は 0 秒から (最初の拍が 0 秒の 1 ms 以内のときだけずれる。それより前を MIDI の既定の 120 bpm にしない)
    measures[0] = (0.0, *measures[0][1:])

    tempos, signatures = [], []
    for i, (time, count, denominator) in enumerate(measures):
        if i + 1 < len(measures):
            length = (measures[i + 1][0] - time) / count  # 拍子の分母の音符 1 つの秒
        else:
            inside = beats[beats >= time - 1e-6]
            if len(inside) >= 2:
                length = float(np.diff(inside).mean())
            elif i > 0:
                length = (time - measures[i - 1][0]) / measures[i - 1][1]
            else:
                length = interval
        tempos.append((time, (4 / denominator) * 60.0 / max(length, 1e-3)))
        if i == 0 or (count, denominator) != measures[i - 1][1:]:
            signatures.append((time, count, denominator))
    return tempos, signatures


def set_sheetsage_beats(score, annotation: dict) -> bool:
    """秒の Score のテンポと拍子を SheetSage2-Refined の拍にする (音などの秒の位置は変えない)。拍が足りなければ何もしない"""
    from symusic import Tempo, TimeSignature

    tempos, signatures = sheetsage_beat_map(annotation)
    if not tempos:
        return False
    score.tempos.clear()
    for time, qpm in tempos:
        score.tempos.append(Tempo(time, qpm=qpm, ttype="second"))
    score.time_signatures.clear()
    for time, numerator, denominator in signatures:
        score.time_signatures.append(TimeSignature(time, numerator, denominator, ttype="second"))
    return True


def apply_sheetsage_beats(path: str | Path, annotation: dict) -> bool:
    """MIDI (tsumugi の原曲) のテンポと拍子を SheetSage2-Refined の拍にする (音などの秒の位置は変えない)。
    書き出しは一時ファイルに書いてから置き換える。拍が足りなければ何もしない"""
    from symusic import Score

    score = Score(str(path)).to("second")
    if not set_sheetsage_beats(score, annotation):
        return False
    tmp = Path(path).with_suffix(".mid.tmp")
    score.dump_midi(str(tmp))
    os.replace(tmp, path)
    return True
