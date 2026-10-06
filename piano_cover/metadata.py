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
