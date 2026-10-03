"""原曲 MIDI のメタデータを、生成したカバーの MIDI に写す。

カバーは原曲の時間軸の上で作るので、テンポ・拍子・調・マーカー (tsumugi のコード) は
そのまま使える。写さないと、開いた側は MIDI の既定 (120 bpm・4/4) で解釈してしまい、
速さも小節の頭も合わない。秒で持っているスコアを書き出すときは、このテンポ図が
秒 -> tick の変換にも使われるので、位置そのものがずれる。

歌詞 (lyrics) はトラックごとの情報で、tsumugi は音符のない専用のトラックに置く。カバーは
ピアノ 1 本なので、原曲の全トラックの歌詞をまとめてそのピアノトラックに置く。

notes / controls はカバーのもの、メタデータと歌詞は原曲のものを、という分担。
"""

from __future__ import annotations

from pathlib import Path

# 写す対象 (Score の属性名)
META_BLOCKS = ("tempos", "time_signatures", "key_signatures", "markers")


def lyrics_of(score) -> list[tuple[float, str]]:
    """全トラックの歌詞を時刻順にまとめたもの"""
    items = [(float(text.time), str(text.text)) for track in score.tracks for text in track.lyrics]
    items.sort(key=lambda pair: pair[0])
    return items


def _snapshot(score) -> tuple:
    """比較用に、メタデータを丸めた数の組にする (秒は 0.1 ms まで)"""
    def at(value: float) -> float:
        return round(float(value), 4)

    return (
        tuple((at(t.time), round(float(t.qpm), 4)) for t in score.tempos),
        tuple((at(t.time), int(t.numerator), int(t.denominator)) for t in score.time_signatures),
        tuple((at(k.time), int(k.key), int(k.tonality)) for k in score.key_signatures),
        tuple((at(m.time), str(m.text)) for m in score.markers),
        tuple((at(time), text) for time, text in lyrics_of(score)),
    )


def copy_metadata(source: str | Path, target: str | Path, *, force: bool = False) -> bool:
    """source のテンポ・拍子・調・マーカー・歌詞を target に写して書き戻す。

    すでに同じものが入っていれば何もしない (何度呼んでも結果は同じ)。
    書き換えたら True、そのままなら False を返す。
    """
    from symusic import Score

    src = Score(str(source)).to("second")
    out = Score(str(target)).to("second")
    if not force and _snapshot(src) == _snapshot(out):
        return False

    for name in META_BLOCKS:
        items = [item.copy() for item in getattr(src, name)]
        block = getattr(out, name)
        block.clear()
        for item in items:
            block.append(item)

    # 歌詞はトラックごとの情報なので、まとめてカバーのピアノトラックに置く
    from symusic import TextMeta

    lyrics = lyrics_of(src)
    if lyrics_of(out) != lyrics:
        for track in out.tracks:
            track.lyrics.clear()
        if lyrics and out.tracks:
            for time, text in lyrics:
                out.tracks[0].lyrics.append(TextMeta(time, text, "second"))
    out.dump_midi(str(target))
    return True
