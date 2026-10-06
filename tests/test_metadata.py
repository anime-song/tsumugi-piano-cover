"""原曲のメタデータを写すときの丸めを確かめる。

原曲の tpq が 384 のような値だと、書き出してから写す (秒 -> tick が 2 回) やり方では
呼ぶたびに少しずつ音がずれていた。書き出す前にメタデータを入れて 1 回だけ書けば、
ずれは「1 回の丸め」で止まる。
"""

from __future__ import annotations

import numpy as np
import pytest
from symusic import KeySignature, Note, Score, Tempo, TextMeta, TimeSignature, Track

from piano_ar.tokenizer import KIND_NOTE, KIND_PEDAL_ON, PianoTokenizer
from piano_cover.metadata import copy_metadata, lyrics_of, score_for_cover

FRAME_RATE = 100  # 10 ms


def write_source(path, tpq: int = 384):
    """テンポ・拍子・調・マーカー・歌詞つきの原曲を 1 つ作る"""
    score = Score(ttype="tick")
    score.tpq = tpq  # 0.5 系と 0.6 系で引数の名前が違うので、作ってから入れる
    track = Track(name="piano", program=0)
    for beat in range(8):
        track.notes.append(Note(beat * tpq, tpq // 2, 60 + beat, 90))
    score.tracks.append(track)
    score.tempos.append(Tempo(0, 120.0))
    score.tempos.append(Tempo(4 * tpq, 90.0))
    score.time_signatures.append(TimeSignature(0, 3, 4))
    score.key_signatures.append(KeySignature(0, 9, 1))  # A minor
    score.markers.append(TextMeta(0, "Intro"))
    score.markers.append(TextMeta(4 * tpq, "Chorus"))
    track.lyrics.append(TextMeta(0, "la"))
    track.lyrics.append(TextMeta(tpq, "li"))
    score.dump_midi(str(path))
    return score


def cover_events():
    """カバーのイベント (onset フレーム, 種類, pitch, 長さフレーム, velocity)"""
    rows = [
        [0, KIND_NOTE, 60, 50, 90],
        [25, KIND_NOTE, 64, 50, 88],  # 0.25 秒 (丸めが出やすい位置)
        [50, KIND_PEDAL_ON, 0, 0, 0],
        [123, KIND_NOTE, 67, 40, 85],
        [250, KIND_NOTE, 72, 40, 80],
    ]
    return np.asarray(rows, dtype=np.int64)


def test_score_for_cover_keeps_metadata_and_notes(tmp_path):
    source_path = tmp_path / "source.mid"
    write_source(source_path, tpq=384)
    tokenizer = PianoTokenizer()
    events = cover_events()

    source = Score(str(source_path)).to("second")
    out_path = tmp_path / "cover.mid"
    score_for_cover(tokenizer, events, source).dump_midi(str(out_path))

    out = Score(str(out_path)).to("second")
    assert len(out.tempos) == 2
    assert [round(t.qpm) for t in out.tempos] == [120, 90]
    assert [(s.numerator, s.denominator) for s in out.time_signatures] == [(3, 4)]
    assert [(k.key, k.tonality) for k in out.key_signatures] == [(9, 1)]
    assert [m.text for m in out.markers] == ["Intro", "Chorus"]
    assert [text for _, text in lyrics_of(out)] == ["la", "li"]

    notes = sorted((note.time, note.pitch) for track in out.tracks for note in track.notes)
    wanted = sorted(
        (onset / FRAME_RATE, pitch) for onset, kind, pitch, _, _ in events.tolist() if kind == KIND_NOTE
    )
    assert [pitch for _, pitch in notes] == [pitch for _, pitch in wanted]
    for (got, _), (expect, _) in zip(notes, wanted):
        assert abs(got - expect) < 0.002  # 1 回の丸め (tpq 960 で 0.5 ms 程度) に収まっている


def test_copy_metadata_writes_once_and_atomically(tmp_path):
    source_path = tmp_path / "source.mid"
    write_source(source_path, tpq=384)
    tokenizer = PianoTokenizer()
    plain = tmp_path / "cover.mid"
    tokenizer.events_to_midi(cover_events(), plain)  # メタデータなし (既定の 120 bpm) の古いテイク
    before = sorted((n.time, n.pitch) for t in Score(str(plain)).to("second").tracks for n in t.notes)

    assert copy_metadata(source_path, plain) is True

    after_score = Score(str(plain)).to("second")
    assert len(after_score.tempos) == 2
    assert [m.text for m in after_score.markers] == ["Intro", "Chorus"]
    after = sorted((n.time, n.pitch) for t in after_score.tracks for n in t.notes)
    assert [pitch for _, pitch in after] == [pitch for _, pitch in before]
    for (got, _), (expect, _) in zip(after, before):
        assert abs(got - expect) < 0.002  # 1 回の丸めだけ
    assert not list(tmp_path.glob("*.tmp"))  # 一時ファイルを残さない


@pytest.mark.parametrize("tpq", [96, 384, 480, 960])
def test_single_copy_drift_is_bounded(tmp_path, tpq):
    """1 回の書き直しで動く量は、tpq によらず 2 ms 未満に収まる"""
    source_path = tmp_path / "source.mid"
    write_source(source_path, tpq=tpq)
    tokenizer = PianoTokenizer()
    plain = tmp_path / "cover.mid"
    tokenizer.events_to_midi(cover_events(), plain)
    before = sorted((n.time, n.pitch) for t in Score(str(plain)).to("second").tracks for n in t.notes)

    copy_metadata(source_path, plain)

    after = sorted((n.time, n.pitch) for t in Score(str(plain)).to("second").tracks for n in t.notes)
    drift = max(abs(a - b) for (a, _), (b, _) in zip(after, before))
    assert drift < 0.002
