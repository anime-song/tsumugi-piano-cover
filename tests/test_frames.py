"""原曲の音声の特徴の行 (source_features の frames と collate_cover) の確かめ。"""

from __future__ import annotations

import numpy as np
import torch

from piano_ar.beat_tokenizer import BeatGrid
from piano_ar.tokenizer import PianoTokenizer
from piano_cover.data import collate_cover, source_tensors
from piano_cover.source import ROW_ONSET, TYPE_NOTE, SourceVocab, source_features


def _features(frames: np.ndarray, max_frame_rows: int = 128) -> dict[str, np.ndarray]:
    # 1 拍 0.5 秒 (50 フレーム) の 4 拍子を 2 小節。行は 1 小節目の頭の音 1 つ
    grid = BeatGrid(np.arange(9, dtype=np.float64) * 50, np.array([0, 4, 8]))
    rows = np.zeros((1, 6), dtype=np.int64)
    rows[0, ROW_ONSET], rows[0, 1], rows[0, 2] = 0, TYPE_NOTE, 60
    tokenizer = PianoTokenizer()
    vocab = SourceVocab(tokenizer.config)
    return source_features(rows, 400, tokenizer, vocab, 256, grid=grid, frames=frames, max_frame_rows=max_frame_rows)


def test_frame_rows_follow_the_other_rows_in_each_bar():
    frames = np.arange(100 * 4, dtype=np.float32).reshape(100, 4)  # 4 秒 = 2 小節
    out = _features(frames)
    index = out["frame_index"]
    # 1 小節 2 秒 = 50 フレーム。中央の時刻で丸めるので、小節の最後のフレームは次の小節の頭になり、
    # 曲の最後のフレームは曲の外になる。音の行は 1 小節目の先頭
    assert (index[0] >= 0).sum() == 49 and (index[1] >= 0).sum() == 50
    assert index[0, 0] == -1 and index[0, 1] == 0
    # 特徴の行の位置は、フレームの中央の時刻を拍の刻みにしたもの (12 刻み / 50 フレーム)
    assert out["onset"][0, 1] == round(0.5 * 4 / 50 * 12)
    np.testing.assert_array_equal(out["frames"][index[1, index[1] >= 0]], frames[49:99].astype(np.float16))


def test_frame_rows_are_thinned_evenly():
    out = _features(np.zeros((100, 4), dtype=np.float32), max_frame_rows=20)
    onset = out["onset"][1][out["frame_index"][1] >= 0]
    assert len(onset) == 20
    assert np.all(np.diff(onset) > 0)


def test_collate_offsets_frame_index():
    a = source_tensors(_features(np.ones((100, 4), dtype=np.float32)))
    b = source_tensors(_features(np.full((100, 4), 2, dtype=np.float32)))
    out = collate_cover([_decoder_part() | a, _decoder_part() | b])
    index = out["src_frame_index"]
    assert len(out["src_frames"]) == 2 * 99
    assert torch.all(out["src_frames"][index[1][index[1] >= 0].long()] == 2)


def _decoder_part() -> dict[str, torch.Tensor]:
    return {"tokens": torch.zeros(1, 4, dtype=torch.long), "patch_valid": torch.ones(1, dtype=torch.bool)}
