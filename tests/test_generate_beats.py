"""拍ベースの生成で、原曲の小節開始位置を学習時と同じ刻みで渡す。"""

from __future__ import annotations

import numpy as np

from piano_ar.beat_tokenizer import BeatGrid, BeatTokenizer
from piano_ar.config import ModelConfig
from piano_ar.tokenizer import EOP
from piano_cover.config import CoverConfig
from piano_cover.generate import CoverParams, PreparedSource, generate_covers
from piano_cover.model import CoverModel


def test_generate_uses_actual_source_bar_starts(monkeypatch):
    tokenizer = BeatTokenizer()
    model = CoverModel(
        ModelConfig(dim=32, heads=4, summary_layers=1, global_layers=1, local_layers=1, dropout=0),
        CoverConfig(source_patch_layers=1, source_song_layers=1, source_latents=1, onset_head_dim=0, planner_dim=0),
        tokenizer,
        source_vocab_size=32,
    ).eval()
    model.context_patches = 8
    grid = BeatGrid(np.arange(9, dtype=np.float64) * 50, np.array([0, 4, 8]))
    features = {
        "features": np.array([[[1, 2, 3, 4, 5]], [[1, 2, 3, 4, 5]]], dtype=np.int16),
        "onset": np.array([[0], [48]], dtype=np.int32),
        "valid": np.ones((2, 1), dtype=bool),
        "patch_start": np.array([0, 48], dtype=np.int64),
    }
    source = PreparedSource(np.zeros((0, 5), dtype=np.int32), 0, features, grid, grid.bar_beats, np.ones(2))
    seen = []

    def fake_generate(*args, **kwargs):
        condition = kwargs["condition"].bind([True])
        seen.extend(int(condition.start(patch).item()) for patch in range(2))
        return [[[EOP], [EOP]]]

    monkeypatch.setattr(model.decoder, "generate", fake_generate)
    generate_covers(model, source, CoverParams(arrangement=False))
    assert seen == [0, 48]
