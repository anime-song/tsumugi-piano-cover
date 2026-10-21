"""楽譜化のモデル (演奏 -> 楽譜) の推論用の重みの読み書き (形式は piano_ar.hub と同じ。config.json + model.safetensors)。

    model, tokenizer, banned = load_perf2score(HF_REPO)                                  # Hugging Face の公開した重み
    model, tokenizer, banned = load_perf2score("pretrained_v2")                          # export の出力
    model, tokenizer, banned = load_perf2score("checkpoints/piano_perf2score_beat5_mix/best.pt")  # 学習のチェックポイント

banned は学習で一度も出なかったトークン (生成で選ばせない。piano_score.generate.unseen_tokens)。
公開用のフォルダでは、その元になるトークンの出現数 (token_counts) を config.json に入れる。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from piano_ar.config import ModelConfig
from piano_ar.hub import HF_REPO, load_weights, read_config, resolve_pretrained, save_pretrained
from piano_score.config import ScoreTokenizerConfig
from piano_score.generate import unseen_tokens
from piano_score.tokenizer import ScoreTokenizer

from .model import Perf2ScoreModel, perf2score_config

SUBFOLDER = "piano_perf2score"
MODEL_TYPE = "piano_perf2score"


def _tokenizer(config: dict) -> ScoreTokenizer:
    config = dict(config)
    config["fraction_denominators"] = tuple(config["fraction_denominators"])
    return ScoreTokenizer(ScoreTokenizerConfig(**config))


def _banned(token_counts, tokenizer: ScoreTokenizer) -> torch.Tensor | None:
    return unseen_tokens(np.asarray(token_counts), tokenizer) if token_counts is not None else None


def load_perf2score_checkpoint(
    checkpoint_path: str | Path, device: str | torch.device = "cpu"
) -> tuple[Perf2ScoreModel, ScoreTokenizer, torch.Tensor | None]:
    """学習のチェックポイント (.pt) を読む"""
    # optimizer の状態 (モデルの 2 倍の大きさ) も入っているので、mmap で必要な分だけ読む
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    tokenizer = _tokenizer(checkpoint["tokenizer_config"])
    model = Perf2ScoreModel(
        ModelConfig.from_dict(checkpoint["model_config"]), perf2score_config(checkpoint["perf2score_config"]), tokenizer
    )
    model.load_state_dict(checkpoint["model"])
    banned = _banned(checkpoint.get("token_counts"), tokenizer)
    del checkpoint
    return model.to(device).eval(), tokenizer, banned


def load_pretrained_perf2score(
    name_or_path: str | Path = HF_REPO,
    *,
    subfolder: str | None = SUBFOLDER,
    revision: str | None = None,
    device: str | torch.device = "cpu",
) -> tuple[Perf2ScoreModel, ScoreTokenizer, torch.Tensor | None]:
    path = resolve_pretrained(name_or_path, subfolder, revision)
    config = read_config(path, MODEL_TYPE)
    tokenizer = _tokenizer(config["tokenizer_config"])
    model = Perf2ScoreModel(
        ModelConfig.from_dict(config["model_config"]), perf2score_config(config["perf2score_config"]), tokenizer
    )
    load_weights(model, path)
    return model.to(device).eval(), tokenizer, _banned(config.get("token_counts"), tokenizer)


def load_perf2score(
    name_or_path: str | Path = HF_REPO,
    device: str | torch.device = "cpu",
    subfolder: str | None = SUBFOLDER,
) -> tuple[Perf2ScoreModel, ScoreTokenizer, torch.Tensor | None]:
    """.pt なら学習のチェックポイント、それ以外は推論用のフォルダか Hugging Face の repository"""
    if str(name_or_path).endswith(".pt"):
        return load_perf2score_checkpoint(name_or_path, device)
    return load_pretrained_perf2score(name_or_path, subfolder=subfolder, device=device)


def export_perf2score(checkpoint_path: str | Path, out_dir: str | Path) -> Path:
    """学習のチェックポイントから、推論に要るものだけ (設定・トークンの出現数・重み) を out_dir に書き出す"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    tokenizer = _tokenizer(checkpoint["tokenizer_config"])
    model = Perf2ScoreModel(
        ModelConfig.from_dict(checkpoint["model_config"]), perf2score_config(checkpoint["perf2score_config"]), tokenizer
    )
    model.load_state_dict(checkpoint["model"])
    counts = checkpoint.get("token_counts")
    config = {
        "model_type": MODEL_TYPE,
        "model_config": dict(checkpoint["model_config"]),
        "perf2score_config": dict(checkpoint["perf2score_config"]),
        "tokenizer_config": dict(checkpoint["tokenizer_config"]),
        "token_counts": np.asarray(counts).tolist() if counts is not None else None,
    }
    return save_pretrained(model, out_dir, config)
