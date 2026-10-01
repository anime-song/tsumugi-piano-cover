"""カバーモデルの推論用の重みの読み書き (形式は piano_ar.hub と同じ。config.json + model.safetensors)。

    model = CoverModel.from_pretrained()                                   # Hugging Face の公開した重み
    model = CoverModel.from_pretrained("pretrained", subfolder="piano_cover")  # piano_cover.export の出力
    model = load_cover("checkpoints/piano_cover_v4/best.pt")              # 学習のチェックポイント

どれも model.tokenizer と model.context_patches (生成で見る文脈の長さ) が付いた評価モードのモデルを返す。
"""

from __future__ import annotations

from pathlib import Path

import torch

from piano_ar.config import ModelConfig, TokenizerConfig
from piano_ar.hub import HF_REPO, load_weights, read_config, resolve_pretrained, save_pretrained
from piano_ar.tokenizer import PianoTokenizer
from piano_ar.train import context_patches

from .config import CoverConfig
from .model import CoverModel

SUBFOLDER = "piano_cover"


def load_pretrained_cover(
    name_or_path: str | Path = HF_REPO,
    *,
    subfolder: str | None = SUBFOLDER,
    revision: str | None = None,
    device: str | torch.device = "cpu",
) -> CoverModel:
    path = resolve_pretrained(name_or_path, subfolder, revision)
    config = read_config(path, "piano_cover")
    tokenizer = PianoTokenizer(TokenizerConfig(**config["tokenizer_config"]))
    model = CoverModel(
        ModelConfig.from_dict(config["model_config"]),
        CoverConfig.from_dict(config["cover_config"]),
        tokenizer,
        config["source_vocab_size"],
    )
    load_weights(model, path)
    model.context_patches = int(config["context_patches"])
    return model.to(device).eval()


def load_cover_checkpoint(
    checkpoint_path: str | Path, planner: str | Path | None = None, device: str | torch.device = "cpu"
) -> CoverModel:
    """学習のチェックポイント (.pt) を読む。planner を渡すと Planner だけを piano_cover.train_planner の出力に差し替える"""
    # optimizer の状態 (モデルの 2 倍の大きさ) も入っているので、mmap で必要な分だけ読む。
    # 学習中に上書きされるファイルを開いたままにすると学習側の保存が失敗するので、読んだらすぐに手放す
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    tokenizer = PianoTokenizer(TokenizerConfig(**checkpoint["tokenizer_config"]))
    model = CoverModel(
        ModelConfig.from_dict(checkpoint["model_config"]),
        CoverConfig.from_dict(checkpoint["cover_config"]),
        tokenizer,
        checkpoint["source_vocab_size"],
    )
    model.load_state_dict(checkpoint["model"])
    model.context_patches = context_patches(checkpoint["args"], tokenizer.config.patch_seconds)
    del checkpoint
    if planner is not None:
        if model.planner is None:
            raise ValueError(f"{checkpoint_path} には Planner がないので差し替えられない")
        model.planner.load_state_dict(torch.load(planner, map_location="cpu", weights_only=False)["planner"])
    return model.to(device).eval()


def load_cover(
    name_or_path: str | Path,
    *,
    planner: str | Path | None = None,
    device: str | torch.device = "cpu",
    subfolder: str | None = SUBFOLDER,
) -> CoverModel:
    """.pt なら学習のチェックポイント、それ以外は推論用のフォルダか Hugging Face の repository"""
    if str(name_or_path).endswith(".pt"):
        return load_cover_checkpoint(name_or_path, planner, device)
    if planner is not None:
        raise ValueError(
            "Planner の差し替えは学習のチェックポイント (.pt) のときだけ (公開用の重みには export で入れる)"
        )
    return load_pretrained_cover(name_or_path, subfolder=subfolder, device=device)


def export_cover(model: CoverModel, out_dir: str | Path) -> Path:
    config = {
        "model_type": "piano_cover",
        "model_config": vars(model.decoder.config),
        "cover_config": vars(model.config),
        "tokenizer_config": vars(model.tokenizer.config),
        "source_vocab_size": model.source_embedding.num_embeddings,
        "context_patches": model.context_patches,
    }
    return save_pretrained(model, out_dir, config)
