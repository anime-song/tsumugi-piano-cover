"""カバーモデルの推論用の重みの読み書き (形式は piano_ar.hub と同じ。config.json + model.safetensors)。

    model = CoverModel.from_pretrained()                                   # Hugging Face の公開した重み
    model = CoverModel.from_pretrained("pretrained", subfolder="piano_cover")  # piano_cover.export の出力
    model = load_cover("checkpoints/piano_cover_v4/best.pt")              # 学習のチェックポイント

どれも model.tokenizer と model.context_patches (生成で見る文脈の長さ) が付いた評価モードのモデルを返す。
"""

from __future__ import annotations

import shutil
from pathlib import Path

import torch

from piano_ar.config import ModelConfig
from piano_ar.hub import HF_REPO, load_weights, make_tokenizer, read_config, resolve_pretrained, save_pretrained
from piano_ar.train import context_patches

from .config import CoverConfig
from .model import CoverModel

SUBFOLDER = "piano_cover"
# 原曲の音声の特徴 (piano_cover.audio_frames) を作る PCA。公開用のフォルダでは重みの隣に置き、学習のチェックポイントでは
# 手元で作ったものを使う。model.frames_pca_path に入れる (特徴を使わないモデルは None)
FRAMES_PCA_NAME = "frames_pca.npz"
LOCAL_FRAMES_PCA = Path("data/audio_frames/original/pca.npz")


def load_pretrained_cover(
    name_or_path: str | Path = HF_REPO,
    *,
    subfolder: str | None = SUBFOLDER,
    revision: str | None = None,
    device: str | torch.device = "cpu",
) -> CoverModel:
    path = resolve_pretrained(name_or_path, subfolder, revision)
    config = read_config(path, "piano_cover")
    tokenizer = make_tokenizer(config["tokenizer_config"], config.get("tokenizer_type"))
    model = CoverModel(
        ModelConfig.from_dict(config["model_config"]),
        CoverConfig.from_dict(config["cover_config"]),
        tokenizer,
        config["source_vocab_size"],
    )
    load_weights(model, path)
    model.context_patches = int(config["context_patches"])
    model.frames_pca_path = path / FRAMES_PCA_NAME if model.config.frame_dims else None
    return model.to(device).eval()


def load_cover_checkpoint(
    checkpoint_path: str | Path, planner: str | Path | None = None, device: str | torch.device = "cpu"
) -> CoverModel:
    """学習のチェックポイント (.pt) を読む。planner を渡すと Planner だけを piano_cover.train_planner の出力に差し替える"""
    # optimizer の状態 (モデルの 2 倍の大きさ) も入っているので、mmap で必要な分だけ読む。
    # 学習中に上書きされるファイルを開いたままにすると学習側の保存が失敗するので、読んだらすぐに手放す
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    tokenizer_type = checkpoint.get("tokenizer_type")
    tokenizer = make_tokenizer(checkpoint["tokenizer_config"], tokenizer_type)
    model = CoverModel(
        ModelConfig.from_dict(checkpoint["model_config"]),
        CoverConfig.from_dict(checkpoint["cover_config"]),
        tokenizer,
        checkpoint["source_vocab_size"],
    )
    model.load_state_dict(checkpoint["model"])
    if tokenizer_type == "beat":  # 文脈は小節の数 (学習の窓 + 記憶)
        model.context_patches = int(checkpoint["args"]["window_bars"] + checkpoint["args"]["memory_bars"])
    else:
        model.context_patches = context_patches(checkpoint["args"], tokenizer.config.patch_seconds)
    del checkpoint
    model.frames_pca_path = LOCAL_FRAMES_PCA if model.config.frame_dims else None
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


def export_cover(model: CoverModel, out_dir: str | Path, frames_pca: str | Path | None = None) -> Path:
    """frames_pca は原曲の音声の特徴を使うモデル (frame_dims) の PCA (省略で model.frames_pca_path)。重みの隣に写す"""
    config = {
        "model_type": "piano_cover",
        "model_config": vars(model.decoder.config),
        "cover_config": vars(model.config),
        "tokenizer_config": vars(model.tokenizer.config),
        "tokenizer_type": "beat" if hasattr(model.tokenizer, "ticks_per_beat") else "time",
        "source_vocab_size": model.source_embedding.num_embeddings,
        "context_patches": model.context_patches,
    }
    out = save_pretrained(model, out_dir, config)
    if model.config.frame_dims:
        source = Path(frames_pca or model.frames_pca_path)
        if not source.is_file():
            raise FileNotFoundError(f"原曲の音声の特徴の PCA がありません: {source}")
        shutil.copy2(source, out / FRAMES_PCA_NAME)
    return out
