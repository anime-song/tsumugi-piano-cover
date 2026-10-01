"""学習したモデルを推論用の重みとして保存し、読み込む。Hugging Face の model repository からも読める。

推論用のフォルダ:
    config.json         モデルの種類 (model_type) と設定 (model_config / tokenizer_config / 生成で見る文脈の長さなど)
    model.safetensors   重みだけ (optimizer の状態・学習の設定・演奏者の一覧は入れない)

1 つの repository に piano_ar/ と piano_cover/ を並べて置き、subfolder で選ぶ。
手元のフォルダ (export の出力) を渡せば、ネットにはつながずにそれを読む。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

HF_REPO = "anime-song/tsumugi-piano-cover"
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
FORMAT_VERSION = 1


def save_pretrained(model: nn.Module, out_dir: str | Path, config: dict) -> Path:
    from safetensors.torch import save_model

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # save_model は重みを共有している層もそのまま保存できる (save_file は共有があると失敗する)
    save_model(model, str(out / WEIGHTS_NAME))
    config = {"format_version": FORMAT_VERSION, **config}
    (out / CONFIG_NAME).write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def resolve_pretrained(name_or_path: str | Path, subfolder: str | None = None, revision: str | None = None) -> Path:
    """config.json のあるフォルダを返す。手元にあればそれ (subfolder の下、なければそのもの)、
    なければ Hugging Face の repository から subfolder の分だけ取ってくる (2 回目からはキャッシュを使う)"""
    local = Path(name_or_path)
    for candidate in ([local / subfolder] if subfolder else []) + [local]:
        if (candidate / CONFIG_NAME).is_file():
            return candidate
    if local.exists():
        raise FileNotFoundError(f"{local} に {CONFIG_NAME} がありません")

    from huggingface_hub import snapshot_download

    patterns = [f"{subfolder}/*"] if subfolder else None
    root = Path(snapshot_download(str(name_or_path), revision=revision, allow_patterns=patterns))
    path = root / subfolder if subfolder else root
    if not (path / CONFIG_NAME).is_file():
        raise FileNotFoundError(f"{name_or_path} の {subfolder or '(直下)'} に {CONFIG_NAME} がありません")
    return path


def read_config(path: Path, model_type: str) -> dict:
    config = json.loads((path / CONFIG_NAME).read_text(encoding="utf-8"))
    if config.get("model_type") != model_type:
        raise ValueError(f"{path} は {config.get('model_type')} のモデルです ({model_type} を読もうとしました)")
    return config


def load_weights(model: nn.Module, path: Path) -> None:
    from safetensors.torch import load_model

    load_model(model, str(path / WEIGHTS_NAME), strict=True)


def load_pretrained_ar(
    name_or_path: str | Path = HF_REPO,
    *,
    subfolder: str | None = "piano_ar",
    revision: str | None = None,
    device: str | torch.device = "cpu",
):
    """PianoARModel.from_pretrained の中身。model.tokenizer と model.context_patches も付けて返す"""
    from .config import ModelConfig, TokenizerConfig
    from .model import PianoARModel
    from .tokenizer import PianoTokenizer

    path = resolve_pretrained(name_or_path, subfolder, revision)
    config = read_config(path, "piano_ar")
    tokenizer = PianoTokenizer(TokenizerConfig(**config["tokenizer_config"]))
    model = PianoARModel(ModelConfig.from_dict(config["model_config"]), tokenizer)
    load_weights(model, path)
    model.tokenizer = tokenizer
    model.context_patches = int(config["context_patches"])
    return model.to(device).eval()


def load_ar_checkpoint(checkpoint_path: str | Path, device: str | torch.device = "cpu"):
    """学習のチェックポイント (.pt) を from_pretrained と同じ形で読む"""
    from .config import ModelConfig, TokenizerConfig
    from .model import PianoARModel
    from .tokenizer import PianoTokenizer
    from .train import context_patches

    # optimizer の状態 (モデルの 2 倍の大きさ) も入っているので、mmap で必要な分だけ読む
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    tokenizer = PianoTokenizer(TokenizerConfig(**checkpoint["tokenizer_config"]))
    model = PianoARModel(ModelConfig.from_dict(checkpoint["model_config"]), tokenizer)
    model.load_state_dict(checkpoint["model"])
    model.tokenizer = tokenizer
    model.context_patches = context_patches(checkpoint["args"], tokenizer.config.patch_seconds)
    return model.to(device).eval()


def load_ar(name_or_path: str | Path, device: str | torch.device = "cpu", subfolder: str | None = "piano_ar"):
    """.pt なら学習のチェックポイント、それ以外は推論用のフォルダか Hugging Face の repository"""
    if str(name_or_path).endswith(".pt"):
        return load_ar_checkpoint(name_or_path, device)
    return load_pretrained_ar(name_or_path, subfolder=subfolder, device=device)
