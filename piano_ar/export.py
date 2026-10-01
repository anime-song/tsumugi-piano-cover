"""学習のチェックポイントから、推論に要るものだけを取り出して公開用のフォルダにする (piano_ar.hub)。

python -m piano_ar.export --checkpoint checkpoints/piano_ar_long/best.pt --out pretrained/piano_ar

出力は config.json と model.safetensors だけ。optimizer の状態・学習の設定 (args)・演奏者の一覧 (channel_index) は
入れない。演奏者は番号 (1〜num_channels-1) でだけ指定できる。
"""

from __future__ import annotations

import argparse

from .hub import load_ar_checkpoint, save_pretrained


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = load_ar_checkpoint(args.checkpoint)
    config = {
        "model_type": "piano_ar",
        "model_config": vars(model.config),
        "tokenizer_config": vars(model.tokenizer.config),
        "context_patches": model.context_patches,
    }
    out = save_pretrained(model, args.out, config)
    print(f"{out}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M パラメーター")


if __name__ == "__main__":
    main()
