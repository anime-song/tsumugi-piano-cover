"""学習のチェックポイントから、推論に要るものだけを取り出して公開用のフォルダにする (piano_cover.hub)。

python -m piano_cover.export --checkpoint checkpoints/piano_cover_v4/best.pt --out pretrained/piano_cover
python -m piano_cover.export --checkpoint ... --planner checkpoints/piano_cover_v4/planner.pt --out ...

出力は config.json と model.safetensors だけ。optimizer の状態・学習の設定 (args)・演奏者の一覧 (channel_index) は
入れない。--planner を渡すと、Planner を学習し直した重みに差し替えてから書き出す。
piano_ar の重み (python -m piano_ar.export) と同じ repository の piano_ar/ と piano_cover/ に置く想定。
"""

from __future__ import annotations

import argparse

from .hub import export_cover, load_cover_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--planner", default=None, help="Planner を差し替える (piano_cover.train_planner の出力)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = load_cover_checkpoint(args.checkpoint, args.planner)
    out = export_cover(model, args.out)
    print(f"{out}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M パラメーター")


if __name__ == "__main__":
    main()
