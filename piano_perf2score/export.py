"""学習のチェックポイントから、推論に要るものだけを取り出して公開用のフォルダにする (piano_perf2score.hub)。

python -m piano_perf2score.export --checkpoint checkpoints/piano_perf2score_beat5_mix/best.pt --out pretrained_v2/piano_perf2score

出力は config.json と model.safetensors だけ。optimizer の状態と学習の設定 (args) は入れない。
カバーの重み (piano_ar / piano_cover) と同じ repository の piano_perf2score/ に置く想定。
"""

from __future__ import annotations

import argparse

from .hub import export_perf2score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = export_perf2score(args.checkpoint, args.out)
    print(out)


if __name__ == "__main__":
    main()
