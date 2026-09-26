"""学習したモデルでピアノ曲を冒頭から生成して MIDI に保存する。

python -m piano_ar.generate --checkpoint checkpoints/piano_ar/latest.pt --seconds 60 --num-samples 4
python -m piano_ar.generate --checkpoint ... --channel UCxxxxxxxx --cfg-scale 1.5
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .config import ModelConfig, TokenizerConfig
from .evaluation import synthesize, write_wav
from .model import PianoARModel
from .tokenizer import PianoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="outputs/piano_ar")
    parser.add_argument("--seconds", type=float, default=60.0, help="生成する長さの上限 (EOS が出たらそこで終わる)")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--channel", default=None, help="チャンネル ID (UC...) か channel_index の番号。省略で無条件")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--context-seconds", type=float, default=None, help="Global が見る長さ。省略で学習時の窓の長さ")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wav", action="store_true", help="確認用の簡易シンセ音声も保存する")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer = PianoTokenizer(TokenizerConfig(**checkpoint["tokenizer_config"]))
    model = PianoARModel(ModelConfig(**checkpoint["model_config"]), tokenizer).to(device).eval()
    model.load_state_dict(checkpoint["model"])

    channel = 0
    if args.channel is not None:
        if args.channel.isdigit():
            channel = int(args.channel)
        else:
            channel = json.loads(Path(checkpoint["channel_index"]).read_text(encoding="utf-8"))[args.channel]

    patch_seconds = tokenizer.config.patch_seconds
    context_seconds = args.context_seconds or checkpoint["args"]["window_seconds"]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        samples = model.generate(
            tokenizer,
            num_patches=math.ceil(args.seconds / patch_seconds),
            channel=channel,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            cfg_scale=args.cfg_scale,
            context_patches=round(context_seconds / patch_seconds),
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.checkpoint).stem
    for i, patches in enumerate(samples):
        events = tokenizer.patches_to_events(patches)
        path = out_dir / f"{stem}_ch{channel}_{i}.mid"
        tokenizer.events_to_midi(events, path)
        if args.wav:
            write_wav(synthesize(events, tokenizer.config.frame_rate), path.with_suffix(".wav"))
        print(f"{path}: {len(patches) * patch_seconds:.0f} 秒 / {len(events)} イベント")


if __name__ == "__main__":
    main()
