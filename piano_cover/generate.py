"""原曲の MIDI (tsumugi で採譜したもの) からピアノカバーを生成する。

python -m piano_cover.generate --checkpoint checkpoints/piano_cover/best.pt --source Dataset/original_midis_v2/merged/<id>.mid
python -m piano_cover.generate ... --channel UCxxxxxxxx --source-cfg 1.75 --onset-bias 4 --seconds 60 --wav

原曲の時間軸の上に生成するので、出力は原曲と同じタイミング・テンポになる。
生成は原曲の最初の音から始め (trim_lead)、出力はそのぶん後ろにずらして原曲の時刻に戻す。
--onset-bias を付けると、出力の onset を原曲の onset (全楽器の音と拍) に寄せる (onset_time_bias)。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from piano_ar.config import ModelConfig, TokenizerConfig
from piano_ar.evaluation import synthesize, write_wav
from piano_ar.tokenizer import PianoTokenizer

from .config import CoverConfig
from .data import source_tensors
from .model import CoverModel, SourceCondition
from .source import SourceVocab, load_source, onset_time_bias, source_features, trim_lead


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", required=True, help="原曲の MIDI (tsumugi の merged)")
    parser.add_argument("--out-dir", default="outputs/piano_cover")
    parser.add_argument("--seconds", type=float, default=None, help="生成する長さ。省略で原曲の最後まで")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--channel", default=None, help="チャンネル ID (UC...) か channel_index の番号。省略で指定なし")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--source-cfg", type=float, default=1.0, help="> 1 で原曲に忠実にする (原曲なしとの差を強調)")
    parser.add_argument(
        "--channel-cfg", type=float, default=1.0, help="> 1 で演奏者らしさを強める (--channel を指定したときだけ効く)"
    )
    parser.add_argument(
        "--onset-bias",
        type=float,
        default=0.0,
        help="原曲の onset に出力の onset を寄せる強さ (TIME の logit に足す最大値)。0 で使わない。4 前後がよい",
    )
    parser.add_argument("--onset-bias-width", type=float, default=0.02, help="寄せる範囲 (秒)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wav", action="store_true", help="確認用の簡易シンセ音声も保存する")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # チェックポイントには optimizer の状態 (モデルの 2 倍の大きさ) も入っているので、GPU に丸ごと載せず、
    # CPU で必要な分だけ読んで (mmap) モデルの重みだけを GPU に移す
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    tokenizer = PianoTokenizer(TokenizerConfig(**checkpoint["tokenizer_config"]))
    cover_config = CoverConfig(**checkpoint["cover_config"])
    model = CoverModel(
        ModelConfig.from_dict(checkpoint["model_config"]), cover_config, tokenizer, checkpoint["source_vocab_size"]
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    channel = 0
    if args.channel is not None:
        if args.channel.isdigit():
            channel = int(args.channel)
        else:
            channel = json.loads(Path(checkpoint["channel_index"]).read_text(encoding="utf-8"))[args.channel]

    rows, end_frame = load_source(args.source, tokenizer.config.frame_rate)
    rows, lead = trim_lead(rows)
    end_frame -= lead
    features = source_features(rows, end_frame, tokenizer, SourceVocab(tokenizer.config), cover_config.max_source_rows)
    source = {key: value.to(device) for key, value in source_tensors(features).items()}
    patch_seconds = tokenizer.config.patch_seconds
    num_patches = features["features"].shape[0]
    if args.seconds is not None:
        num_patches = min(num_patches, math.ceil(args.seconds / patch_seconds))

    common = {
        "num_patches": num_patches,
        "channel": channel,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "cfg_scale": args.channel_cfg,
        "condition_cfg_scale": args.source_cfg,
        "context_patches": round(checkpoint["args"]["window_seconds"] / patch_seconds),
        # 途中で EOS を出して止まらないよう、曲の終わりは原曲の最後の 2 パッチでだけ許す
        "end_after": features["features"].shape[0] - 2,
    }
    if args.onset_bias:
        bias = onset_time_bias(
            rows,
            num_patches,
            tokenizer.patch_frames,
            args.onset_bias,
            args.onset_bias_width * tokenizer.config.frame_rate,
        )
        common["time_bias"] = torch.from_numpy(bias).to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        condition = SourceCondition(model, source)
        samples = model.decoder.generate(tokenizer, num_samples=args.num_samples, condition=condition, **common)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, patches in enumerate(samples):
        events = tokenizer.patches_to_events(patches)
        events[:, 0] += lead  # 原曲の時刻に戻す
        manner = f"ch{channel}" + (f"_onset{args.onset_bias:g}" if args.onset_bias else "")
        path = out_dir / f"{Path(args.source).stem}_{manner}_{i}.mid"
        tokenizer.events_to_midi(events, path)
        if args.wav:
            write_wav(synthesize(events, tokenizer.config.frame_rate), path.with_suffix(".wav"))
        print(f"{path}: {len(patches) * patch_seconds:.0f} 秒 / {len(events)} イベント")


if __name__ == "__main__":
    main()
