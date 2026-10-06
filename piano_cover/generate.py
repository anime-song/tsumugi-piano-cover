"""原曲の MIDI (tsumugi で採譜したもの) からピアノカバーを生成する。

python -m piano_cover.generate --checkpoint checkpoints/piano_cover/best.pt --source Dataset/original_midis_v2/merged/<id>.mid
python -m piano_cover.generate ... --channel UCxxxxxxxx --source-cfg 1.75 --onset-bias 4 --seconds 60 --wav
python -m piano_cover.generate --checkpoint anime-song/tsumugi-piano-cover --source song.mid   # 公開した重み

原曲の時間軸の上に生成するので、出力は原曲と同じタイミング・テンポになる。
生成は原曲の最初の音から始め (trim_lead)、出力はそのぶん後ろにずらして原曲の時刻に戻す。
--onset-bias を付けると、出力の onset を原曲の onset (全楽器の音と拍) に寄せる (onset_time_bias)。
Planner のあるモデルは、原曲から予測した強弱と音の多さの曲線を条件にして生成する。--dynamics / --density は
その曲線の山と谷を何倍にするか (1 で学習データと同じくらい、大きくするほどメリハリが付く。0 で指定なし)。
編曲の性質も条件にしたモデル (piano_cover.arrangement) は、--fill (合いの手・オブリの量) / --above (メロディの上に
音を重ねる割合) / --span (音域の広さ) で、Planner の予測を全カバーでの標準偏差の単位でずらせる
(0 で予測のまま、+0.5 前後で合いの手の多い演奏者くらい)。--no-arrangement でその条件を外す。
--planner で、Planner だけを学習し直した重み (piano_cover.train_planner の出力) に差し替えられる。
--checkpoint は学習のチェックポイント (.pt)、piano_cover.export の出力のフォルダ、Hugging Face の repository のどれか。
UI (cover_studio) からはモデルを読んだまま prepare_source → generate_covers を何度も呼ぶ。
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from piano_ar.evaluation import synthesize, write_wav
from piano_ar.tokenizer import PianoTokenizer

from .arrangement import measurable
from .metadata import copy_metadata
from .config import CoverConfig
from .data import source_tensors
from .hub import load_cover
from .model import CoverModel, SourceCondition
from .source import SourceVocab, load_source, onset_time_bias, source_features, trim_lead


@dataclass
class CoverParams:
    """生成の設定 (CLI の同名の引数と同じ意味)。既定値は聴いて良かった設定"""

    channel: int = 0  # 演奏者の番号 (0 で指定なし)
    temperature: float = 1.0
    top_p: float = 0.90
    source_cfg: float = 2.0
    channel_cfg: float = 3.0
    onset_bias: float = 0.0
    onset_bias_width: float = 0.02
    dynamics: float = 2.0
    density: float = 1.0
    fill: float = 2.0
    above: float = 0.0
    span: float = 1.5
    arrangement: bool = True  # False で編曲の性質 (fill / above / span) の条件を指定なしにする
    seconds: float | None = None  # 生成する長さ。None で原曲の最後まで

    @classmethod
    def from_dict(cls, values: dict) -> CoverParams:
        names = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in names})


@dataclass
class PreparedSource:
    """原曲 1 曲分の、生成に渡す形。生成は原曲の最初の音から始め、出力は lead フレームだけ後ろにずらして原曲の時刻に戻す"""

    rows: np.ndarray
    lead: int
    features: dict[str, np.ndarray]

    @property
    def num_patches(self) -> int:
        return int(self.features["features"].shape[0])


@dataclass
class Cover:
    events: np.ndarray  # 原曲の時刻のイベント (PianoTokenizer.events_to_midi に渡せる)
    patches: list[list[int]]  # パッチごとのトークン列 (lead を詰めた時間軸)。途中から作り直すときの prompt に使う


def prepare_source(path: str | Path, tokenizer: PianoTokenizer, cover_config: CoverConfig) -> PreparedSource:
    rows, end_frame = load_source(path, tokenizer.config.frame_rate)
    rows, lead = trim_lead(rows)
    features = source_features(
        rows, end_frame - lead, tokenizer, SourceVocab(tokenizer.config), cover_config.max_source_rows
    )
    return PreparedSource(rows, lead, features)


def generate_covers(
    model: CoverModel,
    source: PreparedSource,
    params: CoverParams,
    *,
    num_samples: int = 1,
    prompt: list[list[int]] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[Cover]:
    """prompt (パッチごとのトークン列) を渡すと、先頭のパッチをそのまま使って続きだけを生成する。
    progress は PianoARModel.generate と同じ (終わったパッチ数, 全パッチ数)"""
    tokenizer = model.tokenizer
    device = next(model.parameters()).device
    num_patches = source.num_patches
    if params.seconds is not None:
        num_patches = min(num_patches, math.ceil(params.seconds / tokenizer.config.patch_seconds))
    common = {
        "num_patches": num_patches,
        "channel": params.channel,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "cfg_scale": params.channel_cfg,
        "condition_cfg_scale": params.source_cfg,
        "context_patches": model.context_patches,
        # 途中で EOS を出して止まらないよう、曲の終わりは原曲の最後の 2 パッチでだけ許す
        "end_after": source.num_patches - 2,
        "progress": progress,
    }
    if prompt:
        common["prompts"] = [prompt] * num_samples
    if params.onset_bias:
        bias = onset_time_bias(
            source.rows,
            num_patches,
            tokenizer.patch_frames,
            params.onset_bias,
            params.onset_bias_width * tokenizer.config.frame_rate,
        )
        common["time_bias"] = torch.from_numpy(bias).to(device)
    tensors = {key: value.to(device) for key, value in source_tensors(source.features).items()}
    # bf16 を持たない GPU (Colab の T4 など) では、エミュレーションで遅くなったり失敗したりしないよう fp32 のまま動かす
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        condition = SourceCondition(model, tensors)
        arrangement = (params.fill, params.above, params.span) if params.arrangement else (None,) * 3
        known = measurable(source.rows, num_patches, tokenizer.patch_frames, tokenizer.config.frame_rate)
        dynamics = model.planned_dynamics(
            condition.memory, params.channel, num_patches, params.dynamics, params.density, arrangement, known
        )
        samples = model.decoder.generate(
            tokenizer, num_samples=num_samples, condition=condition, dynamics=dynamics, **common
        )
    covers = []
    for patches in samples:
        events = tokenizer.patches_to_events(patches)
        events[:, 0] += source.lead  # 原曲の時刻に戻す
        covers.append(Cover(events, patches))
    return covers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", required=True, help="原曲の MIDI (tsumugi の merged)")
    parser.add_argument("--planner", default=None, help="Planner の重みを差し替える (piano_cover.train_planner の出力)")
    parser.add_argument("--out-dir", default="outputs/piano_cover")
    parser.add_argument("--seconds", type=float, default=None, help="生成する長さ。省略で原曲の最後まで")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--channel", default=None, help="チャンネル ID (UC...) か channel_index の番号。省略で指定なし")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.90)
    parser.add_argument("--source-cfg", type=float, default=2.0, help="> 1 で原曲に忠実にする (原曲なしとの差を強調)")
    parser.add_argument(
        "--channel-cfg", type=float, default=3.0, help="> 1 で演奏者らしさを強める (--channel を指定したときだけ効く)"
    )
    parser.add_argument(
        "--onset-bias",
        type=float,
        default=0.0,
        help="原曲の onset に出力の onset を寄せる強さ (TIME の logit に足す最大値)。0 で使わない。4 前後がよい",
    )
    parser.add_argument("--onset-bias-width", type=float, default=0.02, help="寄せる範囲 (秒)")
    parser.add_argument("--dynamics", type=float, default=2.0, help="強弱の曲線の倍率 (0 で指定なし)")
    parser.add_argument("--density", type=float, default=1.0, help="音の多さの曲線の倍率 (0 で指定なし)")
    parser.add_argument("--fill", type=float, default=2.0, help="合いの手・オブリの量を増やす量 (標準偏差の単位)")
    parser.add_argument("--above", type=float, default=0.0, help="メロディの上に音を重ねる割合を増やす量 (同上)")
    parser.add_argument("--span", type=float, default=1.5, help="音域の広さを増やす量 (同上)")
    parser.add_argument("--no-arrangement", action="store_true", help="編曲の性質の条件を指定なしにする")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wav", action="store_true", help="確認用の簡易シンセ音声も保存する")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_cover(args.checkpoint, planner=args.planner, device=device)
    tokenizer = model.tokenizer

    channel = 0
    if args.channel is not None:
        if args.channel.isdigit():
            channel = int(args.channel)
        else:  # チャンネル ID から番号への対応は公開しないので、学習のチェックポイントのときだけ使える
            index_path = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)["channel_index"]
            channel = json.loads(Path(index_path).read_text(encoding="utf-8"))[args.channel]

    params = CoverParams.from_dict({**vars(args), "channel": channel, "arrangement": not args.no_arrangement})
    source = prepare_source(args.source, tokenizer, model.config)
    covers = generate_covers(model, source, params, num_samples=args.num_samples)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    patch_seconds = tokenizer.config.patch_seconds
    for i, cover in enumerate(covers):
        manner = f"ch{channel}" + (f"_onset{args.onset_bias:g}" if args.onset_bias else "")
        path = out_dir / f"{Path(args.source).stem}_{manner}_{i}.mid"
        tokenizer.events_to_midi(cover.events, path)
        # 原曲のテンポ・拍子・調・コードマーカーを写す
        copy_metadata(args.source, path)
        if args.wav:
            write_wav(synthesize(cover.events, tokenizer.config.frame_rate), path.with_suffix(".wav"))
        print(f"{path}: {len(cover.patches) * patch_seconds:.0f} 秒 / {len(cover.events)} イベント")


if __name__ == "__main__":
    main()
