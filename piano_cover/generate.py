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
from piano_ar.beat_tokenizer import BeatGrid, BeatTokenizer
from piano_ar.tokenizer import PianoTokenizer

from .arrangement import measurable
from .metadata import copy_metadata
from .config import CoverConfig
from .data import source_tensors
from .hub import load_cover
from .model import CoverModel, SourceCondition
from .source import (
    ROW_ONSET,
    ROW_TYPE,
    TYPE_NOTE,
    SourceVocab,
    load_source,
    onset_time_bias,
    source_features,
    trim_lead,
    with_sheetsage,
)

# 原曲の音声の SheetSage2-Refined の注釈 (piano_ar.sheetsage の出力) の置き場所
SHEETSAGE_DIR = Path("data/sheetsage/original")


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
    # 拍のトークンのモデル: 原曲の拍 (最初の音を含む小節から)、小節の拍数とテンポの段階
    grid: BeatGrid | None = None
    bar_beats: np.ndarray | None = None
    tempo: np.ndarray | None = None

    @property
    def num_patches(self) -> int:
        return int(self.features["features"].shape[0])


@dataclass
class Cover:
    events: np.ndarray  # 原曲の時刻のイベント (PianoTokenizer.events_to_midi に渡せる)
    patches: list[list[int]]  # パッチごとのトークン列 (lead を詰めた時間軸)。途中から作り直すときの prompt に使う


def prepare_source(
    path: str | Path, tokenizer: PianoTokenizer, cover_config: CoverConfig, sheetsage: str | Path | None = None
) -> PreparedSource:
    """sheetsage は原曲の音声の SheetSage2-Refined の注釈 (piano_ar.sheetsage の JSON)。SheetSage で学習したモデル
    (cover_config.sheetsage_beats / sheetsage_melody) では要る。省略すると data/sheetsage/original/<原曲>.json を探す"""
    rows, end_frame = load_source(path, tokenizer.config.frame_rate)
    beat_tokens = isinstance(tokenizer, BeatTokenizer)
    if cover_config.sheetsage_beats or cover_config.sheetsage_melody or beat_tokens:
        annotation_path = Path(sheetsage) if sheetsage else SHEETSAGE_DIR / f"{Path(path).stem}.json"
        if not annotation_path.exists():
            raise FileNotFoundError(
                f"{annotation_path} がありません。原曲の音声に python -m piano_ar.sheetsage をかけて作ってください"
            )
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        rows = with_sheetsage(
            rows,
            annotation,
            tokenizer.config.frame_rate,
            beats=cover_config.sheetsage_beats,
            melody=cover_config.sheetsage_melody,
        )
        rows = rows[rows[:, ROW_ONSET] < end_frame]
    if beat_tokens:
        return _prepare_beats(rows, end_frame, annotation, tokenizer, cover_config)
    rows, lead = trim_lead(rows)
    features = source_features(
        rows, end_frame - lead, tokenizer, SourceVocab(tokenizer.config), cover_config.max_source_rows
    )
    return PreparedSource(rows, lead, features)


def _prepare_beats(
    rows: np.ndarray, end_frame: int, annotation: dict, tokenizer: BeatTokenizer, cover_config: CoverConfig
) -> PreparedSource:
    """拍のトークンのモデルの原曲: 原曲の拍で小節ごとのパッチにし、原曲の最初の音を含む小節から生成する
    (それより前は無音か拍だけなので、学習で見た曲の始まりと合わない。時刻のトークンの trim_lead と同じ理由)"""
    fr = tokenizer.config.frame_rate
    grid = BeatGrid.build(
        annotation["beats"], annotation["downbeats"], fr, 0.0, end_frame, tokenizer.config.max_bar_beats
    )
    if grid is None:
        raise ValueError("原曲の拍が少なすぎます")
    return prepare_beats_source(rows, end_frame, grid, tokenizer, cover_config)


def prepare_beats_source(
    rows: np.ndarray, end_frame: int, grid: BeatGrid, tokenizer: BeatTokenizer, cover_config: CoverConfig
) -> PreparedSource:
    """原曲の行と拍 (grid) から、拍のトークンのモデルに渡す形を作る (学習中の生成評価からも使う)"""
    fr = tokenizer.config.frame_rate
    notes = rows[rows[:, ROW_TYPE] == TYPE_NOTE]
    first = float(notes[:, ROW_ONSET].min()) if len(notes) else 0.0
    first_bar = int(np.searchsorted(grid.bar_starts, grid.beat_position(np.array([first]))[0], side="right") - 1)
    grid = BeatGrid(grid.beats, grid.bar_starts[max(first_bar, 0) :])
    features = source_features(
        rows, end_frame, PianoTokenizer(), SourceVocab(tokenizer.config), cover_config.max_source_rows,
        grid=grid, ticks_per_beat=tokenizer.ticks_per_beat,
    )  # fmt: skip
    S = len(features["features"])
    bar_beats = grid.bar_beats[:S]
    tempo = tokenizer.tempo_bin(grid.tempo()[:S] * fr * 60)
    return PreparedSource(rows, 0, features, grid, bar_beats, tempo)


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
        if source.grid is not None:  # 拍のトークン: 小節の頭が seconds を超えるまで
            starts = (
                source.grid.beats[source.grid.bar_starts[:num_patches]] - source.grid.beats[source.grid.bar_starts[0]]
            )
            num_patches = max(1, int(np.searchsorted(starts, params.seconds * tokenizer.config.frame_rate)))
        else:
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
    if source.grid is not None:
        common["bar_beats"] = torch.from_numpy(source.bar_beats).to(device)
        common["tempo"] = torch.from_numpy(source.tempo).to(device)
    if params.onset_bias and source.grid is None:
        bias = onset_time_bias(
            source.rows,
            num_patches,
            tokenizer.patch_frames,
            params.onset_bias,
            params.onset_bias_width * tokenizer.config.frame_rate,
        )
        common["time_bias"] = torch.from_numpy(bias).to(device)
    tensors = {key: value.to(device) for key, value in source_tensors(source.features).items()}
    if source.grid is not None:
        # 小節の長さは曲ごとに異なり、patch * max_bar_ticks では原曲の位置とずれる。
        # 学習時と同じ小節開始の刻みを SourceCondition に渡す。
        tensors["src_patch_start"] = torch.from_numpy(source.features["patch_start"]).to(device)
    # bf16 を持たない GPU (Colab の T4 など) では、エミュレーションで遅くなったり失敗したりしないよう fp32 のまま動かす
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        condition = SourceCondition(model, tensors)
        arrangement = (params.fill, params.above, params.span) if params.arrangement else (None,) * 3
        known = (
            measurable(source.rows, num_patches, tokenizer.patch_frames, tokenizer.config.frame_rate)
            if source.grid is None
            else None
        )
        dynamics = model.planned_dynamics(
            condition.memory, params.channel, num_patches, params.dynamics, params.density, arrangement, known
        )
        samples = model.decoder.generate(
            tokenizer, num_samples=num_samples, condition=condition, dynamics=dynamics, **common
        )
    covers = []
    for patches in samples:
        if source.grid is not None:  # 拍のトークン: 原曲の拍の時刻に置く
            events = tokenizer.patches_to_events(patches, source.grid)
        else:
            events = tokenizer.patches_to_events(patches)
            events[:, 0] += source.lead  # 原曲の時刻に戻す
        covers.append(Cover(events, patches))
    return covers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", required=True, help="原曲の MIDI (tsumugi の merged)")
    parser.add_argument(
        "--sheetsage",
        default=None,
        help="原曲の SheetSage2-Refined の注釈 (省略で data/sheetsage/original/<原曲>.json)",
    )
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
    source = prepare_source(args.source, tokenizer, model.config, args.sheetsage)
    covers = generate_covers(model, source, params, num_samples=args.num_samples)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, cover in enumerate(covers):
        manner = f"ch{channel}" + (f"_onset{args.onset_bias:g}" if args.onset_bias else "")
        path = out_dir / f"{Path(args.source).stem}_{manner}_{i}.mid"
        tokenizer.events_to_midi(cover.events, path)
        # 原曲のテンポ・拍子・調・コードマーカーを写す
        copy_metadata(args.source, path)
        if args.wav:
            write_wav(synthesize(cover.events, tokenizer.config.frame_rate), path.with_suffix(".wav"))
        seconds = (cover.events[:, 0].max(initial=0) - cover.events[:, 0].min(initial=0)) / tokenizer.config.frame_rate
        print(f"{path}: {len(cover.patches)} パッチ ({seconds:.0f} 秒) / {len(cover.events)} イベント")


if __name__ == "__main__":
    main()
