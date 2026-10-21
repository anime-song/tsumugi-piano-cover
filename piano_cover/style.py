"""好みのベクトル: お気に入りのカバーから、自分用の「演奏者」の埋め込みを 1 本だけ学習する。

モデルの重みは変えず、演奏者の埋め込みの表の後ろに 1 行 (StyleEmbedding) を足して、その 1 行 (dim 次元) だけを
お気に入りのカバーの損失で学習する (画像生成の textual inversion と同じ考え方)。生成ではその行の番号を演奏者として渡すので、
演奏者らしさの強さ (channel_cfg、指定なしとの差を強調する CFG) と Planner (演奏者の埋め込みから強弱と編曲の性質の曲線を
予測する) はそのまま効く。

学習に使うカバーは、生成したときと同じ原曲の時間軸の上にある (カバーの小節 p = 原曲の小節 p) ので、アラインメントは要らない。
損失はデコーダの損失 (窓 window_bars 小節、その前に記憶 memory_bars 小節) と Planner の損失 (曲全体の強弱と編曲の性質の曲線)。
Planner の損失は演奏者の埋め込みにだけ流す (CoverModel.forward の channel_grad)。強弱と編曲の性質の条件は学習と同じ割合で
落とすので、条件を指定しないときの好みも覚える。

拍のトークンのモデル (V2 以降) だけで使える。Cover Studio (cover_studio) の「好みを学習」から呼ぶ。
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from piano_ar.beat_data import bar_dynamics
from piano_ar.beat_tokenizer import BeatTokenizer
from piano_ar.data import quantize_dynamics
from piano_ar.evaluation import _pedal_intervals, events_end_frame
from piano_ar.tokenizer import KIND, KIND_NOTE, KIND_PEDAL_OFF, KIND_PEDAL_ON, ONSET, PAD, sort_events

from .arrangement import ARRANGEMENT_NAMES, arrangement_values
from .beat_data import bar_plan_target
from .data import collate_cover, source_tensors
from .generate import PreparedSource
from .model import CoverModel


class StyleEmbedding(nn.Module):
    """演奏者の埋め込みの表 (base) の後ろに、好みのベクトル (style) を 1 行足した表。番号 index が好みのベクトル"""

    def __init__(self, base: nn.Embedding) -> None:
        super().__init__()
        self.base = base
        self.style = nn.Parameter(torch.zeros(base.embedding_dim, device=base.weight.device), requires_grad=False)

    @property
    def index(self) -> int:
        return self.base.num_embeddings

    def forward(self, channel: Tensor) -> Tensor:
        weight = self.base.weight
        return F.embedding(channel, torch.cat((weight, self.style[None].to(weight.dtype))))


def style_slot(model: CoverModel) -> StyleEmbedding:
    """model の演奏者の埋め込みを StyleEmbedding に差し替えて返す (差し替え済みならそのまま)"""
    embedding = model.decoder.channel_embedding
    if not isinstance(embedding, StyleEmbedding):
        embedding = StyleEmbedding(embedding)
        model.decoder.channel_embedding = embedding
    return embedding


def set_style(model: CoverModel, vector: np.ndarray | list[float] | Tensor) -> int:
    """好みのベクトルを入れて、生成で演奏者として渡す番号を返す"""
    slot = style_slot(model)
    value = torch.as_tensor(np.asarray(vector, dtype=np.float32), device=slot.style.device)
    if value.shape != slot.style.shape:
        raise ValueError(f"好みのベクトルの次元が合いません ({tuple(value.shape)} と {tuple(slot.style.shape)})")
    with torch.no_grad():
        slot.style.copy_(value)
    return slot.index


@dataclass
class StyleExample:
    """お気に入りのカバー 1 本。events は原曲の時刻 (フレーム) のイベント配列 (カバーの MIDI そのもの)。
    同じモデルで生成したカバーなら patches (生成したトークン列) も渡すと、損失はそのトークン列で取る
    (イベントからトークンにし直すと、ペダルの近くの長さなどが 2 割ほどの小節で少し変わる)"""

    source: PreparedSource
    events: np.ndarray
    patches: list[list[int]] | None = None


@dataclass
class StyleSettings:
    steps: int = 300
    lr: float = 1e-4
    batch_size: int = 2
    window_bars: int = 32
    memory_bars: int = 96
    song_start_prob: float = 0.3
    dynamics_dropout: float = 0.2  # 強弱と編曲の性質の条件を落とす確率 (学習と同じ)
    plan_weight: float = 1.0
    seed: int = 0


@dataclass
class StyleResult:
    vector: np.ndarray
    loss_before: float  # お気に入りの全小節の損失 (デコーダ + Planner)。初期値のベクトルで
    loss_after: float  # 学習したベクトルで
    distance: float  # 初期値からどれだけ動いたか (演奏者どうしの平均の距離を 1 とする)
    history: list[float] = field(default_factory=list)  # 更新ごとの損失


def _paired_pedals(events: np.ndarray) -> np.ndarray:
    """ペダルを on と off が交互に並ぶ形にする (BeatTokenizer.quantize の前提)。生成したトークンから戻したイベントでは
    on が続いたり、最後の on が閉じていなかったりする。評価 (piano_ar.evaluation) と同じく、踏んでいる間の on と
    離している間の off は捨て、閉じていない on は曲の終わりで離す"""
    pedal = events[:, KIND] != KIND_NOTE
    intervals = _pedal_intervals(events[pedal], events_end_frame(events))
    rows = np.zeros((2 * len(intervals), events.shape[1]), dtype=np.int64)
    rows[0::2, ONSET], rows[0::2, KIND] = intervals[:, 0], KIND_PEDAL_ON
    rows[1::2, ONSET], rows[1::2, KIND] = intervals[:, 1], KIND_PEDAL_OFF
    return sort_events(np.concatenate([events[~pedal].astype(np.int64), rows])).astype(events.dtype)


class _Song:
    """1 曲分の、窓を切り出す前の量 (小節に丸めたカバーと、Planner の正解)"""

    def __init__(self, model: CoverModel, example: StyleExample) -> None:
        tok = model.tokenizer
        source = example.source
        grid = source.grid
        self.source = source
        self.events = _paired_pedals(example.events)
        self.patches = example.patches
        self.quantized = tok.quantize(self.events, grid)
        if self.patches is not None:
            last = len(self.patches)
        else:
            bar_ticks = grid.bar_starts * tok.ticks_per_beat
            last = int(np.searchsorted(bar_ticks, max(self.quantized.end_tick - 1, 0), side="right"))
        self.num_bars = max(1, min(last, grid.num_bars, source.num_patches))
        self.source_tensors = source_tensors(source.features)
        self.source_tensors["src_patch_start"] = torch.from_numpy(source.features["patch_start"])
        self.arrangement = model.decoder.config.dynamics_columns >= 2 + len(ARRANGEMENT_NAMES)
        notes = self.events[self.events[:, KIND] == KIND_NOTE]
        self.note_frames = notes[:, ONSET].astype(np.float64)  # 原曲の時刻のまま (カバーの小節 = 原曲の小節)
        self.plan_target = None
        if model.planner is not None:
            S = source.num_patches
            # カバーの拍 i は原曲の拍 i
            target = bar_plan_target(self.quantized, tok, np.arange(len(grid.beats), dtype=np.float64), grid, S)
            if self.arrangement:
                bounds = grid.frames_at(grid.bar_starts[: S + 1].astype(np.float64))
                values = arrangement_values(self.events, self.note_frames, source.rows, bounds, tok.config.frame_rate)
                target = np.concatenate([target, values.astype(np.float32)], axis=1)
            self.plan_target = torch.from_numpy(target)


def _item(
    model: CoverModel,
    song: _Song,
    channel: int,
    first_bar: int,
    settings: StyleSettings,
    rng: random.Random | None = None,
) -> dict[str, Tensor]:
    """song の first_bar から window_bars 小節を損失に取る 1 件 (piano_cover.beat_data.BeatCoverDataset.cover_item と同じ形)。
    rng を渡すと、強弱と編曲の性質の条件を学習と同じ割合で落とす"""
    tok: BeatTokenizer = model.tokenizer
    tpb = tok.ticks_per_beat
    grid = song.source.grid
    start = max(0, first_bar - settings.memory_bars)
    offset = first_bar - start
    total = settings.memory_bars + settings.window_bars
    window = tok.tokenize_window(song.quantized, grid, start, total)
    if song.patches is not None:
        tokens = window["tokens"]
        for p in range(total):
            tokens[p] = PAD
            if start + p < len(song.patches):
                sequence = song.patches[start + p]
                tokens[p, : len(sequence)] = sequence
        window["patch_valid"] = np.arange(start, start + total) < len(song.patches)
    patch_loss = np.zeros(total, dtype=bool)
    patch_loss[offset : offset + settings.window_bars] = True
    window["patch_valid"][offset + settings.window_bars :] = False
    patch_loss &= window["patch_valid"]

    # 生成と同じく、カバーの小節 p は原曲の小節 p (CoverModel の SourceCondition)
    S = song.source.num_patches
    bars = np.clip(np.arange(start, start + total), 0, S - 1)
    patch_start = song.source.features["patch_start"]
    align = np.stack([patch_start[bars], patch_start[bars] + tok.config.max_bar_ticks], axis=1).astype(np.float32)

    item = {
        "tokens": torch.from_numpy(window["tokens"]),
        "patch_valid": torch.from_numpy(window["patch_valid"]),
        "pedal_state": torch.from_numpy(window["pedal_state"]),
        "bar_beats": torch.from_numpy(window["bar_beats"]),
        "tempo": torch.from_numpy(window["tempo"]),
        "channel": torch.tensor(channel),
        "song_start": torch.tensor(int(start == 0)),
        "align": torch.from_numpy(align),
        "align_bar": torch.from_numpy(bars.astype(np.float32)),
        "align_center": torch.from_numpy(bars.astype(np.int64)),
        "has_source": torch.tensor(True),
        "patch_loss": torch.from_numpy(patch_loss),
        **song.source_tensors,
    }
    if song.plan_target is not None:
        item["plan_target"] = song.plan_target
    bins = model.decoder.config.dynamics_bins
    if bins:
        index = quantize_dynamics(bar_dynamics(song.quantized, grid, tok, start, total), bins)
        if song.arrangement:
            window_bars = np.clip(np.arange(start, start + total + 1), 0, grid.num_bars)
            bounds = grid.frames_at(grid.bar_starts[window_bars].astype(np.float64))
            values = arrangement_values(song.events, song.note_frames, song.source.rows, bounds, tok.config.frame_rate)
            index = np.concatenate([index, quantize_dynamics(values, bins)], axis=1)
        if rng is not None and settings.dynamics_dropout > 0:
            if rng.random() < settings.dynamics_dropout:
                index[:] = 0
            else:
                for column in range(index.shape[1]):
                    if rng.random() < settings.dynamics_dropout / 2:
                        index[:, column] = 0
        item["dynamics"] = torch.from_numpy(index)
    return item


def _loss(model: CoverModel, items: list[dict[str, Tensor]], device: torch.device, plan_weight: float) -> Tensor:
    batch = {key: value.to(device) for key, value in collate_cover(items).items()}
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        output = model(batch, channel_grad=True)
    return output["loss"] + plan_weight * output.get("loss_plan", 0.0)


def _evaluate(model: CoverModel, songs: list[_Song], channel: int, settings: StyleSettings, device) -> float:
    """お気に入りの全小節を窓に分けて損失を測る (条件は落とさない)"""
    losses = []
    with torch.no_grad():
        for song in songs:
            for first_bar in range(0, song.num_bars, settings.window_bars):
                item = _item(model, song, channel, first_bar, settings)
                losses.append(float(_loss(model, [item], device, settings.plan_weight)))
    return float(np.mean(losses)) if losses else math.nan


def channel_distance_scale(model: CoverModel) -> float:
    """演奏者どうしの埋め込みの平均の距離 (指定なしの 0 番を除く。好みのベクトルがどれだけ動いたかの目安)"""
    weight = style_slot(model).base.weight.detach().float()[1:]
    sample = weight[torch.randperm(len(weight), generator=torch.Generator().manual_seed(0))[:512].to(weight.device)]
    return float(torch.cdist(sample, sample).mean())


def train_style(
    model: CoverModel,
    examples: list[StyleExample],
    init: Tensor | np.ndarray,
    settings: StyleSettings | None = None,
    progress: Callable[[int, int, float], None] | None = None,
) -> StyleResult:
    """examples (お気に入りのカバー) から好みのベクトルを学習する。init は初期値 (お気に入りで使った演奏者の埋め込みなど)。
    progress(終わった更新の数, 全体, 直近の損失) は更新のたびに呼ぶ (例外を投げれば止まる)。モデルの重みは変えない"""
    if not isinstance(model.tokenizer, BeatTokenizer):
        raise ValueError("好みの学習は拍のトークンのモデル (V2 以降) だけで使えます")
    if not examples:
        raise ValueError("お気に入りのカバーがありません")
    settings = settings or StyleSettings()
    device = next(model.parameters()).device
    rng = random.Random(settings.seed)
    songs = [_Song(model, example) for example in examples]
    slot = style_slot(model)
    channel = slot.index
    init = torch.as_tensor(init).detach().to(device=device, dtype=torch.float32)

    was_training = model.training
    requires_grad = [p.requires_grad for p in model.parameters()]
    model.eval()  # dropout なし。勾配は好みのベクトルにだけ流す
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        with torch.no_grad():
            slot.style.copy_(init)
        loss_before = _evaluate(model, songs, channel, settings, device)
        slot.style.requires_grad_(True)
        optimizer = torch.optim.Adam([slot.style], lr=settings.lr)
        # 曲の長さに比例して選ぶ
        weights = [song.num_bars for song in songs]
        history = []
        for step in range(settings.steps):
            items = []
            for song in rng.choices(songs, weights=weights, k=settings.batch_size):
                if rng.random() < settings.song_start_prob:
                    first_bar = 0
                else:
                    first_bar = rng.randint(0, max(0, song.num_bars - settings.window_bars // 2))
                items.append(_item(model, song, channel, first_bar, settings, rng))
            loss = _loss(model, items, device, settings.plan_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach()))
            if progress is not None:
                progress(step + 1, settings.steps, float(np.mean(history[-20:])))
        slot.style.requires_grad_(False)
        loss_after = _evaluate(model, songs, channel, settings, device)
        vector = slot.style.detach().float().cpu().numpy().copy()
    finally:
        slot.style.requires_grad_(False)
        for p, flag in zip(model.parameters(), requires_grad):
            p.requires_grad_(flag)
        model.train(was_training)
    distance = float(np.linalg.norm(vector - init.cpu().numpy())) / max(channel_distance_scale(model), 1e-6)
    return StyleResult(vector, loss_before, loss_after, distance, history)
