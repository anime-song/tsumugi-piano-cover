"""演奏 -> 楽譜モデル: 楽譜の事前学習モデル (piano_score、中身は piano_ar の PianoARModel) に、演奏のエンコーダと
cross-attention を足す。piano_cover のカバーモデルと同じ組み立て。

    演奏の行 (1 音 / ペダル = 1 行) --埋め込み--> 2 秒パッチごとの PatchEncoder (双方向) --> K 本の要約 + 各行のベクトル
    K 本の要約 x 全パッチ --SongEncoder (双方向, 窓全体)--> 演奏のメモリ
    Global: 数ブロックごとに、小節の基準の時刻の前後の演奏のメモリへ cross-attention
    Local : 数ブロックごとに、小節の基準の時刻の前後の演奏の各行へ cross-attention

カバーと違うのは、パッチ (小節) の長さが生成しながらでないと分からないこと。小節 p を予測し始める時点で分かっている
「前の小節の開始」(measure_ref) を基準にし、Local の各トークンが見る時刻は次のように見積もる:
    最初のトークン (MTIME) を予測する位置   前の小節の開始
    MTIME の後                              その小節の開始 + 小節の中の位置 (4 分音符単位) x 4 分音符の長さの見積もり
どこまでがその小節の音かは、モデルが演奏の行を見て判断する。
追加した層の出力のゲートは 0 で初期化するので、学習の始めは楽譜の事前学習モデルとまったく同じ出力になる。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from piano_ar.config import ModelConfig
from piano_ar.model import Block, CrossBlock, CrossHook, PianoARModel, Transformer, prepare_block_compile
from piano_score.tokenizer import MLEN, ScoreTokenizer

from .data import PATCH_FRAMES, PerformanceVocab
from .onset_head import ScoreOnsetHead, frac_positions, row_channels

CHUNK_TOKENS = 16384  # 演奏のパッチエンコーダに一度に通すトークン数の上限


@dataclass(frozen=True)
class Perf2ScoreConfig:
    perf_dim: int = 0  # 演奏エンコーダと cross-attention の中の幅 (0 ならデコーダと同じ)
    latents: int = 4  # 2 秒パッチごとの要約の本数
    patch_layers: int = 2
    song_layers: int = 4
    global_cross_every: int = 3
    local_cross_every: int = 2
    # Global が見る範囲 (基準の時刻からのパッチ数)
    global_back: float = 1.5
    global_ahead: float = 6.0
    # Local が見る範囲 (基準の時刻からのフレーム数)。遅い曲の長い小節 (4/4 で ♩=40 なら 6 秒) まで入るように
    local_back: int = 100
    local_ahead: int = 1100
    # ScoreOnsetHead (位置のトークンに、対応する時刻の近くの打鍵の量を足す) の幅。0 なら使わない
    onset_head_dim: int = 64


def perf2score_config(values: dict) -> Perf2ScoreConfig:
    """チェックポイントに保存した設定から作る。OnsetHead を足す前のチェックポイント (onset_head_dim がない) は使わない設定にする"""
    values = dict(values)
    values.setdefault("onset_head_dim", 0)
    return Perf2ScoreConfig(**values)


@dataclass
class PerformanceMemory:
    song: Tensor  # [B, S*K, D]
    song_pos: Tensor  # [B, S*K] パッチ番号 (の中央)
    song_valid: Tensor  # [B, S*K]
    rows: Tensor  # [B, S, R, D] 各行のベクトル
    row_onset: Tensor  # [B, S, R] フレーム
    row_valid: Tensor  # [B, S, R]
    row_channels: Tensor  # [B, S, R, C] 打鍵の種類 (onset_head.CHANNELS)


class Perf2ScoreModel(nn.Module):
    def __init__(self, model_config: ModelConfig, config: Perf2ScoreConfig, tokenizer: ScoreTokenizer) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.decoder = PianoARModel(model_config, tokenizer)
        self.vocab = PerformanceVocab()

        dim = config.perf_dim or model_config.dim
        heads = dim // (model_config.dim // model_config.heads)
        perf_config = replace(model_config, dim=dim, heads=heads)
        K = config.latents
        self.row_embedding = nn.Embedding(self.vocab.size, dim, padding_idx=0)
        self.row_onset = nn.Embedding(PATCH_FRAMES, dim)
        self.latent_queries = nn.Parameter(torch.randn(K, dim) * 0.02)
        self.latent_slot = nn.Embedding(K, dim)
        self.patch_encoder = Transformer(perf_config, config.patch_layers, causal=False)
        self.song_encoder = Transformer(perf_config, config.song_layers, causal=False)

        def cross_layers(num_blocks: int, every: int) -> list[int]:
            return [i for i in range(num_blocks) if (i + 1) % every == 0]

        self.global_cross_blocks = cross_layers(model_config.global_layers, config.global_cross_every)
        self.local_cross_blocks = cross_layers(model_config.local_layers, config.local_cross_every)

        def make_cross() -> CrossBlock:
            return CrossBlock(
                model_config.dim, heads, model_config.mlp_ratio, model_config.dropout, memory_dim=dim, inner_dim=dim
            )

        self.global_cross = nn.ModuleList(make_cross() for _ in self.global_cross_blocks)
        self.local_cross = nn.ModuleList(make_cross() for _ in self.local_cross_blocks)
        self.onset_head = None
        if config.onset_head_dim:
            self.onset_head = ScoreOnsetHead(
                model_config.dim, config.onset_head_dim, tokenizer, config.local_back, config.local_ahead
            )
            self.onset_head.apply(PianoARModel._init_weights)
            nn.init.zeros_(self.onset_head.query.weight)
        self.gradient_checkpointing = False

        for module in (self.row_embedding, self.row_onset, self.latent_slot, self.patch_encoder, self.song_encoder,
                       self.global_cross, self.local_cross):  # fmt: skip
            module.apply(PianoARModel._init_weights)
        with torch.no_grad():
            self.row_embedding.weight[0].zero_()

        # Local の各位置の「小節の中の位置」を求める表 (拍のトークン -> 拍の数、分数のトークン -> 値。それ以外は -1)
        beat = torch.full((tokenizer.vocab_size,), -1.0)
        frac = torch.full((tokenizer.vocab_size,), -1.0)
        for i, kind in enumerate(tokenizer.kinds):
            if kind == "beat":
                beat[i] = float(tokenizer.values[i])
            elif kind == "frac":
                frac[i] = float(tokenizer.values[i])
        self.register_buffer("beat_value", beat, persistent=False)
        self.register_buffer("frac_value", frac, persistent=False)

    def new_parameters(self) -> list[nn.Parameter]:
        decoder = {id(p) for p in self.decoder.parameters()}
        return [p for p in self.parameters() if id(p) not in decoder]

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.decoder.set_gradient_checkpointing(enabled)
        self.patch_encoder.gradient_checkpointing = False  # 塊ごとにまとめて checkpoint する
        self.song_encoder.gradient_checkpointing = enabled
        self.gradient_checkpointing = enabled

    def compile_blocks(self) -> None:
        prepare_block_compile()
        for module in self.modules():
            if isinstance(module, (Block, CrossBlock)):
                module.compile(dynamic=True)

    def run_cross(self, block: CrossBlock, *args: Tensor) -> Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(block, *args, use_reentrant=False)
        return block(*args)

    # ------------------------------------------------------------------
    # 演奏のエンコーダ
    # ------------------------------------------------------------------
    def encode_performance(self, batch: dict[str, Tensor]) -> PerformanceMemory:
        features = batch["perf_features"].long()
        onset = batch["perf_onset"].long()
        row_valid = batch["perf_valid"]
        patch_valid = batch["perf_patch_valid"]
        B, S, R = row_valid.shape
        K = self.config.latents
        dim = self.row_onset.weight.shape[1]
        device = features.device

        flat = patch_valid.nonzero()
        local_onset = (onset[patch_valid] - flat[:, 1:2] * PATCH_FRAMES).clamp(0, PATCH_FRAMES - 1)
        patch_features, patch_rows_valid = features[patch_valid], row_valid[patch_valid]

        def encode(features: Tensor, onset: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
            n, length = valid.shape
            rows = F.embedding_bag(
                features.reshape(-1, features.shape[-1]), self.row_embedding.weight, mode="sum", padding_idx=0
            ).reshape(n, length, -1)
            rows = rows + self.row_onset(onset)
            queries = self.latent_queries.to(rows.dtype).expand(n, -1, -1)
            key_valid = torch.cat((torch.ones_like(valid[:, :1]).expand(-1, K), valid), dim=1)
            out = self.patch_encoder(torch.cat((queries, rows), dim=1), key_valid)
            return out[:, :K], out[:, K:]

        latent_parts, row_parts = [], []
        chunk = max(1, CHUNK_TOKENS // (K + R))
        for index in torch.arange(len(flat), device=device).split(chunk):
            args = (patch_features[index], local_onset[index], patch_rows_valid[index])
            if self.gradient_checkpointing and self.training:
                latents, rows = checkpoint(encode, *args, use_reentrant=False)
            else:
                latents, rows = encode(*args)
            latent_parts.append(latents)
            row_parts.append(rows)
        latents = torch.cat(latent_parts) if latent_parts else torch.zeros(0, K, dim, device=device)
        grid = latents.new_zeros(B, S, K, dim)
        grid[patch_valid] = latents
        rows = latents.new_zeros(B, S, R, dim)
        if row_parts:
            rows[patch_valid] = torch.cat(row_parts)

        x = grid.reshape(B, S * K, dim) + self.latent_slot.weight.to(grid.dtype).repeat(S, 1)
        song_pos = (torch.arange(S, device=device).float() + 0.5).repeat_interleave(K).expand(B, -1)
        song_valid = patch_valid.repeat_interleave(K, dim=1)
        key_valid = song_valid | ~song_valid.any(dim=1, keepdim=True)
        song = self.song_encoder(x, key_valid, positions=song_pos)
        channels = row_channels(features, self.vocab)
        return PerformanceMemory(song, song_pos, song_valid, rows, onset.float(), row_valid, channels)

    def local_memory(self, memory: PerformanceMemory, song: Tensor, ref: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """小節ごとに、基準の時刻 ref [N] (フレーム) の前後のパッチの行を並べる。(行 [N, P*R, D], 位置 [N, P*R], 有効)。
        位置は ref からの相対のフレーム (RoPE の角度を小さく保つ)"""
        return self.local_window(memory, song, ref)[:3]

    def local_window(
        self, memory: PerformanceMemory, song: Tensor, ref: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """local_memory に打鍵の種類 [N, P*R, C] を足したもの"""
        c = self.config
        S = memory.rows.shape[1]
        first = torch.div(ref - c.local_back, PATCH_FRAMES, rounding_mode="floor").long()
        P = (c.local_back + c.local_ahead) // PATCH_FRAMES + 2
        patches = first[:, None] + torch.arange(P, device=ref.device)
        inside = (patches >= 0) & (patches < S)
        safe = patches.clamp(0, S - 1)
        rows = memory.rows[song[:, None], safe]  # [N, P, R, D]
        n = rows.shape[0]
        pos = memory.row_onset[song[:, None], safe] - ref[:, None, None]
        valid = memory.row_valid[song[:, None], safe] & inside[..., None]
        valid &= (pos >= -c.local_back) & (pos <= c.local_ahead)
        channels = memory.row_channels[song[:, None], safe].reshape(n, -1, memory.row_channels.shape[-1])
        return rows.reshape(n, -1, rows.shape[-1]), pos.reshape(n, -1), valid.reshape(n, -1), channels

    def query_times(self, prefix: Tensor, ref: Tensor, start: Tensor, spq: Tensor) -> Tensor:
        """Local の各入力位置 (BOS + prefix) で見る演奏の時刻 (ref からの相対フレーム) [N, T+1]"""
        N, T = prefix.shape
        times = (start - ref)[:, None].expand(N, T + 1).clone()
        times[:, 0] = 0.0  # MTIME を予測する位置は前の小節の開始
        if T == 0:
            return times
        beat, frac = self.beat_value[prefix], self.frac_value[prefix]
        previous = F.pad(prefix, (1, 0), value=-1)[:, :-1]
        before_previous = F.pad(prefix, (2, 0), value=-1)[:, :-2]
        is_beat = (beat >= 0) & (previous != MLEN)  # MLEN の後の拍は小節の長さ
        is_frac = (frac >= 0) & (before_previous != MLEN)
        index = torch.arange(T, device=prefix.device).expand(N, T)
        last_beat = torch.where(is_beat, index, -1).cummax(1).values
        last_frac = torch.where(is_frac, index, -1).cummax(1).values
        position = beat.clamp(min=0).gather(1, last_beat.clamp(min=0)) * (last_beat >= 0)
        position = position + frac.clamp(min=0).gather(1, last_frac.clamp(min=0)) * (last_frac > last_beat)
        times[:, 1:] = (start - ref)[:, None] + position * spq[:, None]
        return times

    # ------------------------------------------------------------------
    # 学習
    # ------------------------------------------------------------------
    def forward(self, batch: dict[str, Tensor], num_length_buckets: int = 8) -> dict[str, Tensor]:
        memory = self.encode_performance(batch)
        condition = _TrainingCondition(self, memory, batch)
        return self.decoder(batch, num_length_buckets, condition)


class _TrainingCondition:
    def __init__(self, model: Perf2ScoreModel, memory: PerformanceMemory, batch: dict[str, Tensor]) -> None:
        self.model = model
        self.memory = memory
        c = model.config
        valid = batch["patch_valid"]
        # Global: 小節の基準の時刻 (パッチ単位) の前後
        self.global_pos = batch["measure_ref"] / PATCH_FRAMES
        delta = memory.song_pos[:, None, :] - self.global_pos[:, :, None]
        self.global_mask = (delta >= -c.global_back) & (delta <= c.global_ahead) & memory.song_valid[:, None, :]
        # Local: 有効な小節を並べた順
        index = valid.nonzero()
        self.song_of = index[:, 0]
        # 曲の最初の小節 (MTIME が「最初の音から小節線までさかのぼる秒数」になる)
        self.first = (index[:, 1] == 0) & (batch["song_start"][self.song_of] == 1)
        self.ref = batch["measure_ref"][valid]
        self.start = batch["measure_start"][valid]
        self.spq = batch["measure_spq"][valid]

    def global_cross(self) -> CrossHook:
        model, memory = self.model, self.memory

        def hook(i: int, x: Tensor) -> Tensor:
            if i not in model.global_cross_blocks:
                return x
            block = model.global_cross[model.global_cross_blocks.index(i)]
            return model.run_cross(block, x, self.global_pos, memory.song, memory.song_pos, self.global_mask)

        return hook

    def local_cross(self, index: Tensor, prefix: Tensor) -> CrossHook:
        model = self.model
        ref = self.ref[index]
        rows, k_pos, valid = model.local_memory(self.memory, self.song_of[index], ref)
        q_pos = model.query_times(prefix, ref, self.start[index], self.spq[index])
        mask = valid[:, None, :]

        def hook(i: int, x: Tensor) -> Tensor:
            if i not in model.local_cross_blocks:
                return x
            block = model.local_cross[model.local_cross_blocks.index(i)]
            return model.run_cross(block, x, q_pos, rows, k_pos, mask)

        return hook

    def local_output(self, index: Tensor, prefix: Tensor):
        head = self.model.onset_head
        if head is None:
            return None
        ref, start, spq = self.ref[index], self.start[index], self.spq[index]
        _, pos, valid, channels = self.model.local_window(self.memory, self.song_of[index], ref)
        smoothed = head.smoothed(pos, valid, channels)
        mtime, beat = head.candidates(smoothed, start - ref, spq, self.first[index])
        n, t, beat_value = frac_positions(prefix, self.model.beat_value)
        frac = head.frac_candidates(smoothed[n], (start - ref)[n], spq[n], beat_value) if len(n) else None
        return lambda h, logits: head(h, logits, mtime, beat, frac, (n, t))
