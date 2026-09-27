"""ピアノカバーモデル: 事前学習したピアノ生成モデル (PianoARModel) に、原曲エンコーダと cross-attention を足す。

    原曲の行 (音・拍・コード・キー) --埋め込み--> 2 秒パッチごとの PatchEncoder (双方向) --> K 本の要約 + 各行のベクトル
    K 本の要約 x 全パッチ --SongEncoder (双方向, 全曲)--> 原曲のメモリ
    Global: 数ブロックごとに、対応する時刻 ± global_cross_window パッチの原曲のメモリへ cross-attention
    Local : 数ブロックごとに、対応するパッチ ± local_cross_radius パッチの原曲の各行へ cross-attention

「対応する時刻」は、学習時はアラインメント (カバー -> 原曲)、生成時は原曲の時間軸の上に生成するので恒等写像。
cross-attention の RoPE の位置は、クエリに対応する原曲の時刻、キーに原曲の時刻を使うので、時刻の差で見る場所が決まる。
曲全体の構造 (サビの繰り返しなど) は SongEncoder が各パッチのメモリに埋め込むので、デコーダは近くだけ見ればよい。
追加した層は出力のゲートを 0 で初期化するので、学習の始めは事前学習モデルとまったく同じ出力になる。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from piano_ar.config import ModelConfig
from piano_ar.model import Block, CrossBlock, CrossHook, PianoARModel, Transformer
from piano_ar.tokenizer import PianoTokenizer

from .config import CoverConfig


@dataclass
class SourceMemory:
    """原曲エンコーダの出力"""

    song: Tensor  # [B, S*K, D] SongEncoder の出力
    song_pos: Tensor  # [B, S*K] パッチ番号
    song_valid: Tensor  # [B, S*K]
    rows: Tensor  # [U, R, D] Local が見るパッチ (needed) の各行のベクトル
    row_onset: Tensor  # [U, R] 行の絶対フレーム
    row_valid: Tensor  # [U, R]
    lookup: Tensor  # [B, S] rows の何番目か (Local が見ないパッチは -1)


def query_onsets(prefix: Tensor, tokenizer: PianoTokenizer) -> Tensor:
    """Local の各入力位置 (BOS + prefix) の時点で、パッチ内のどの onset まで進んでいるか [N, T+1]"""
    is_time = (prefix >= tokenizer.time_offset) & (prefix < tokenizer.pitch_offset)
    values = torch.where(is_time, prefix - tokenizer.time_offset, torch.zeros_like(prefix))
    # パッチ内の TIME は増える一方なので、累積最大が「直前の TIME」になる
    current = values.cummax(dim=1).values if prefix.shape[1] else values
    # 生成の最初 (prefix が空) でも BOS の位置の 1 列は必要なので、prefix[:, :1] ではなく明示的に作る
    return torch.cat((prefix.new_zeros(prefix.shape[0], 1), current), dim=1).float()


class CoverModel(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        cover_config: CoverConfig,
        tokenizer: PianoTokenizer,
        source_vocab_size: int,
    ) -> None:
        super().__init__()
        self.config = cover_config
        self.tokenizer = tokenizer
        self.patch_frames = tokenizer.patch_frames
        dim = model_config.dim
        self.decoder = PianoARModel(model_config, tokenizer)

        K = cover_config.source_latents
        self.source_embedding = nn.Embedding(source_vocab_size, dim, padding_idx=0)
        self.source_onset = nn.Embedding(tokenizer.patch_frames, dim)
        self.latent_queries = nn.Parameter(torch.randn(K, dim) * 0.02)
        self.latent_slot = nn.Embedding(K, dim)
        self.patch_encoder = Transformer(model_config, cover_config.source_patch_layers, causal=False)
        self.song_encoder = Transformer(model_config, cover_config.source_song_layers, causal=False)

        def cross_layers(num_blocks: int, every: int) -> list[int]:
            return [i for i in range(num_blocks) if (i + 1) % every == 0]

        self.global_cross_blocks = cross_layers(model_config.global_layers, cover_config.global_cross_every)
        self.local_cross_blocks = cross_layers(model_config.local_layers, cover_config.local_cross_every)

        def make_cross() -> CrossBlock:
            return CrossBlock(dim, model_config.heads, model_config.mlp_ratio, model_config.dropout)

        self.global_cross = nn.ModuleList(make_cross() for _ in self.global_cross_blocks)
        self.local_cross = nn.ModuleList(make_cross() for _ in self.local_cross_blocks)
        self.gradient_checkpointing = False

        for module in (self.source_embedding, self.source_onset, self.latent_slot, self.patch_encoder,
                       self.song_encoder, self.global_cross, self.local_cross):  # fmt: skip
            module.apply(PianoARModel._init_weights)
        with torch.no_grad():
            self.source_embedding.weight[0].zero_()

    def new_parameters(self) -> list[nn.Parameter]:
        """事前学習にない (新しく足した) パラメータ"""
        decoder = {id(p) for p in self.decoder.parameters()}
        return [p for p in self.parameters() if id(p) not in decoder]

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.decoder.set_gradient_checkpointing(enabled)
        self.patch_encoder.gradient_checkpointing = enabled
        self.song_encoder.gradient_checkpointing = enabled
        self.gradient_checkpointing = enabled

    def compile_blocks(self) -> None:
        """デコーダ・原曲エンコーダ・cross-attention のブロックを 1 つずつ torch.compile する (PianoARModel.compile_blocks と同じ)。
        batch 16 で 1 ステップ 1.49 秒 -> 1.02 秒。"""
        torch._dynamo.config.recompile_limit = 64
        torch._dynamo.config.cache_size_limit = 64
        for module in self.modules():
            if isinstance(module, (Block, CrossBlock)):
                module.compile(dynamic=True)

    def run_cross(self, block: CrossBlock, *args: Tensor) -> Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(block, *args, use_reentrant=False)
        return block(*args)

    # ------------------------------------------------------------------
    # 原曲エンコーダ
    # ------------------------------------------------------------------
    def encode_source(self, batch: dict[str, Tensor], needed: Tensor, num_length_buckets: int = 8) -> SourceMemory:
        """needed [B, S]: Local が行を見るパッチ。それ以外のパッチは要約だけを残す"""
        features = batch["src_features"].long()
        row_valid = batch["src_valid"]
        patch_valid = batch["src_patch_valid"]
        B, S = patch_valid.shape
        K = self.config.source_latents
        dim = self.source_onset.weight.shape[1]
        device = features.device

        flat = patch_valid.nonzero()  # [Np, 2] (曲, パッチ)
        patch_features = features[patch_valid]
        patch_rows_valid = row_valid[patch_valid]
        onset = batch["src_onset"][patch_valid].long()
        local_onset = (onset - flat[:, 1:2] * self.patch_frames).clamp(0, self.patch_frames - 1)
        needed_flat = needed[patch_valid]
        lookup_flat = torch.where(needed_flat, needed_flat.long().cumsum(0) - 1, -1)
        num_needed = int(needed_flat.sum())
        lengths = patch_rows_valid.sum(-1)
        row_width = max(1, int(lengths[needed_flat].max())) if num_needed else 1

        # パッチごとの行数の差が大きいので、デコーダと同じく長さ順の塊に分けて通す
        order = lengths.argsort()
        latent_parts, row_parts = [], []
        for index in order.tensor_split(min(num_length_buckets, max(len(order), 1))):
            if len(index) == 0:
                continue
            length = int(lengths[index].max())
            rows = self.source_embedding(patch_features[index, :length]).sum(-2)
            rows = rows + self.source_onset(local_onset[index, :length])
            queries = self.latent_queries.to(rows.dtype).expand(len(index), -1, -1)
            key_valid = torch.cat((torch.ones_like(patch_rows_valid[index, :1]).expand(-1, K),
                                   patch_rows_valid[index, :length]), dim=1)  # fmt: skip
            out = self.patch_encoder(torch.cat((queries, rows), dim=1), key_valid)
            latent_parts.append(out[:, :K])
            keep = needed_flat[index]
            if keep.any():
                width = min(length, row_width)
                row_parts.append((lookup_flat[index[keep]], out[keep, K : K + width]))

        if latent_parts:
            latents = torch.cat(latent_parts)[order.argsort()]
            grid = latents.new_zeros(B, S, K, dim)
            grid[patch_valid] = latents
        else:
            grid = torch.zeros(B, S, K, dim, device=device)
        x = grid.reshape(B, S * K, dim) + self.latent_slot.weight.to(grid.dtype).repeat(S, 1)
        song_pos = torch.arange(S, device=device).repeat_interleave(K).float().expand(B, -1)
        song_valid = patch_valid.repeat_interleave(K, dim=1)
        # 原曲のない行 (事前学習の曲・原曲を外した窓) は全部を見せて NaN を防ぐ。出力は cross 側でマスクする
        key_valid = song_valid | ~song_valid.any(dim=1, keepdim=True)
        song = self.song_encoder(x, key_valid, positions=song_pos)

        rows = song.new_zeros(max(num_needed, 1), row_width, dim)
        for target, values in row_parts:
            rows[target, : values.shape[1]] = values.to(rows.dtype)
        row_onset = torch.zeros(max(num_needed, 1), row_width, device=device)
        valid_rows = torch.zeros(max(num_needed, 1), row_width, dtype=torch.bool, device=device)
        if num_needed:
            needed_rows = needed_flat.nonzero().squeeze(1)
            row_onset[lookup_flat[needed_rows]] = onset[needed_rows, :row_width].float()
            valid_rows[lookup_flat[needed_rows]] = patch_rows_valid[needed_rows, :row_width]
        lookup = torch.full((B, S), -1, dtype=torch.long, device=device)
        lookup[patch_valid] = lookup_flat
        return SourceMemory(song, song_pos, song_valid, rows, row_onset, valid_rows, lookup)

    def local_neighbors(self, centers: Tensor, lookup_rows: Tensor) -> Tensor:
        """原曲のパッチ centers [N] の ± local_cross_radius を lookup_rows [N, S] で rows の番号にする (範囲外は -1)"""
        r = self.config.local_cross_radius
        S = lookup_rows.shape[1]
        neighbors = centers[:, None] + torch.arange(-r, r + 1, device=centers.device)
        inside = (neighbors >= 0) & (neighbors < S)
        index = lookup_rows.gather(1, neighbors.clamp(0, S - 1))
        return torch.where(inside, index, -1)

    def local_memory(self, memory: SourceMemory, index: Tensor, start: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """index [N, 2r+1] の rows を並べて (メモリ [N, M, D], 位置 [N, M], 有効 [N, M]) にする。
        位置は start [N] (対応する原曲のフレーム) からの相対にして、RoPE の角度を小さく保つ"""
        n = index.shape[0]
        safe = index.clamp(min=0)
        rows = memory.rows[safe].reshape(n, -1, memory.rows.shape[-1])
        valid = (memory.row_valid[safe] & (index >= 0)[..., None]).reshape(n, -1)
        pos = memory.row_onset[safe].reshape(n, -1) - start[:, None]
        return rows, pos, valid

    # ------------------------------------------------------------------
    # 学習
    # ------------------------------------------------------------------
    def forward(self, batch: dict[str, Tensor], num_length_buckets: int = 8) -> dict[str, Tensor]:
        F_ = self.patch_frames
        valid = batch["patch_valid"]
        has_source = batch["has_source"]
        S = batch["src_patch_valid"].shape[1]

        # Local が見る原曲のパッチ (カバーの各パッチの中央に対応する原曲のパッチ ± r)
        align = batch["align"][valid]  # [N, 2] 原曲のフレーム
        song_of = valid.nonzero()[:, 0]
        centers = torch.div(align.mean(-1), F_, rounding_mode="floor").long()
        r = self.config.local_cross_radius
        neighbors = centers[:, None] + torch.arange(-r, r + 1, device=centers.device)
        inside = (neighbors >= 0) & (neighbors < S) & has_source[song_of][:, None]
        needed = torch.zeros_like(batch["src_patch_valid"])
        needed[song_of[:, None].expand_as(neighbors)[inside], neighbors[inside]] = True
        needed &= batch["src_patch_valid"]

        memory = self.encode_source(batch, needed, num_length_buckets)
        condition = _TrainingCondition(self, memory, batch, centers, song_of)
        return self.decoder(batch, num_length_buckets, condition)


class _TrainingCondition:
    def __init__(
        self, model: CoverModel, memory: SourceMemory, batch: dict[str, Tensor], centers: Tensor, song_of: Tensor
    ) -> None:
        self.model = model
        self.memory = memory
        F_ = model.patch_frames
        has_source = batch["has_source"]
        align = batch["align"]
        # Global: カバーのパッチ p の中央に対応する原曲の位置 (パッチ単位)。キーはパッチ番号 (の中央) と比べる
        self.global_pos = align.mean(-1) / F_ - 0.5
        distance = (memory.song_pos[:, None, :] - self.global_pos[:, :, None]).abs()
        self.global_mask = (
            (distance <= model.config.global_cross_window) & memory.song_valid[:, None, :] & has_source[:, None, None]
        )
        # Local
        self.local_align = align[batch["patch_valid"]]
        lookup = torch.where(has_source[:, None], memory.lookup, -1)
        self.local_index = model.local_neighbors(centers, lookup[song_of])

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
        align = self.local_align[index]
        rows, k_pos, valid = model.local_memory(self.memory, self.local_index[index], align[:, 0])
        # パッチ内の onset を、そのパッチに対応する原曲の区間に線形に写す
        q_pos = (align[:, 1:2] - align[:, 0:1]) * query_onsets(prefix, model.tokenizer) / model.patch_frames
        mask = valid[:, None, :]

        def hook(i: int, x: Tensor) -> Tensor:
            if i not in model.local_cross_blocks:
                return x
            block = model.local_cross[model.local_cross_blocks.index(i)]
            return model.run_cross(block, x, q_pos, rows, k_pos, mask)

        return hook


# ----------------------------------------------------------------------
# 生成
# ----------------------------------------------------------------------
class SourceCondition:
    """原曲 1 曲を条件にして生成する (PianoARModel.generate の condition に渡す)。原曲の時間軸の上に生成する"""

    def __init__(self, model: CoverModel, source: dict[str, Tensor]) -> None:
        """source は piano_cover.data.source_tensors の出力 (バッチの次元なし)"""
        self.model = model
        batch = {key: value[None] for key, value in source.items()}
        self.num_patches = int(batch["src_patch_valid"].shape[1])
        self.memory = model.encode_source(batch, batch["src_patch_valid"])

    def bind(self, conditioned: list[bool]) -> _BoundSourceCondition:
        return _BoundSourceCondition(self, conditioned)


class _BoundSourceCondition:
    def __init__(self, source: SourceCondition, conditioned: list[bool]) -> None:
        self.model = source.model
        self.memory = source.memory
        self.rows = len(conditioned)
        # False の行 (CFG の「原曲なし」) は原曲を見せない
        self.conditioned = torch.tensor(conditioned, dtype=torch.bool, device=self.memory.song.device)

    def global_cross(self, first: int, last: int) -> CrossHook:
        model, memory = self.model, self.memory
        device = memory.song.device
        q_pos = torch.arange(first, last + 1, device=device).float().expand(self.rows, -1)
        distance = (memory.song_pos[:, None, :] - q_pos[:1, :, None]).abs()
        mask = (distance <= model.config.global_cross_window) & memory.song_valid[:, None, :]
        mask = mask.expand(self.rows, -1, -1) & self.conditioned[:, None, None]
        song = memory.song.expand(self.rows, -1, -1)
        song_pos = memory.song_pos.expand(self.rows, -1)

        def hook(i: int, x: Tensor) -> Tensor:
            if i not in model.global_cross_blocks:
                return x
            return model.global_cross[model.global_cross_blocks.index(i)](x, q_pos, song, song_pos, mask)

        return hook

    def local_cross(self, patch: int, prefix: Tensor) -> CrossHook:
        model, memory = self.model, self.memory
        device = memory.song.device
        F_ = model.patch_frames
        index = model.local_neighbors(torch.tensor([patch], device=device), memory.lookup)
        rows, k_pos, valid = model.local_memory(memory, index, torch.tensor([patch * F_], device=device).float())
        rows, k_pos = rows.expand(self.rows, -1, -1), k_pos.expand(self.rows, -1)
        mask = (valid.expand(self.rows, -1) & self.conditioned[:, None])[:, None, :]
        q_pos = query_onsets(prefix, model.tokenizer)

        def hook(i: int, x: Tensor) -> Tensor:
            if i not in model.local_cross_blocks:
                return x
            return model.local_cross[model.local_cross_blocks.index(i)](x, q_pos, rows, k_pos, mask)

        return hook
