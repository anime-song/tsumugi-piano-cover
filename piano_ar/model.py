"""Global Patch -> Local Decoder のピアノ生成モデル (NotaGen 型)。

    パッチ p-1 のトークン --PatchSummarizer--> 要約ベクトル
    Global (因果): 位置 p の入力 = 要約(p-1) (p=0 は BOS) + ペダル状態(p) + チャンネル  -> h_p
    Local  (因果): h_p を全位置に足して、パッチ p のトークン列を 1 つずつ生成する

Global の入力は生成済みパッチの中身そのものなので、Local の出力は要約を通じて次のパッチに渡る。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .tokenizer import EOS, PAD, TOKEN_GROUPS, BatchGrammar, PianoTokenizer

# Transformer のブロックの後に挟む処理 (ブロック番号, x) -> x。カバーモデルの cross-attention に使う
CrossHook = Callable[[int, Tensor], Tensor]
# Local の最後の出力 (h, logits) -> logits。カバーモデルで原曲の onset の位置を TIME の予測に足すのに使う
OutputHook = Callable[[Tensor, Tensor], Tensor]


class Condition(Protocol):
    """学習時に外から条件 (原曲など) を cross-attention で入れるためのフック"""

    def global_cross(self) -> CrossHook: ...

    def local_cross(self, index: Tensor, prefix: Tensor) -> CrossHook:
        """index は有効なパッチを並べたときの番号、prefix はそのパッチの Local の入力トークン [N, T]"""
        ...

    def local_output(self, index: Tensor, prefix: Tensor) -> OutputHook | None: ...


class StepCondition(Protocol):
    """Local を 1 トークンずつ生成するとき (local_step) に条件を入れる。パッチごとに変わる量は tensors にまとめて先に作り
    (形は同じ生成の中でパッチによらず同じ)、トークンごとにはそれを読むだけにする (CUDA Graph に取り込めるように)"""

    def step_cross(self, tensors: dict[str, Tensor], i: int, x: Tensor, onset: Tensor) -> Tensor:
        """Local の i 番目のブロックの後に挟む。x [N, 1, D]、onset [N] はその位置までに進んだパッチ内の onset"""
        ...

    def step_output(self, tensors: dict[str, Tensor], h: Tensor, logits: Tensor) -> Tensor: ...


class GenerationCondition(StepCondition, Protocol):
    def global_cross(self, first: int, last: int) -> CrossHook:
        """Global にパッチ first..last を入れるときのフック"""
        ...

    def local_cross(self, patch: int, prefix: Tensor) -> CrossHook: ...

    def local_step_tensors(self, patch: int) -> dict[str, Tensor]:
        """パッチ patch を local_step で生成するときの tensors"""
        ...


class GenerationConditionSource(Protocol):
    def bind(self, conditioned: list[bool]) -> GenerationCondition:
        """生成の行ごとに条件を入れるか (conditioned[i]) を決めたフックを作る。False の行は条件なしにする"""
        ...


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # 融合カーネル版。統計量は内部で fp32 で計算され、自前で書くより中間テンソルが少なく速い
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


def _rope(positions: Tensor, head_dim: int) -> tuple[Tensor, Tensor]:
    """positions ([L] か [B, L]、小数も可) の回転角の cos/sin。[B, L] なら [B, 1, L, head_dim/2] で返す"""
    # 位置と角度は必ず fp32 で計算する。bf16 だと 256 を超える位置が 2 刻みでしか表せず、別の位置が同じ角度になる。
    # autocast の中から呼ばれても下げられないよう明示的に切っておく (回転そのものは _apply_rope で bf16 で行ってよい)
    device = positions.device
    with torch.autocast(device.type, enabled=False):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
        angles = positions.float()[..., None] * inv_freq
        if angles.dim() == 3:
            angles = angles[:, None]  # ヘッドの次元
        return angles.cos(), angles.sin()


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # 回転は入力と同じ精度 (学習時は bf16) で行う。fp32 に上げると逆伝播用に保持する中間テンソルが倍になる。
    # 角度は fp32 で求めた cos/sin を丸めるだけなので、誤差は位置によらず一定 (相対 0.2% 程度) で位置が潰れることはない
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


def prepare_block_compile(recompile_limit: int = 64) -> None:
    """ブロック単位の torch.compile の前に、作り直しの上限を上げて、使うグラフの順番を固定する。

    ブロックの種類 x 学習/評価 x マスクの有無、生成時の長さ 1 の入力などで作り直しが起きる。既定の上限 (8) を超えると
    以降は compile されずに通常実行になり、gradient checkpointing の再計算が forward と別の実行 (graph) になって失敗する。
    torch 2.13 から dynamo の設定への代入はスレッドごとで、backward (checkpoint の再計算) は autograd の別スレッドで
    動くので、代入ではなく既定値を書き換えて全スレッドに効かせる。
    """
    for name in ("recompile_limit", "cache_size_limit"):
        setattr(torch._dynamo.config, name, recompile_limit)
    torch._dynamo.config._config["recompile_limit"].default = recompile_limit
    # グラフが増えたときに、checkpoint の再計算が forward と別のグラフを選ばないよう、見る順番を固定する
    torch._C._dynamo.eval_frame._set_lru_cache(False)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.attn_norm = RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        # QK-norm: q と k をヘッドごとに正規化してから内積を取る。これがないと qkv の重みが育ち続けて
        # attention logit が際限なく大きくなり (初回の事前学習では Local の 1 層目で ~1e7)、学習が崩壊した
        self.q_norm = RMSNorm(dim // heads)
        self.k_norm = RMSNorm(dim // heads)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.mlp_norm = RMSNorm(dim)
        hidden = int(dim * mlp_ratio * 2 / 3 + 63) // 64 * 64
        self.gate_up = nn.Linear(dim, hidden * 2, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor], causal: bool, key_valid: Tensor | None) -> Tensor:
        batch, length, dim = x.shape
        q, k, v = (
            self.qkv(self.attn_norm(x)).view(batch, length, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4)
        )
        # rms_norm は autocast 下で fp32 を返すので、SDPA に渡す前に v と同じ精度に戻す
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
        q, k = _apply_rope(q, *rope), _apply_rope(k, *rope)
        # 因果マスクとパディングマスクを同時に使う場面はない (因果側は右詰めのパディングで結果に影響しない)
        mask = key_valid[:, None, None, :] if key_valid is not None else None
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=causal, dropout_p=self.dropout if self.training else 0.0
        )
        x = x + self.residual_dropout(self.attn_out(attn.transpose(1, 2).reshape(batch, length, dim)))
        gate, up = self.gate_up(self.mlp_norm(x)).chunk(2, dim=-1)
        return x + self.residual_dropout(self.down(F.silu(gate) * up))

    def forward_step(
        self, x: Tensor, rope: tuple[Tensor, Tensor], cache: tuple[Tensor, Tensor], step: Tensor, key_valid: Tensor
    ) -> Tensor:
        """生成用 (因果・推論のみ): 位置 step ([1]、GPU の上の番号) の 1 トークン x [N, 1, D] だけを計算する。
        k / v は固定長の cache [N, H, L, head_dim] の step の所に書き、key_valid [L] (step 以前) の位置を見る。
        形が毎回同じなので CUDA Graph に取り込める"""
        batch, _, dim = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).view(batch, 1, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
        q, k = _apply_rope(q, *rope), _apply_rope(k, *rope)
        cache_k, cache_v = cache
        cache_k.index_copy_(2, step, k.to(cache_k.dtype))
        cache_v.index_copy_(2, step, v.to(cache_v.dtype))
        attn = F.scaled_dot_product_attention(q, cache_k.to(v.dtype), cache_v.to(v.dtype), attn_mask=key_valid)
        x = x + self.attn_out(attn.transpose(1, 2).reshape(batch, 1, dim))
        gate, up = self.gate_up(self.mlp_norm(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * up)


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig, layers: int, causal: bool) -> None:
        super().__init__()
        self.causal = causal
        self.head_dim = config.dim // config.heads
        self.blocks = nn.ModuleList(
            Block(config.dim, config.heads, config.mlp_ratio, config.dropout) for _ in range(layers)
        )
        self.norm = RMSNorm(config.dim)
        # True にすると各ブロックの入力だけを保持し、backward で中身を再計算してメモリを節約する
        self.gradient_checkpointing = False

    def forward(
        self,
        x: Tensor,
        key_valid: Tensor | None = None,
        positions: Tensor | None = None,
        cross: CrossHook | None = None,
    ) -> Tensor:
        """positions を省略すると 0, 1, 2, ... の位置で RoPE をかける。
        cross を渡すと i 番目のブロックの後に x = cross(i, x) を挟む (カバーモデルの cross-attention 用)。
        """
        if positions is None:
            positions = torch.arange(x.shape[1], device=x.device)
        rope = _rope(positions, self.head_dim)
        for i, block in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, rope, self.causal, key_valid, use_reentrant=False)
            else:
                x = block(x, rope, self.causal, key_valid)
            if cross is not None:
                x = cross(i, x)
        return self.norm(x)


class CrossBlock(nn.Module):
    """ゲート付きの cross-attention + MLP (Flamingo 型)。キーには常に見られる学習済みの「空」のキーを 1 つ足す
    (原曲がない窓や、範囲内に何もないときの行き先)。

    memory_dim はキー側 (原曲エンコーダなど) の幅、inner_dim は attention と MLP の中間の幅 (省略でどちらも dim)。
    dim より細くすると、残差の幅はデコーダのまま、足す部分のパラメータを減らせる。
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
        memory_dim: int | None = None,
        inner_dim: int | None = None,
    ) -> None:
        super().__init__()
        memory_dim = memory_dim or dim
        inner_dim = inner_dim or dim
        self.heads = heads
        self.dropout = dropout
        self.inner_dim = inner_dim
        head_dim = inner_dim // heads
        self.norm = RMSNorm(dim)
        self.memory_norm = RMSNorm(memory_dim)
        self.q = nn.Linear(dim, inner_dim, bias=False)
        self.kv = nn.Linear(memory_dim, inner_dim * 2, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.null_k = nn.Parameter(torch.randn(heads, head_dim) * 0.02)
        self.null_v = nn.Parameter(torch.zeros(heads, head_dim))
        self.out = nn.Linear(inner_dim, dim, bias=False)
        self.mlp_norm = RMSNorm(dim)
        hidden = int(inner_dim * mlp_ratio * 2 / 3 + 63) // 64 * 64
        self.gate_up = nn.Linear(dim, hidden * 2, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.mlp_gate = nn.Parameter(torch.zeros(1))
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, q_pos: Tensor, memory: Tensor, k_pos: Tensor, mask: Tensor) -> Tensor:
        """x [B, Lq, D], q_pos [B, Lq], memory [B, M, memory_dim], k_pos [B, M], mask [B, Lq or 1, M] (True で見る)"""
        return self.attend(x, q_pos, *self.memory_kv(memory, k_pos), mask)

    def memory_kv(self, memory: Tensor, k_pos: Tensor) -> tuple[Tensor, Tensor]:
        """キー側の k / v [B, H, 1+M, head_dim] (先頭は空のキー)。生成ではパッチごとに 1 回だけ作って使い回す"""
        batch = memory.shape[0]
        head_dim = self.inner_dim // self.heads
        k, v = self.kv(self.memory_norm(memory)).view(batch, -1, 2, self.heads, head_dim).permute(2, 0, 3, 1, 4)
        k = _apply_rope(self.k_norm(k).to(v.dtype), *_rope(k_pos, head_dim))
        # 空のキーは位置によらないよう回転をかけない
        null_k = self.k_norm(self.null_k).to(v.dtype)[None, :, None].expand(batch, -1, -1, -1)
        null_v = self.null_v.to(v.dtype)[None, :, None].expand(batch, -1, -1, -1)
        return torch.cat((null_k, k), dim=2), torch.cat((null_v, v), dim=2)

    def attend(self, x: Tensor, q_pos: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
        """memory_kv の k / v を見る。mask [B, Lq or 1, M] は空のキーを除いた分"""
        batch, length, _ = x.shape
        head_dim = self.inner_dim // self.heads
        q = self.q(self.norm(x)).view(batch, length, self.heads, head_dim).transpose(1, 2)
        q = _apply_rope(self.q_norm(q).to(v.dtype), *_rope(q_pos, head_dim))
        mask = F.pad(mask, (1, 0), value=True)[:, None]
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        attn = self.out(attn.transpose(1, 2).reshape(batch, length, self.inner_dim))
        x = x + torch.tanh(self.attn_gate).to(x.dtype) * self.residual_dropout(attn)
        gate, up = self.gate_up(self.mlp_norm(x)).chunk(2, dim=-1)
        return x + torch.tanh(self.mlp_gate).to(x.dtype) * self.residual_dropout(self.down(F.silu(gate) * up))


class PianoARModel(nn.Module):
    def __init__(self, config: ModelConfig, tokenizer: PianoTokenizer) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = tokenizer.vocab_size
        self.register_buffer("token_group", torch.from_numpy(tokenizer.token_group), persistent=False)
        # 損失をトークンの種類別に見るときの名前。楽譜のトークナイザーなど、別の語彙で使うときは tokenizer.group_names に持たせる
        self.group_names: tuple[str, ...] = tuple(getattr(tokenizer, "group_names", TOKEN_GROUPS))
        dim = config.dim

        self.token_embedding = nn.Embedding(self.vocab_size, dim)
        # PatchSummarizer: 1 パッチ分のトークン列を 1 本の要約ベクトルにまとめる双方向 Transformer。
        # 先頭に置いた summary_token の位置の出力を要約として使う。要約専用の損失はなく、
        # Local の次トークン予測の損失が Global を通って流れてくることで、次のパッチの予測に役立つ情報を詰めるよう学習される
        self.summary_token = nn.Parameter(torch.randn(dim) * 0.02)
        self.patch_summarizer = Transformer(config, config.summary_layers, causal=False)
        self.summary_projection = nn.Linear(dim, dim)

        # BOS は窓が曲の冒頭から始まるか (1) 途中から始まるか (0) で分ける
        self.bos = nn.Embedding(2, dim)
        self.pedal_embedding = nn.Embedding(2, dim)
        self.channel_embedding = nn.Embedding(config.num_channels, dim)
        # パッチごとの強さと音の多さなど (0 は指定なし)。列 c の番号 + c * (bins + 1) を同じ表で引く
        if config.dynamics_bins:
            self.dynamics_embedding = nn.Embedding(config.dynamics_columns * (config.dynamics_bins + 1), dim)
        self.global_transformer = Transformer(config, config.global_layers, causal=True)

        self.local_bos = nn.Parameter(torch.randn(dim) * 0.02)
        self.local_context = nn.Linear(dim, dim)
        self.local_transformer = Transformer(config, config.local_layers, causal=True)
        self.head = nn.Linear(dim, self.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        if config.dynamics_bins:
            # 0 で始めて、足す前 (以前のチェックポイント) と同じ出力から学習する
            nn.init.zeros_(self.dynamics_embedding.weight)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    # ------------------------------------------------------------------
    # 部品
    # ------------------------------------------------------------------
    def summarize_patches(self, tokens: Tensor) -> Tensor:
        """[N, L] のパッチトークンを [N, D] の要約ベクトルにする (Global の入力になる)"""
        summary_token = self.summary_token.to(self.token_embedding.weight.dtype).expand(tokens.shape[0], 1, -1)
        x = torch.cat((summary_token, self.token_embedding(tokens)), dim=1)
        key_valid = torch.cat((torch.ones_like(tokens[:, :1], dtype=torch.bool), tokens != PAD), dim=1)
        return self.summary_projection(self.patch_summarizer(x, key_valid)[:, 0])

    def global_forward(
        self,
        summaries: Tensor,
        song_start: Tensor,
        pedal_state: Tensor,
        channel: Tensor,
        cross: CrossHook | None = None,
        dynamics: Tensor | None = None,
    ) -> Tensor:
        """summaries[:, p] はパッチ p の要約。位置 p にはパッチ p-1 の要約を入れて h を返す。
        dynamics [B, P, C] はパッチ p の強さと音の多さ (カバーでは編曲の性質も) の番号 (0 は指定なし)"""
        inputs = torch.cat((self.bos(song_start)[:, None], summaries[:, :-1]), dim=1)
        inputs = inputs + self.pedal_embedding(pedal_state) + self.channel_embedding(channel)[:, None]
        if dynamics is not None and self.config.dynamics_bins:
            offset = torch.arange(dynamics.shape[-1], device=dynamics.device) * (self.config.dynamics_bins + 1)
            inputs = inputs + self.dynamics_embedding(dynamics + offset).sum(-2)
        return self.global_transformer(inputs, cross=cross)

    def local_forward(
        self, prefix: Tensor, context: Tensor, cross: CrossHook | None = None, output: OutputHook | None = None
    ) -> Tensor:
        """prefix [N, T] の続きを予測する logits [N, T+1, V] を返す"""
        bos = self.local_bos.to(self.token_embedding.weight.dtype).expand(prefix.shape[0], 1, -1)
        x = torch.cat((bos, self.token_embedding(prefix)), dim=1) + self.local_context(context)[:, None]
        h = self.local_transformer(x, cross=cross)
        logits = self.head(h)
        return output(h, logits) if output is not None else logits

    def local_step(
        self,
        token: Tensor,
        step: Tensor,
        context: Tensor,
        cache: list[tuple[Tensor, Tensor]],
        onset: Tensor,
        condition: StepCondition | None = None,
        tensors: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """local_forward の生成用の 1 トークン版: 位置 step ([1]) の入力 token [N] (step 0 は BOS) から次の logits [N, V]。
        cache はブロックごとの固定長の k / v (LocalSampler)。onset [N] はその位置までに進んだパッチ内の onset
        (query_onsets の最後の列と同じ)。condition と tensors (パッチごとの量) で原曲などの条件を入れる"""
        embedded = self.token_embedding(token)[:, None]
        bos = self.local_bos.to(embedded.dtype)
        x = torch.where(step == 0, bos, embedded) + self.local_context(context)[:, None]
        transformer = self.local_transformer
        rope = _rope(step, transformer.head_dim)
        key_valid = torch.arange(cache[0][0].shape[2], device=step.device) <= step
        for i, block in enumerate(transformer.blocks):
            x = getattr(block, "_orig_mod", block).forward_step(x, rope, cache[i], step, key_valid)
            if condition is not None:
                x = condition.step_cross(tensors, i, x, onset)
        h = transformer.norm(x)
        logits = self.head(h)
        if condition is not None:
            logits = condition.step_output(tensors, h, logits)
        return logits[:, 0]

    # ------------------------------------------------------------------
    # 学習
    # ------------------------------------------------------------------
    def set_gradient_checkpointing(self, enabled: bool) -> None:
        for transformer in (self.patch_summarizer, self.global_transformer, self.local_transformer):
            transformer.gradient_checkpointing = enabled

    def compile_blocks(self) -> None:
        """Transformer のブロック (と cross-attention のブロック) を 1 つずつ torch.compile する。

        Transformer ごと compile すると、原曲の cross-attention のフック (毎回作り直す関数) が
        compile の対象に入って作り直しが続く。中身の決まったブロック単位なら、パッチ数やトークン長を可変 (dynamic) にして
        使い回せる。学習時間の多くは行列積ではなく RMSNorm・RoPE・型変換などの細かい演算なので、融合の効果が大きい。
        nn.Module.compile はその場で置き換えるので、state_dict のキー名は変わらない。最初の 1 ステップは数分かかる。
        """
        prepare_block_compile()
        for module in self.modules():
            if isinstance(module, (Block, CrossBlock)):
                module.compile(dynamic=True)

    def forward(
        self, batch: dict[str, Tensor], num_length_buckets: int = 8, condition: Condition | None = None
    ) -> dict[str, Tensor]:
        """学習時の損失。

        batch["patch_loss"] [B, P] があれば、False のパッチは文脈 (記憶) としてだけ使い、損失に入れない。
        記憶のパッチは要約を勾配なしで作って Global に入れるだけで、Local は通さない。窓の前に長い記憶を付けても
        (曲全体の文脈)、計算の重い要約の backward と Local は損失を取るパッチの分しか増えない (Transformer-XL と同じ考え方)。

        パッチごとのトークン長は中央値 70 前後に対して最長は 200 を超えるので、全パッチを最長に揃えると
        計算とメモリの大半がパディングに使われる。パッチを長さ順に num_length_buckets 個の塊に分け、
        塊ごとにその中の最長まで切り詰めて PatchSummarizer と Local Decoder を通す。
        """
        tokens = batch["tokens"]
        valid = batch["patch_valid"]
        patch_loss = batch["patch_loss"] & valid if "patch_loss" in batch else valid
        memory = valid & ~patch_loss
        # condition には、有効なパッチを並べたときの番号で渡す (学習するパッチはその一部)
        trained = patch_loss[valid].nonzero().squeeze(1)
        flat = tokens[patch_loss]  # [N, L] 損失を取るパッチだけ

        def length_buckets(patches: Tensor) -> list[tuple[Tensor, int]]:
            lengths = (patches != PAD).sum(-1)
            order = lengths.argsort()
            return [
                (index, int(lengths[index].max()))
                for index in order.tensor_split(min(num_length_buckets, len(order)))
                if len(index)
            ]

        def summarize(patches: Tensor, buckets: list[tuple[Tensor, int]]) -> Tensor:
            order = torch.cat([index for index, _ in buckets])
            summarized = torch.cat([self.summarize_patches(patches[index, :length]) for index, length in buckets])
            return summarized[order.argsort()]

        buckets = length_buckets(flat)
        summarized = summarize(flat, buckets)
        summaries = summarized.new_zeros(*valid.shape, self.config.dim)
        if memory.any():
            # 記憶は compile しない通常の実行で通す。勾配なしの呼び出しで compile のグラフが増えると、gradient checkpointing の
            # 再計算が forward と別のグラフを選んで失敗する (pytorch/pytorch#166926)
            with torch.no_grad(), torch.compiler.set_stance("force_eager"):
                patches = tokens[memory]
                summaries[memory] = summarize(patches, length_buckets(patches)).to(summaries.dtype)
        summaries[patch_loss] = summarized
        context = self.global_forward(
            summaries,
            batch["song_start"],
            batch["pedal_state"],
            batch["channel"],
            cross=condition.global_cross() if condition is not None else None,
            dynamics=batch.get("dynamics"),
        )[patch_loss]

        loss_sum = torch.zeros(len(self.group_names), device=tokens.device)
        counts = torch.zeros(len(self.group_names), device=tokens.device)
        for index, length in buckets:
            target = flat[index, :length]
            cross = output = None
            if condition is not None:
                cross = condition.local_cross(trained[index], target[:, :-1])
                output = condition.local_output(trained[index], target[:, :-1])
            logits = self.local_forward(target[:, :-1], context[index], cross, output)
            token_loss = F.cross_entropy(
                logits.float().reshape(-1, self.vocab_size), target.reshape(-1), ignore_index=PAD, reduction="none"
            )
            group = self.token_group[target.reshape(-1)]
            keep = group >= 0  # PAD は除く
            loss_sum = loss_sum.index_add(0, group[keep], token_loss[keep])
            counts = counts.index_add(0, group[keep], torch.ones_like(token_loss[keep]))

        output = {"loss": loss_sum.sum() / counts.sum().clamp_min(1), "tokens": counts.sum()}
        for index, name in enumerate(self.group_names):
            output[f"loss_{name}"] = loss_sum[index] / counts[index].clamp_min(1)
        return output

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        tokenizer: PianoTokenizer,
        *,
        num_patches: int,
        channel: int = 0,
        num_samples: int = 1,
        temperature: float = 1.0,
        top_p: float = 0.95,
        cfg_scale: float = 1.0,
        context_patches: int = 32,
        prompts: list[list[list[int]]] | None = None,
        condition: GenerationConditionSource | None = None,
        condition_cfg_scale: float = 1.0,
        time_bias: Tensor | None = None,
        end_after: int = 0,
        dynamics: Tensor | None = None,
    ) -> list[list[list[int]]]:
        """曲の冒頭から生成し、サンプルごとにパッチのトークン列のリストを返す。

        context_patches を超えたら、学習時の「途中から始まる窓」と同じ形 (BOS(途中) + 直近のパッチ) で続ける。
        cfg_scale > 1 ならチャンネル指定なしとの差を強調する classifier-free guidance を使う。
        prompts[i] (パッチごとのトークン列) を渡すと、サンプル i の先頭のパッチはそのトークンで埋めて続きを生成する。
        condition (原曲など) を渡すと cross-attention で条件を入れる。
        time_bias [num_patches, patch_frames] を渡すと、パッチ内の各 onset の TIME トークンの logit にその値を足す
        (_bias_time。原曲の onset に出力の onset を寄せるときに使う)。
        end_after を渡すと、EOS (曲の終わり) はパッチ end_after 以降でだけ出せる (カバーは原曲の長さで終わりが決まる)。
        dynamics [num_patches, C] はパッチごとの強さと音の多さなどの番号 (piano_ar.data.quantize_dynamics、0 は指定なし)。

        条件と「演奏の仕方」(チャンネル) の強さは別々に決める (InstructPix2Pix と同じ 2 段の guidance)。
        それぞれを外したときの予測を使って
            L(なし, なし) + condition_cfg_scale * (L(条件, なし) - L(なし, なし)) + cfg_scale * (L(条件, 仕方) - L(条件, なし))
        にする。両方 1 なら L(条件, 仕方) そのもの。必要な組み合わせだけを num_samples 行ずつ並べて一緒に計算する。
        """
        device = self.token_embedding.weight.device
        has_manner = channel != 0
        manner_guidance = cfg_scale != 1.0 and has_manner
        condition_guidance = condition is not None and condition_cfg_scale != 1.0
        # (条件を入れるか, 演奏の仕方を入れるか) の組。先頭が本来の予測
        blocks = [(True, True)]
        if has_manner and (manner_guidance or condition_guidance):
            blocks.append((True, False))
        if condition_guidance:
            blocks.append((False, False))
        rows = num_samples * len(blocks)
        manner = torch.tensor([m for _, m in blocks for _ in range(num_samples)], device=device)
        channels = torch.where(manner, channel, 0).long()
        conditioned = [flag for flag, _ in blocks for _ in range(num_samples)]
        bound = condition.bind(conditioned) if condition is not None else None

        def guide(logits: Tensor) -> Tensor:
            parts = logits.split(num_samples)
            full = parts[0]
            no_channel = parts[1] if has_manner and len(parts) > 1 else full
            guided = no_channel + cfg_scale * (full - no_channel)
            if condition_guidance:
                nothing = parts[-1]
                guided = nothing + condition_cfg_scale * (no_channel - nothing) + cfg_scale * (full - no_channel)
            return guided

        sampler = LocalSampler(self, tokenizer, bound)
        patches: list[list[list[int]]] = [[] for _ in range(num_samples)]
        summaries: list[Tensor] = []  # パッチごとの要約 [rows, D]
        pedal_states: list[bool] = [False] * num_samples
        pedal_history: list[list[bool]] = [[] for _ in range(num_samples)]
        done = [False] * num_samples

        for p in range(num_patches):
            if all(done):
                break
            for i in range(num_samples):
                pedal_history[i].append(pedal_states[i])
            first = max(0, p - context_patches + 1)
            song_start = int(first == 0)
            window = summaries[first:p]
            # global_forward は最後の要約を捨てるので、ダミーを 1 つ足して位置 p まで計算する
            stacked = torch.stack(window + [torch.zeros(rows, self.config.dim, device=device)], dim=1)
            pedal = torch.tensor([h[first : p + 1] for h in pedal_history], dtype=torch.long, device=device)
            pedal = pedal.repeat(rows // num_samples, 1)
            song_start_tensor = torch.full((rows,), song_start, dtype=torch.long, device=device)
            global_cross = bound.global_cross(first, p) if bound is not None else None
            window_dynamics = dynamics[first : p + 1][None].expand(rows, -1, -1) if dynamics is not None else None
            context = self.global_forward(
                stacked, song_start_tensor, pedal, channels, global_cross, dynamics=window_dynamics
            )[:, -1]

            forced = None
            if prompts is not None and any(p < len(prompt) for prompt in prompts):
                forced = torch.full((num_samples, tokenizer.config.max_patch_tokens), -1, dtype=torch.long)
                for i, prompt in enumerate(prompts):
                    if p < len(prompt):
                        forced[i, : len(prompt[p])] = torch.tensor(prompt[p])
                forced = forced.to(device)
            sequence, pedal_down, song_end = sampler.sample_patch(
                context,
                bound.local_step_tensors(p) if bound is not None else {},
                torch.tensor(pedal_states, device=device),
                guide=guide,
                temperature=temperature,
                top_p=top_p,
                finished=torch.tensor(done, device=device),
                allow_eos=p >= end_after,
                time_bias=time_bias[p] if time_bias is not None else None,
                forced=forced,
            )

            pedal_down, song_end = pedal_down.tolist(), song_end.tolist()
            for i, tokens in enumerate(sequence.tolist()):
                if done[i]:
                    continue
                patches[i].append([t for t in tokens if t != PAD])
                pedal_states[i] = pedal_down[i]
                done[i] = song_end[i]
            summaries.append(self.summarize_patches(sequence).repeat(rows // num_samples, 1))
        return patches


class LocalSampler:
    """Local (パッチ内のトークン列) を 1 トークンずつ生成する。

    1 トークンの計算は小さい (1 行 1 位置) ので、素直に書くと GPU のカーネルの起動・Python の文法の判定・CPU と GPU の
    同期に時間の大半を使う (1 トークン約 10ms)。ここでは
        - 文法 (BatchGrammar) とサンプリングを GPU の上で行い、トークンごとに CPU に戻さない
        - 過去の k / v を固定長のキャッシュに置き、1 トークンの計算 (local_step) の形を毎回同じにする
        - 原曲のキー側など、パッチの中で変わらない量はパッチごとに 1 回だけ作る (StepCondition の tensors)
        - GPU なら 1 トークン分 (モデル・文法・サンプリング) を CUDA Graph に取り込み、1 回の起動で流す
    Graph は入力の形 (行数・tensors の形など) ごとに取り込んで使い回すので、同じ sampler を続けて使うほど速い。
    guide (cfg) は取り込んだときのものが使われるので、同じ sampler では同じ計算の guide を渡す。
    """

    def __init__(
        self, model: PianoARModel, tokenizer: PianoTokenizer, condition: StepCondition | None = None, graph: bool = True
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.condition = condition
        self.device = model.token_embedding.weight.device
        self.use_graph = graph and self.device.type == "cuda"
        self.caches: dict[tuple, list[tuple[Tensor, Tensor]]] = {}
        self.states: dict[tuple, _SamplerState] = {}

    @torch.no_grad()
    def sample_patch(
        self,
        context: Tensor,
        tensors: dict[str, Tensor],
        pedal_down: Tensor,
        *,
        guide: Callable[[Tensor], Tensor],
        temperature: float = 1.0,
        top_p: float = 0.95,
        finished: Tensor | None = None,
        allow_eos: bool = True,
        time_bias: Tensor | None = None,
        forced: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """1 パッチを生成する。context [rows, D] は Global の出力、tensors はこのパッチの StepCondition の量。
        guide は rows 行の logits を samples 行にまとめる (cfg)。行 r の入力はサンプル r % samples のトークン。
        pedal_down / finished [samples] はパッチの頭のペダルと、もう曲が終わった行 (PAD だけを出す)。
        time_bias [patch_frames] は TIME の logit に足す量 (_bias_time)、forced [samples, max_patch_tokens] は
        決めたトークン (-1 は決めない)。返り値は生成したトークン [samples, T] (PAD 埋め)・パッチの終わりのペダル・
        曲の終わり (EOS) を出したか"""
        state = self._state(context, tensors, pedal_down.shape[0], temperature, top_p, time_bias, forced)
        state.context.copy_(context)
        for name, value in tensors.items():
            state.tensors[name].copy_(value)
        if time_bias is not None:
            state.time_bias.copy_(time_bias)
        if forced is not None:
            state.forced.copy_(forced)
        state.allow_eos.fill_(allow_eos)

        def reset() -> None:
            state.grammar.reset(pedal_down, finished)
            state.step.zero_()
            state.token.zero_()
            state.sequence.fill_(PAD)

        def run() -> None:
            self._token_step(state, guide, temperature, top_p)

        reset()
        if self.use_graph and state.graph is None:
            # 取り込みの前に別のストリームで数回流す (PyTorch の CUDA Graph の決まった手順)。流した分の状態は戻す
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    run()
            torch.cuda.current_stream().wait_stream(side)
            reset()
            state.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(state.graph):
                run()
        for step in range(self.tokenizer.config.max_patch_tokens):
            if state.graph is not None:
                state.graph.replay()
            else:
                run()
            # 終わったかを見るのは CPU と同期するので数トークンに 1 回にする (終わった後は PAD が並ぶだけ)
            if step % 8 == 7 and bool(state.grammar.finished.all()):
                break
        grammar = state.grammar
        return state.sequence[:, : step + 1].clone(), grammar.pedal_down.clone(), grammar.song_end.clone()

    def _state(
        self,
        context: Tensor,
        tensors: dict[str, Tensor],
        samples: int,
        temperature: float,
        top_p: float,
        time_bias: Tensor | None,
        forced: Tensor | None,
    ) -> _SamplerState:
        device = self.device
        autocast = torch.is_autocast_enabled(device.type)
        dtype = torch.get_autocast_dtype(device.type) if autocast else self.model.token_embedding.weight.dtype
        rows = context.shape[0]
        shapes = tuple((name, tuple(value.shape), value.dtype) for name, value in tensors.items())
        key = (
            rows,
            samples,
            tuple(context.shape),
            shapes,
            temperature,
            top_p,
            time_bias is None,
            forced is None,
            dtype,
        )
        if key in self.states:
            return self.states[key]
        transformer = self.model.local_transformer
        length = self.tokenizer.config.max_patch_tokens
        if (rows, dtype) not in self.caches:
            heads = getattr(transformer.blocks[0], "_orig_mod", transformer.blocks[0]).heads
            shape = (rows, heads, length, transformer.head_dim)
            self.caches[rows, dtype] = [
                (torch.zeros(shape, dtype=dtype, device=device), torch.zeros(shape, dtype=dtype, device=device))
                for _ in transformer.blocks
            ]
        state = _SamplerState(
            autocast=(autocast, dtype),
            cache=self.caches[rows, dtype],
            grammar=BatchGrammar(self.tokenizer, samples, device),
            token=torch.zeros(rows, dtype=torch.long, device=device),
            step=torch.zeros(1, dtype=torch.long, device=device),
            sequence=torch.full((samples, length), PAD, dtype=torch.long, device=device),
            context=context.clone(),
            tensors={name: value.clone() for name, value in tensors.items()},
            time_bias=time_bias.clone() if time_bias is not None else None,
            forced=forced.clone() if forced is not None else None,
            allow_eos=torch.ones(1, dtype=torch.bool, device=device),
        )
        self.states[key] = state
        return state

    def _token_step(
        self, state: _SamplerState, guide: Callable[[Tensor], Tensor], temperature: float, top_p: float
    ) -> None:
        """1 トークン分: モデル -> cfg -> 文法 -> サンプリング -> 状態の更新。すべて state のテンソルをその場で書き換える"""
        grammar = state.grammar
        repeat = state.token.shape[0] // grammar.onset.shape[0]
        enabled, dtype = state.autocast
        # CUDA Graph に取り込むときは、autocast の型変換のキャッシュを切る (取り込んだ後に消えるテンソルを指さないよう)
        with torch.autocast(self.device.type, dtype=dtype, enabled=enabled, cache_enabled=False):
            onset = grammar.onset.clamp(min=0).float().repeat(repeat)
            logits = self.model.local_step(
                state.token, state.step, state.context, state.cache, onset, self.condition, state.tensors
            )
            logits = guide(logits.float())
            allowed = grammar.allowed()
            # EOP はいつも一緒に許されているので、EOS を止めてもパッチは EOP で終われる
            allowed[:, EOS] &= state.allow_eos
            logits = logits.masked_fill(~allowed, float("-inf"))
            if state.time_bias is not None:
                logits = _bias_time(logits, state.time_bias.to(logits), self.tokenizer)
            next_token = _sample(logits, temperature, top_p)
            if state.forced is not None:
                forced = state.forced.index_select(1, state.step).squeeze(1)
                next_token = torch.where((forced >= 0) & ~grammar.finished, forced, next_token)
            grammar.update(next_token)
            state.sequence.index_copy_(1, state.step, next_token[:, None])
            state.token.copy_(next_token.repeat(repeat))
            state.step += 1


@dataclass
class _SamplerState:
    """LocalSampler の入力の形ごとの置き場。CUDA Graph はこれらのテンソルを読み書きする"""

    autocast: tuple[bool, torch.dtype]
    cache: list[tuple[Tensor, Tensor]]
    grammar: BatchGrammar
    token: Tensor  # [rows] 次の位置の入力 (直前に出したトークン)
    step: Tensor  # [1] 次の位置
    sequence: Tensor  # [samples, max_patch_tokens] 出したトークン
    context: Tensor
    tensors: dict[str, Tensor]
    time_bias: Tensor | None
    forced: Tensor | None
    allow_eos: Tensor  # [1]
    graph: torch.cuda.CUDAGraph | None = None


def _bias_time(logits: Tensor, bias: Tensor, tokenizer: PianoTokenizer) -> Tensor:
    """TIME トークンの logit [N, patch_frames] に bias を足す。TIME 全体の確率は元に戻し、TIME の中の配分だけを変える。
    TIME の確率ごと上げると「同じ onset に音を足す」が選ばれにくくなって和音が薄くなる (1 onset あたり 1.75 音 -> 1.04 音)"""
    time = logits[:, tokenizer.time_offset : tokenizer.pitch_offset]
    shifted = time + bias
    before, after = time.logsumexp(-1, keepdim=True), shifted.logsumexp(-1, keepdim=True)
    # TIME を出せない位置 (すべて -inf) はそのまま
    shifted = torch.where(torch.isfinite(before), shifted - after + before, time)
    return torch.cat((logits[:, : tokenizer.time_offset], shifted, logits[:, tokenizer.pitch_offset :]), dim=1)


def _sample(logits: Tensor, temperature: float, top_p: float) -> Tensor:
    if temperature <= 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, order = probs.sort(dim=-1, descending=True)
        # 累積確率が top_p を超える手前までを残す (最も確率の高いトークンは必ず残る)
        remove = sorted_probs.cumsum(-1) - sorted_probs > top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs)
    return torch.multinomial(probs, 1).squeeze(-1)
