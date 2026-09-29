"""Global Patch -> Local Decoder のピアノ生成モデル (NotaGen 型)。

    パッチ p-1 のトークン --PatchSummarizer--> 要約ベクトル
    Global (因果): 位置 p の入力 = 要約(p-1) (p=0 は BOS) + ペダル状態(p) + チャンネル  -> h_p
    Local  (因果): h_p を全位置に足して、パッチ p のトークン列を 1 つずつ生成する

Global の入力は生成済みパッチの中身そのものなので、Local の出力は要約を通じて次のパッチに渡る。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .tokenizer import PAD, TOKEN_GROUPS, PatchGrammar, PianoTokenizer

# Transformer のブロックの後に挟む処理 (ブロック番号, x) -> x。カバーモデルの cross-attention に使う
CrossHook = Callable[[int, Tensor], Tensor]


class Condition(Protocol):
    """学習時に外から条件 (原曲など) を cross-attention で入れるためのフック"""

    def global_cross(self) -> CrossHook: ...

    def local_cross(self, index: Tensor, prefix: Tensor) -> CrossHook:
        """index は有効なパッチを並べたときの番号、prefix はそのパッチの Local の入力トークン [N, T]"""
        ...


class GenerationCondition(Protocol):
    def global_cross(self, first: int, last: int) -> CrossHook:
        """Global にパッチ first..last を入れるときのフック"""
        ...

    def local_cross(self, patch: int, prefix: Tensor) -> CrossHook: ...


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
        batch, length, _ = x.shape
        head_dim = self.inner_dim // self.heads
        q = self.q(self.norm(x)).view(batch, length, self.heads, head_dim).transpose(1, 2)
        k, v = self.kv(self.memory_norm(memory)).view(batch, -1, 2, self.heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
        q, k = _apply_rope(q, *_rope(q_pos, head_dim)), _apply_rope(k, *_rope(k_pos, head_dim))
        # 空のキーは位置によらないよう回転をかけない
        null_k = self.k_norm(self.null_k).to(v.dtype)[None, :, None].expand(batch, -1, -1, -1)
        null_v = self.null_v.to(v.dtype)[None, :, None].expand(batch, -1, -1, -1)
        k, v = torch.cat((null_k, k), dim=2), torch.cat((null_v, v), dim=2)
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
        self.global_transformer = Transformer(config, config.global_layers, causal=True)

        self.local_bos = nn.Parameter(torch.randn(dim) * 0.02)
        self.local_context = nn.Linear(dim, dim)
        self.local_transformer = Transformer(config, config.local_layers, causal=True)
        self.head = nn.Linear(dim, self.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

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
    ) -> Tensor:
        """summaries[:, p] はパッチ p の要約。位置 p にはパッチ p-1 の要約を入れて h を返す"""
        inputs = torch.cat((self.bos(song_start)[:, None], summaries[:, :-1]), dim=1)
        inputs = inputs + self.pedal_embedding(pedal_state) + self.channel_embedding(channel)[:, None]
        return self.global_transformer(inputs, cross=cross)

    def local_forward(self, prefix: Tensor, context: Tensor, cross: CrossHook | None = None) -> Tensor:
        """prefix [N, T] の続きを予測する logits [N, T+1, V] を返す"""
        bos = self.local_bos.to(self.token_embedding.weight.dtype).expand(prefix.shape[0], 1, -1)
        x = torch.cat((bos, self.token_embedding(prefix)), dim=1) + self.local_context(context)[:, None]
        return self.head(self.local_transformer(x, cross=cross))

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
        # ブロックの種類 x 学習/評価 x マスクの有無、生成時の長さ 1 の入力などで作り直しが起きる。
        # 既定の上限 (8) を超えると以降は compile されずに通常実行になるので、余裕を持たせる
        torch._dynamo.config.recompile_limit = 64
        torch._dynamo.config.cache_size_limit = 64
        for module in self.modules():
            if isinstance(module, (Block, CrossBlock)):
                module.compile(dynamic=True)

    def forward(
        self, batch: dict[str, Tensor], num_length_buckets: int = 8, condition: Condition | None = None
    ) -> dict[str, Tensor]:
        """学習時の損失。

        パッチごとのトークン長は中央値 70 前後に対して最長は 200 を超えるので、全パッチを最長に揃えると
        計算とメモリの大半がパディングに使われる。パッチを長さ順に num_length_buckets 個の塊に分け、
        塊ごとにその中の最長まで切り詰めて PatchSummarizer と Local Decoder を通す。
        """
        tokens = batch["tokens"]
        valid = batch["patch_valid"]
        flat = tokens[valid]  # [N, L] 有効なパッチだけ
        lengths = (flat != PAD).sum(-1)
        order = lengths.argsort()
        buckets = [
            (index, int(lengths[index].max()))
            for index in order.tensor_split(min(num_length_buckets, len(order)))
            if len(index)
        ]

        summarized = torch.cat([self.summarize_patches(flat[index, :length]) for index, length in buckets])
        summarized = summarized[order.argsort()]
        summaries = summarized.new_zeros(*valid.shape, self.config.dim)
        summaries[valid] = summarized
        context = self.global_forward(
            summaries,
            batch["song_start"],
            batch["pedal_state"],
            batch["channel"],
            cross=condition.global_cross() if condition is not None else None,
        )[valid]

        loss_sum = torch.zeros(len(self.group_names), device=tokens.device)
        counts = torch.zeros(len(self.group_names), device=tokens.device)
        for index, length in buckets:
            target = flat[index, :length]
            cross = condition.local_cross(index, target[:, :-1]) if condition is not None else None
            logits = self.local_forward(target[:, :-1], context[index], cross)
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
    ) -> list[list[list[int]]]:
        """曲の冒頭から生成し、サンプルごとにパッチのトークン列のリストを返す。

        context_patches を超えたら、学習時の「途中から始まる窓」と同じ形 (BOS(途中) + 直近のパッチ) で続ける。
        cfg_scale > 1 ならチャンネル指定なしとの差を強調する classifier-free guidance を使う。
        prompts[i] (パッチごとのトークン列) を渡すと、サンプル i の先頭のパッチはそのトークンで埋めて続きを生成する。
        condition (原曲など) を渡すと cross-attention で条件を入れる。

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
            context = self.global_forward(stacked, song_start_tensor, pedal, channels, global_cross)[:, -1]

            grammars = [PatchGrammar(tokenizer, pedal_states[i]) for i in range(num_samples)]
            for i in range(num_samples):
                if done[i]:
                    grammars[i].finished = True
            forced = [
                prompts[i][p] if prompts is not None and p < len(prompts[i]) else None for i in range(num_samples)
            ]
            sequence = torch.zeros(rows, 0, dtype=torch.long, device=device)
            while not all(g.finished for g in grammars):
                local_cross = bound.local_cross(p, sequence) if bound is not None else None
                logits = self.local_forward(sequence, context, local_cross)[:, -1].float()
                logits = guide(logits)
                allowed = torch.stack([g.allowed() for g in grammars]).to(device)
                next_token = _sample(logits.masked_fill(~allowed, float("-inf")), temperature, top_p)
                step = sequence.shape[1]
                for i, prompt in enumerate(forced):
                    if prompt is not None and not grammars[i].finished:
                        next_token[i] = prompt[step]
                for i, g in enumerate(grammars):
                    g.update(int(next_token[i]))
                sequence = torch.cat((sequence, next_token.repeat(rows // num_samples)[:, None]), dim=1)

            for i, g in enumerate(grammars):
                if done[i]:
                    continue
                tokens = [t for t in sequence[i].tolist() if t != PAD]
                patches[i].append(tokens)
                pedal_states[i] = g.pedal_down
                done[i] = g.song_end
            summaries.append(self.summarize_patches(sequence))
        return patches


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
