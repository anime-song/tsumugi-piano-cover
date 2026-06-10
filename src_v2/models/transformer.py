from __future__ import annotations

from typing import Optional

import einops
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint


def choose_low_precision_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    major_cc, _ = torch.cuda.get_device_capability()
    if major_cc >= 6:
        return torch.float16
    return torch.float32


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 5.960464477539063e-08) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l2_norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        denom = torch.maximum(l2_norm, torch.full_like(l2_norm, self.eps))
        return (x / denom) * self.scale * self.gamma


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"rotary dim must be even, got {dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim

    def _cos_sin(
        self, length: int, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(length, device=device, dtype=self.inv_freq.dtype).unsqueeze(0).expand(batch_size, -1)
        angles = torch.einsum("bt,j->btj", positions, self.inv_freq)
        cos = angles.cos().unsqueeze(1).to(dtype)
        sin = angles.sin().unsqueeze(1).to(dtype)
        return cos, sin

    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_odd * cos + x_even * sin
        return out

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, q_len, q_dim = q.shape
        _, _, k_len, k_dim = k.shape
        if q_dim != self.dim or k_dim != self.dim:
            raise ValueError(f"unexpected q/k dim for rotary embedding: {q_dim}, {k_dim}, expected {self.dim}")
        q_cos, q_sin = self._cos_sin(q_len, batch_size, q.device, q.dtype)
        k_cos, k_sin = self._cos_sin(k_len, batch_size, k.device, k.dtype)
        return self._apply_rotary(q, q_cos, q_sin), self._apply_rotary(k, k_cos, k_sin)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_size_factor: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = int(dim * hidden_size_factor)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = num_heads * head_dim
        self.norm_q = RMSNorm(input_dim)
        self.norm_context = RMSNorm(input_dim)
        self.to_q = nn.Linear(input_dim, self.hidden_size, bias=True)
        self.to_k = nn.Linear(input_dim, self.hidden_size, bias=True)
        self.to_v = nn.Linear(input_dim, self.hidden_size, bias=True)
        self.to_gates = nn.Linear(input_dim, num_heads)
        self.to_out = nn.Sequential(nn.Linear(self.hidden_size, input_dim), nn.Dropout(dropout))
        self.rope = RotaryEmbedding(head_dim) if use_rope else None
        self.lowp_dtype = choose_low_precision_dtype()

    @staticmethod
    def _expand_mask(mask: torch.Tensor, query_length: int, key_length: int) -> torch.Tensor:
        if mask.dtype != torch.bool:
            mask = mask != 0
        if mask.dim() == 2:
            return mask.unsqueeze(1).unsqueeze(1).expand(-1, 1, query_length, key_length)
        if mask.dim() == 3:
            return mask.unsqueeze(1)
        if mask.dim() == 4:
            return mask
        raise ValueError(f"unsupported attention mask shape: {tuple(mask.shape)}")

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        attention_bias_alpha: Optional[torch.Tensor | float] = None,
    ) -> torch.Tensor:
        if context is None:
            context = x

        q = self.to_q(self.norm_q(x))
        k = self.to_k(self.norm_context(context))
        v = self.to_v(self.norm_context(context))

        q = einops.rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)
        k = einops.rearrange(k, "b t (h d) -> b h t d", h=self.num_heads)
        v = einops.rearrange(v, "b t (h d) -> b h t d", h=self.num_heads)

        if self.rope is not None:
            q, k = self.rope(q, k)

        q = q.to(self.lowp_dtype)
        k = k.to(self.lowp_dtype)
        v = v.to(self.lowp_dtype)

        attn_mask = None
        if attention_bias is not None:
            if attention_mask is not None or is_causal:
                raise ValueError("attention_bias only supports non-causal attention without an explicit attention mask")
            bias = attention_bias.to(device=q.device, dtype=q.dtype)
            alpha = 1.0 if attention_bias_alpha is None else attention_bias_alpha
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(float(alpha), device=q.device, dtype=q.dtype)
            attn_mask = (alpha * bias).unsqueeze(0).unsqueeze(0)
        elif attention_mask is not None:
            attn_mask = self._expand_mask(attention_mask.to(device=q.device), q.shape[2], k.shape[2])

        if is_causal and attn_mask is not None:
            causal_mask = torch.ones((q.shape[2], k.shape[2]), device=q.device, dtype=torch.bool).tril()
            attn_mask = attn_mask & causal_mask.unsqueeze(0).unsqueeze(0)
            is_causal = False

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            fetched = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)

        gates = self.to_gates(self.norm_q(x)).sigmoid()
        out = fetched.float() * einops.rearrange(gates, "b t h -> b h t 1")
        out = einops.rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        head_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_hidden_size_factor: int = 4,
        dropout: float = 0.0,
        use_cross_attention: bool = False,
        output_norm: bool = False,
    ) -> None:
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.gradient_checkpointing = False
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            blocks = [
                MultiHeadAttention(
                    input_dim=input_dim,
                    head_dim=head_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_rope=True,
                )
            ]
            if use_cross_attention:
                blocks.append(
                    MultiHeadAttention(
                        input_dim=input_dim,
                        head_dim=head_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        use_rope=False,
                    )
                )
            blocks.append(FeedForward(dim=input_dim, hidden_size_factor=ffn_hidden_size_factor, dropout=dropout))
            self.layers.append(nn.ModuleList(blocks))

        self.norm = RMSNorm(input_dim) if output_norm else nn.Identity()

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = enabled

    def _forward_layer(
        self,
        blocks: nn.ModuleList,
        x: torch.Tensor,
        context: Optional[torch.Tensor],
        causal_self_attn: bool,
        attention_mask: Optional[torch.Tensor],
        context_attention_mask: Optional[torch.Tensor],
        attention_bias: Optional[torch.Tensor],
        attention_bias_alpha: Optional[torch.Tensor | float],
    ) -> torch.Tensor:
        residual = x
        x = blocks[0](
            x,
            is_causal=causal_self_attn,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            attention_bias_alpha=attention_bias_alpha,
        )
        x = x + residual

        next_index = 1
        if self.use_cross_attention:
            if context is None:
                raise ValueError("context is required when use_cross_attention=True")
            residual = x
            x = blocks[next_index](x, context=context, attention_mask=context_attention_mask)
            x = x + residual
            next_index += 1

        residual = x
        x = blocks[next_index](x) + residual
        return x

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        causal_self_attn: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        context_attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        attention_bias_alpha: Optional[torch.Tensor | float] = None,
        return_hiddens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states: list[torch.Tensor] = []
        use_gradient_checkpointing = self.gradient_checkpointing and self.training and torch.is_grad_enabled()

        for blocks in self.layers:
            if use_gradient_checkpointing:
                def custom_forward(hidden: torch.Tensor) -> torch.Tensor:
                    return self._forward_layer(
                        blocks=blocks,
                        x=hidden,
                        context=context,
                        causal_self_attn=causal_self_attn,
                        attention_mask=attention_mask,
                        context_attention_mask=context_attention_mask,
                        attention_bias=attention_bias,
                        attention_bias_alpha=attention_bias_alpha,
                    )

                x = checkpoint(custom_forward, x, use_reentrant=False)
            else:
                x = self._forward_layer(
                    blocks=blocks,
                    x=x,
                    context=context,
                    causal_self_attn=causal_self_attn,
                    attention_mask=attention_mask,
                    context_attention_mask=context_attention_mask,
                    attention_bias=attention_bias,
                    attention_bias_alpha=attention_bias_alpha,
                )
            if return_hiddens:
                hidden_states.append(x)

        x = self.norm(x)
        if return_hiddens:
            return x, hidden_states
        return x
