"""Stage 1 — the core modeling substrate.

You implement the forward passes below. Everything in later stages (KV cache,
sampling, batching, speculative decoding) is a *transformation* of this code, so
get it right and keep it readable.

Layer modules (`__init__`s) are given so weight shapes/names are fixed and the
tests are stable. Your job is the math in each `forward` and in `attention()`.

Tensor shape conventions used throughout:
    B  = batch size
    T  = sequence length (number of query positions)
    C  = n_embd (residual stream width)
    nh = n_head, nkv = n_kv_head, hd = head_dim
    K/V time dimension may differ from T once a KV cache is involved (Stage 2),
    so write `attention` to handle Tq (query len) != Tk (key len).

Run the tests for this stage with:
    pytest tests/test_attention.py tests/test_model.py
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig
from .rope import apply_rope, build_rope_cache


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match the number of query heads (GQA).

    Args:
        x: key or value tensor of shape (B, nkv, T, hd)
        n_rep: how many query heads share each KV head (config.n_rep)

    Returns:
        Tensor of shape (B, nkv * n_rep, T, hd), where head group g is repeated
        n_rep times contiguously so it lines up with the query head layout.
        When n_rep == 1 (standard multi-head attention) this is a no-op.

    Stuck? See HINTS.md (model.repeat_kv).
    """
    raise NotImplementedError


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
) -> torch.Tensor:
    """Scaled dot-product attention — implement it by hand (no F.sdpa here).

    The tests compare your output against torch.nn.functional.scaled_dot_product_attention,
    so this is where you prove you know the actual computation.

    Args:
        q: (B, nh, Tq, hd)   queries
        k: (B, nh, Tk, hd)   keys   (already repeated to nh heads via repeat_kv)
        v: (B, nh, Tk, hd)   values (already repeated to nh heads)
        causal: if True, query position i may only attend to key positions
            j <= j0 + i, where j0 = Tk - Tq aligns the last query with the last
            key. This alignment is what makes incremental decoding (Tq=1, Tk=cache
            length) attend to the whole cache — keep it in mind for Stage 2.

    Returns:
        (B, nh, Tq, hd) attention output.

    Stuck? See HINTS.md (model.attention).
    """
    raise NotImplementedError


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.nh = config.n_head
        self.nkv = config.n_kv_head
        self.hd = config.head_dim
        # Fused QKV projection. Q gets nh*hd cols; K and V each get nkv*hd cols.
        q_dim = config.n_head * config.head_dim
        kv_dim = config.n_kv_head * config.head_dim
        self.c_attn = nn.Linear(config.n_embd, q_dim + 2 * kv_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        if config.rope:
            # Precompute the rotary cos/sin tables and register them as BUFFERS
            # (not plain attributes) so model.to(device)/.half() move them too.
            # persistent=False keeps them out of the state_dict (recomputable).
            cos, sin = build_rope_cache(config.max_seq_len, config.head_dim)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention over the whole sequence (no cache yet — that's Stage 2).

        Args:
            x: (B, T, C)
        Returns:
            (B, T, C)

        Stuck? See HINTS.md (model.CausalSelfAttention.forward).
        """
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Position-wise feed-forward: c_fc -> GELU -> c_proj. Shapes unchanged."""
        raise NotImplementedError


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=False)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=False)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-norm transformer block with residual connections:

            x = x + attn(ln_1(x))
            x = x + mlp(ln_2(x))
        """
        raise NotImplementedError


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=False),
        ))
        if not config.rope:
            # learned absolute position table; with RoPE there is no wpe — the
            # attention layers inject position by rotating q/k instead.
            self.transformer["wpe"] = nn.Embedding(config.block_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_weights:
            # share one matrix between input embedding and output projection
            self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Full forward pass over a batch of token ids.

        Args:
            idx: (B, T) int64 token ids
        Returns:
            logits: (B, T, vocab_size)

        Position handling becomes interesting in Stage 2 when you decode one
        token at a time — the position of the new token is the cache length, not
        zero. Write this now in a way you'll be able to generalize.

        Stuck? See HINTS.md (model.GPT.forward).
        """
        raise NotImplementedError
