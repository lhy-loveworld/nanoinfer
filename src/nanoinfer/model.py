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


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match the number of query heads (GQA).

    Args:
        x: key or value tensor of shape (B, nkv, T, hd)
        n_rep: how many query heads share each KV head (config.n_rep)

    Returns:
        Tensor of shape (B, nkv * n_rep, T, hd), where head group g is repeated
        n_rep times contiguously so it lines up with the query head layout.

    Implement this without allocating more memory than necessary where you can
    (hint: torch.expand + reshape, or repeat_interleave on the head dim). When
    n_rep == 1 (standard multi-head attention) this should be a no-op.
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

    Steps: scores = q @ k^T / sqrt(hd)  ->  apply causal mask  ->  softmax over
    the key axis  ->  weighted sum with v.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention over the whole sequence (no cache yet — that's Stage 2).

        Args:
            x: (B, T, C)
        Returns:
            (B, T, C)

        Steps:
            1. project x with c_attn, split into q (nh*hd), k (nkv*hd), v (nkv*hd)
            2. reshape each into (B, heads, T, hd) — note q has nh heads, k/v nkv
            3. expand k, v to nh heads with repeat_kv
            4. y = attention(q, k, v, causal=True)
            5. reshape y back to (B, T, C) and apply c_proj
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
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=False),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying (standard GPT trick)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Full forward pass over a batch of token ids.

        Args:
            idx: (B, T) int64 token ids, with T <= config.block_size
        Returns:
            logits: (B, T, vocab_size)

        Steps:
            1. token embeddings wte(idx) + position embeddings wpe(positions),
               where positions = [0, 1, ..., T-1]
            2. run through each block in self.transformer.h
            3. final layernorm ln_f
            4. project to vocab with lm_head

        (Position handling becomes interesting in Stage 2 when you decode one
        token at a time — the position of the new token is the cache length, not
        zero. Write this now in a way you'll be able to generalize.)
        """
        raise NotImplementedError
