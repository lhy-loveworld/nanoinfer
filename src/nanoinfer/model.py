"""Stage 1 — the core modeling substrate. [REFERENCE SOLUTION]

See the `master` branch for the skeleton + full specs. This branch implements the
bodies so you can diff against your own attempt.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, nkv, T, hd) -> (B, nkv*n_rep, T, hd), each KV head repeated n_rep times."""
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
) -> torch.Tensor:
    """Hand-rolled scaled dot-product attention. Handles Tq != Tk."""
    hd = q.size(-1)
    Tq, Tk = q.size(-2), k.size(-2)
    att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)  # (B, nh, Tq, Tk)
    if causal:
        # align the last query with the last key: query i sees keys j <= (Tk-Tq)+i
        q_idx = torch.arange(Tq, device=q.device).view(Tq, 1)
        k_idx = torch.arange(Tk, device=q.device).view(1, Tk)
        allowed = k_idx <= (Tk - Tq) + q_idx
        att = att.masked_fill(~allowed, float("-inf"))
    att = F.softmax(att, dim=-1)
    return att @ v


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.nh = config.n_head
        self.nkv = config.n_kv_head
        self.hd = config.head_dim
        q_dim = config.n_head * config.head_dim
        kv_dim = config.n_kv_head * config.head_dim
        self.c_attn = nn.Linear(config.n_embd, q_dim + 2 * kv_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q_dim = self.nh * self.hd
        kv_dim = self.nkv * self.hd
        q, k, v = self.c_attn(x).split([q_dim, kv_dim, kv_dim], dim=-1)
        q = q.view(B, T, self.nh, self.hd).transpose(1, 2)   # (B, nh, T, hd)
        k = k.view(B, T, self.nkv, self.hd).transpose(1, 2)  # (B, nkv, T, hd)
        v = v.view(B, T, self.nkv, self.hd).transpose(1, 2)
        k = repeat_kv(k, self.config.n_rep)
        v = repeat_kv(v, self.config.n_rep)
        y = attention(q, k, v, causal=True)                  # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=False)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=False)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


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
        if config.tie_weights:
            self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)
