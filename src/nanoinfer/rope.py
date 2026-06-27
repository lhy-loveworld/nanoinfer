"""Stage 1+ — Rotary Position Embeddings (RoPE). [REFERENCE SOLUTION]

See the `master` branch for the skeleton + full specs and integration notes.
"""
from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[x1, x2] -> [-x2, x1] over the last dim (split into halves)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device=None,
    dtype: torch.dtype = torch.float32,
):
    assert head_dim % 2 == 0, "RoPE requires an even head_dim"
    # one frequency per dimension-pair: theta_i = base^(-2i/hd)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device,
                                            dtype=torch.float32) / head_dim))  # (hd/2,)
    t = torch.arange(seq_len, device=device, dtype=torch.float32)              # (seq_len,)
    angles = torch.outer(t, inv_freq)                                         # (seq_len, hd/2)
    emb = torch.cat([angles, angles], dim=-1)                                 # (seq_len, hd)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x * cos + rotate_half(x) * sin  (cos/sin broadcast over x's leading dims)."""
    return x * cos + rotate_half(x) * sin
