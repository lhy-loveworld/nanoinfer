"""Stage 1+ — Rotary Position Embeddings (RoPE).

Learned absolute position embeddings (the `wpe` table) have two problems for an
inference engineer: they hard-cap the context at `block_size` (you hit exactly
this during training), and they inject position additively into the residual
stream. RoPE instead *rotates* each query and key vector by an angle
proportional to its absolute position, using a different frequency per
dimension-pair. The magic: the dot product of a rotated query at position m and
a rotated key at position n depends only on the *relative* offset (m - n) — which
is what attention actually cares about — and it extrapolates beyond the trained
length far more gracefully.

You implement three pieces:
    rotate_half      — the (-x2, x1) half-rotation helper
    build_rope_cache — precompute cos/sin tables for positions 0..seq_len-1
    apply_rope       — rotate a (..., T, hd) tensor using the cache

Convention (LLaMA / HF "NeoX" style): pair dim i with dim i + hd/2 (the two
halves), not adjacent dims. `hd` must be even.

How it would integrate (NOT needed to pass these tests, and intentionally not
wired into model.py so you can do it yourself):
    - drop `wpe` from GPT; build ONE cache of length block_size up front
    - inside attention, after the head reshape and BEFORE the q@k^T score, do
      q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)   # v is NOT rotated
    - with a KV cache (Stage 2), slice cos/sin at the token's ABSOLUTE position
      (start_pos .. start_pos+T) — same offset logic as the position ids.

Run the tests with:
    pytest tests/test_rope.py
"""
from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split the last dim into halves and rotate: [x1, x2] -> [-x2, x1].

    Args:
        x: (..., hd) with hd even.
    Returns:
        (..., hd) with the first half replaced by -(second half) and the second
        half replaced by the first half. Combined with cos/sin in apply_rope,
        this realizes a 2D rotation of each (i, i+hd/2) dimension pair.
    """
    raise NotImplementedError


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device=None,
    dtype: torch.dtype = torch.float32,
):
    """Precompute the cos/sin rotation tables for positions 0..seq_len-1.

    Args:
        seq_len: number of positions to precompute.
        head_dim: per-head dimension hd (must be even).
        base: rotary base theta (10000 is standard).
    Returns:
        (cos, sin), each of shape (seq_len, head_dim).

    Steps:
        1. inv_freq = 1 / base ** (arange(0, head_dim, 2) / head_dim)    # (hd/2,)
        2. angles   = outer(arange(seq_len), inv_freq)                   # (seq_len, hd/2)
        3. emb      = cat([angles, angles], dim=-1)                      # (seq_len, hd)
           (duplicated so dim i and dim i+hd/2 share a frequency — that's the
            half-split pairing rotate_half assumes)
        4. return emb.cos(), emb.sin()
    Compute angles in float32 for precision, then cast to `dtype` at the end.
    """
    raise NotImplementedError


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate x by the precomputed angles.

    Args:
        x: (..., T, hd) queries or keys, already split into heads.
        cos, sin: (T, hd) from build_rope_cache, sliced to x's positions. They
            broadcast over x's leading dims (B, nh, ...).
    Returns:
        (..., T, hd) = x * cos + rotate_half(x) * sin
    """
    raise NotImplementedError
