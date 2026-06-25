"""Stage 5c — tiled ("flash"-style) attention with online softmax.

Naive attention materializes the full (T, T) scores matrix — O(T^2) memory and a
lot of HBM traffic. FlashAttention never materializes it: it streams over blocks
of keys/values, keeping a running softmax (running max + running denominator) and
a running weighted output, so memory is O(T) and the data stays in fast SRAM.

You won't write CUDA here — you implement the *algorithm* in PyTorch (looping
over K/V blocks) so you understand the online-softmax recurrence. Correctness is
checked against F.scaled_dot_product_attention; a GPU perf test compares wall
time against the naive O(T^2) version on a long sequence.

Run the tests with:
    pytest tests/test_flash.py
    pytest tests/test_flash.py --device cuda -m perf   # timing on the 5080
"""
from __future__ import annotations

import math

import torch


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    *, causal: bool = True, block_size: int = 64) -> torch.Tensor:
    """Tiled attention via the online-softmax recurrence.

    Args:
        q, k, v: (B, nh, T, hd). Assume q, k, v share the same T here.
        causal: standard causal masking (query i attends to keys j <= i).
        block_size: number of key/value positions processed per tile.

    Returns:
        (B, nh, T, hd), numerically equal (within tolerance) to softmax attention.

    Algorithm (per query block, streaming over key blocks):
        Maintain for each query row: m (running max logit, init -inf),
        l (running sum of exp, init 0), acc (running output accumulator, init 0).
        For each key/value block (Kj, Vj):
            S   = (Qi @ Kj^T) / sqrt(hd)              # (block_q, block_k)
            apply causal mask to S where needed (skip fully-masked future blocks)
            m_new = max(m, rowmax(S))
            p   = exp(S - m_new)                      # rescaled probabilities
            l   = l * exp(m - m_new) + rowsum(p)
            acc = acc * exp(m - m_new)[:, None] + p @ Vj
            m   = m_new
        Output_i = acc / l[:, None]

    The exp(m_old - m_new) rescaling is the crux — it retroactively corrects the
    earlier partial sums when a later block raises the running max. Get that right
    and your output matches a single full softmax exactly.
    """
    raise NotImplementedError
