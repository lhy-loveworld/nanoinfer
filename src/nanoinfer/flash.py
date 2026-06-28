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

    The crux is the online-softmax recurrence: a running max, running denominator,
    and running output accumulator that get rescaled whenever a later key block
    raises the running max. Get that rescaling right and your output matches a
    single full softmax exactly.

    Stuck? See HINTS.md (flash.flash_attention).
    """
    raise NotImplementedError
