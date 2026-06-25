"""Stage 5c — tiled ("flash"-style) attention with online softmax. [REFERENCE SOLUTION]"""
from __future__ import annotations

import math

import torch


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    *, causal: bool = True, block_size: int = 64) -> torch.Tensor:
    B, nh, T, hd = q.shape
    scale = 1.0 / math.sqrt(hd)
    out = torch.zeros_like(q)

    for qi in range(0, T, block_size):
        qe = min(qi + block_size, T)
        bq = qe - qi
        Qi = q[:, :, qi:qe]                                   # (B, nh, bq, hd)
        m = torch.full((B, nh, bq, 1), float("-inf"), device=q.device, dtype=q.dtype)
        l = torch.zeros((B, nh, bq, 1), device=q.device, dtype=q.dtype)
        acc = torch.zeros((B, nh, bq, hd), device=q.device, dtype=q.dtype)

        for ki in range(0, T, block_size):
            ke = min(ki + block_size, T)
            if causal and ki > qe - 1:
                break  # this key block is entirely in the future for every query here
            Kj = k[:, :, ki:ke]
            Vj = v[:, :, ki:ke]
            S = (Qi @ Kj.transpose(-2, -1)) * scale          # (B, nh, bq, bk)
            if causal:
                q_idx = torch.arange(qi, qe, device=q.device).view(bq, 1)
                k_idx = torch.arange(ki, ke, device=q.device).view(1, ke - ki)
                S = S.masked_fill(k_idx > q_idx, float("-inf"))

            m_new = torch.maximum(m, S.amax(dim=-1, keepdim=True))
            alpha = torch.exp(m - m_new)                     # rescale prior partials
            p = torch.exp(S - m_new)                         # (B, nh, bq, bk)
            l = l * alpha + p.sum(dim=-1, keepdim=True)
            acc = acc * alpha + p @ Vj
            m = m_new

        out[:, :, qi:qe] = acc / l

    return out
