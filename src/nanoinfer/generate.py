"""Stage 3 — sampling and the decode loop. [REFERENCE SOLUTION]"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import GPTConfig
from .kv_cache import KVCache, decode_step
from .model import GPT


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / temperature


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k >= logits.size(-1):
        return logits
    kth = torch.topk(logits, k, dim=-1).values[..., -1, None]  # kth largest per row
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cum_prev = probs.cumsum(dim=-1) - probs  # exclusive prefix sum
    # remove a token if everything *before* it already reached p (keeps >=1 token,
    # since the top token's cum_prev is always 0 < p)
    remove_sorted = cum_prev >= p
    remove = torch.zeros_like(remove_sorted)
    remove.scatter_(-1, sorted_idx, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


def sample_next(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    logits = apply_temperature(logits, temperature)
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    B, T0 = idx.shape
    cfg = model.config
    total = T0 + max_new_tokens
    dtype = next(model.parameters()).dtype
    cache = KVCache(cfg, batch_size=B, max_seq_len=total, device=idx.device, dtype=dtype)

    seq = idx
    logits = decode_step(model, idx, cache, start_pos=0)  # prefill
    for _ in range(max_new_tokens):
        nxt = sample_next(
            logits[:, -1], temperature=temperature, top_k=top_k, top_p=top_p,
            generator=generator,
        )  # (B,)
        seq = torch.cat([seq, nxt[:, None]], dim=1)
        logits = decode_step(model, nxt[:, None], cache, start_pos=cache.length)
    return seq
