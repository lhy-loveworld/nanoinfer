"""Stage 3 — sampling and the decode loop.

Sampling is where serving meets product behavior: temperature, top-k, and top-p
(nucleus) all reshape the next-token distribution before you draw from it. Get
the filtering semantics exactly right — off-by-one nucleus bugs are a classic.

All logit filters operate on the last dim (vocab) and return logits (with
removed entries set to -inf) so a single softmax downstream stays correct.

Run the tests with:
    pytest tests/test_generate.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import GPTConfig
from .kv_cache import KVCache, decode_step
from .model import GPT


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by 1/temperature.

    Args:
        logits: (..., vocab)
        temperature: > 0. Higher = flatter/more random, lower = sharper.
    Returns:
        logits / temperature. (Temperature == 0 is handled by the caller as
        greedy/argmax — you may assume temperature > 0 here.)
    """
    raise NotImplementedError


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the k highest-logit entries; set the rest to -inf.

    Args:
        logits: (B, vocab)
        k: number of entries to keep (k >= 1). If k >= vocab, return logits as-is.
    Returns:
        (B, vocab) with all but the top-k entries (per row) set to -inf.
    """
    raise NotImplementedError


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest set of tokens whose cumulative
    probability mass exceeds p; set the rest to -inf.

    Args:
        logits: (B, vocab)
        p: cumulative probability threshold in (0, 1].
    Returns:
        (B, vocab) filtered logits.

    Semantics (match these exactly — the tests check the boundary):
        - sort probabilities descending
        - include tokens until the cumulative sum first reaches/exceeds p
        - ALWAYS keep at least the single most probable token (even if it alone
          already exceeds p)
    """
    raise NotImplementedError


def sample_next(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pick the next token id for each row of `logits`.

    Args:
        logits: (B, vocab) raw logits for the next position.
        temperature: 0.0 means greedy (argmax). Otherwise scale then sample.
        top_k / top_p: optional filters applied (in that order) before sampling.
        generator: optional torch.Generator for reproducible sampling.
    Returns:
        (B,) int64 token ids.

    Order of operations: greedy short-circuit -> temperature -> top_k -> top_p
    -> softmax -> multinomial sample (use the generator).
    """
    raise NotImplementedError


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
    """Autoregressively extend `idx` by `max_new_tokens`, using the KV cache.

    Args:
        idx: (B, T0) prompt token ids.
        max_new_tokens: how many tokens to append.
    Returns:
        (B, T0 + max_new_tokens) token ids.

    Sketch:
        1. allocate a KVCache sized for T0 + max_new_tokens
        2. prefill: logits = decode_step(model, idx, cache, start_pos=0)
        3. loop max_new_tokens times:
             - next_tok = sample_next(logits[:, -1], ...)   # (B,)
             - append next_tok to the running sequence
             - logits = decode_step(model, next_tok[:, None], cache, start_pos=cache.length)
        4. return the full sequence

    This is the function the batching stage (Stage 4) generalizes to many
    sequences of different lengths sharing the model.
    """
    raise NotImplementedError
