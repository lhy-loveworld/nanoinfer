"""Stage 5a — speculative decoding.

A small/cheap *draft* model proposes K tokens autoregressively; the large
*target* model verifies all K in a single forward pass (plus one bonus token).
You accept the longest prefix the target agrees with and resample at the first
disagreement. Net effect: up to K+1 tokens per target forward instead of 1,
with output drawn from the *target's* distribution — a latency win for free
quality.

We implement the GREEDY variant first because it has an exact, testable
invariant: speculative-greedy output is bit-identical to plain greedy decoding
of the target. (The stochastic variant of Leviathan et al. 2023 — accept token
x with prob min(1, p_target(x)/p_draft(x)), else resample from the residual —
is noted at the bottom as an extension.)

Run the tests with:
    pytest tests/test_speculative.py
"""
from __future__ import annotations

import torch

from .model import GPT


@torch.no_grad()
def draft_k(draft: GPT, idx: torch.Tensor, k: int) -> torch.Tensor:
    """Greedily extend idx by k tokens using the draft model.

    Args:
        idx: (1, T) current sequence
        k: number of speculative tokens to propose
    Returns:
        (1, k) the proposed token ids (greedy argmax at each step).
    You may use plain forward passes (no cache needed for this exercise).
    """
    raise NotImplementedError


@torch.no_grad()
def verify(target: GPT, idx: torch.Tensor, proposed: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Verify proposed tokens against the target in ONE forward pass.

    Args:
        idx: (1, T) the accepted sequence so far.
        proposed: (1, k) draft's proposed continuation.
    Returns:
        (accepted_tokens, n_accepted) where accepted_tokens is (1, n_accepted+1):
        the matched prefix PLUS one bonus token, following greedy rules below.

    Greedy acceptance:
        - Run target on torch.cat([idx, proposed], dim=1). The logits at position
          (T-1+i) predict the token that should follow proposed[:i] — i.e. they
          give the target's greedy choice for proposal slot i.
        - Walk i = 0..k-1: if target_greedy[i] == proposed[i], accept it and
          continue; at the FIRST mismatch, emit target_greedy[i] (the corrected
          token) and stop.
        - If all k match, emit one bonus token = target's greedy choice at the
          last position. Either way you return between 1 and k+1 tokens.
    """
    raise NotImplementedError


@torch.no_grad()
def speculative_generate(target: GPT, draft: GPT, idx: torch.Tensor,
                         max_new_tokens: int, k: int = 4) -> torch.Tensor:
    """Generate with draft+verify until at least max_new_tokens are produced.

    Args:
        idx: (1, T0) prompt.
        max_new_tokens: minimum number of new tokens to produce (you may overshoot
            within a verification round; truncate the return to exactly this many).
        k: speculation length.
    Returns:
        (1, T0 + max_new_tokens). For greedy decoding this MUST equal plain greedy
        decoding of the target (that's the Stage-5 test).

    Loop: propose k with draft_k -> verify against target -> append accepted
    tokens -> repeat until enough tokens, then truncate.
    """
    raise NotImplementedError
