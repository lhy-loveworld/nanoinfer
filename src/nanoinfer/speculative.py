"""Stage 5a — speculative decoding (greedy variant). [REFERENCE SOLUTION]"""
from __future__ import annotations

import torch

from .model import GPT


@torch.no_grad()
def draft_k(draft: GPT, idx: torch.Tensor, k: int) -> torch.Tensor:
    seq = idx
    out = []
    for _ in range(k):
        nxt = draft(seq)[:, -1].argmax(dim=-1, keepdim=True)
        out.append(nxt)
        seq = torch.cat([seq, nxt], dim=1)
    return torch.cat(out, dim=1)  # (1, k)


@torch.no_grad()
def verify(target: GPT, idx: torch.Tensor, proposed: torch.Tensor) -> tuple[torch.Tensor, int]:
    T = idx.shape[1]
    k = proposed.shape[1]
    full = torch.cat([idx, proposed], dim=1)          # (1, T+k)
    logits = target(full)                              # (1, T+k, V)
    # target's greedy choice for proposal slot i is at position T-1+i
    greedy = logits[:, T - 1:T - 1 + k].argmax(dim=-1)  # (1, k)

    accepted = []
    for i in range(k):
        g = greedy[0, i]
        accepted.append(g)
        if g != proposed[0, i]:
            # first mismatch: emit the corrected token and stop. i tokens matched.
            return torch.stack(accepted).view(1, -1), i
    # all k matched -> append one bonus token (target greedy at the last position)
    bonus = logits[:, T - 1 + k].argmax(dim=-1)[0]
    accepted.append(bonus)
    return torch.stack(accepted).view(1, -1), k


@torch.no_grad()
def speculative_generate(target: GPT, draft: GPT, idx: torch.Tensor,
                         max_new_tokens: int, k: int = 4) -> torch.Tensor:
    seq = idx
    produced = 0
    while produced < max_new_tokens:
        proposed = draft_k(draft, seq, k)
        accepted, _ = verify(target, seq, proposed)
        seq = torch.cat([seq, accepted], dim=1)
        produced += accepted.shape[1]
    return seq[:, :idx.shape[1] + max_new_tokens]
