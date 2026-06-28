"""Stage 4 — continuous (iteration-level) batching.

Static batching wastes the GPU: the whole batch waits for its slowest sequence
to finish before anything new starts. Continuous batching (Orca / vLLM style)
schedules at the granularity of a *single decode step* — as soon as one sequence
finishes, its slot is freed and a waiting request is admitted. The GPU stays
full; short requests don't get stuck behind long ones (no head-of-line blocking).

This stage has two implementable pieces:
    1. `batched_decode_step` — one fused decode step over a batch whose rows are
       at *different* cache lengths (ragged), masking each row to its own valid
       prefix. This is the compute primitive.
    2. `ContinuousBatchingScheduler.step` / `.run` — the scheduling policy that
       admits, decodes, and retires requests slot-by-slot.

Sampling stays per-request (each Request carries its own generator) so a request
produces the *same* tokens whether it ran alone or inside a batch — that's the
correctness contract the tests enforce.

Run the tests with:
    pytest tests/test_batching.py
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .config import GPTConfig
from .generate import sample_next
from .model import GPT, attention, repeat_kv


@dataclass
class Request:
    """One generation request. Complete — nothing to implement here."""
    req_id: int
    prompt: torch.Tensor          # (T0,) int64 token ids
    max_new_tokens: int
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    seed: int = 0
    # filled in as it runs:
    generated: list[int] = field(default_factory=list)
    slot: int | None = None       # which batch row it occupies while running
    done: bool = False

    def output(self) -> torch.Tensor:
        """Full sequence: prompt followed by generated tokens."""
        gen = torch.tensor(self.generated, dtype=torch.long)
        return torch.cat([self.prompt, gen])


class BatchedKVCache:
    """KV cache with a fixed number of slots (rows), each tracking its own
    length. Complete — use it from your scheduler / decode step.

    A slot is a row of the batch. `lengths[s]` is how many valid positions slot
    s currently holds. Free slots are tracked in `free_slots`.
    """

    def __init__(self, config: GPTConfig, n_slots: int, max_seq_len: int,
                 device=None, dtype=torch.float32):
        self.config = config
        self.n_slots = n_slots
        self.max_seq_len = max_seq_len
        shape = (config.n_layer, n_slots, config.n_kv_head, max_seq_len, config.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.lengths = torch.zeros(n_slots, dtype=torch.long, device=device)
        self.free_slots = list(range(n_slots))

    def allocate(self) -> int:
        """Reserve a free slot, returning its index. Raises if none free."""
        if not self.free_slots:
            raise RuntimeError("no free slots")
        s = self.free_slots.pop(0)
        self.lengths[s] = 0
        return s

    def release(self, slot: int) -> None:
        self.lengths[slot] = 0
        if slot not in self.free_slots:
            self.free_slots.append(slot)
            self.free_slots.sort()

    def write(self, layer_idx: int, rows: list[int], k: torch.Tensor, v: torch.Tensor) -> None:
        """Append one new K/V vector for the given rows at each row's length.

        Args:
            rows: the active slot indices, in the same order as the batch dim of k/v
            k, v: (len(rows), nkv, 1, hd) new keys/values (single decode step)
        Does NOT advance lengths — the decode step advances them after all layers.
        """
        for i, s in enumerate(rows):
            pos = int(self.lengths[s])
            self.k[layer_idx, s, :, pos:pos + 1] = k[i]
            self.v[layer_idx, s, :, pos:pos + 1] = v[i]


@torch.no_grad()
def batched_decode_step(model: GPT, idx: torch.Tensor, cache: BatchedKVCache,
                        rows: list[int]) -> torch.Tensor:
    """One fused decode step for a ragged batch of single new tokens.

    Args:
        model: GPT with working Stage 1 weights/forward.
        idx: (n_active, 1) the new token id for each active row (order matches `rows`).
        cache: the BatchedKVCache.
        rows: active slot indices, parallel to idx's batch dimension.

    Returns:
        logits: (n_active, vocab_size) for the new position of each active row.

    Why this is the interesting part: each row attends over a *different* number
    of cached positions (cache.lengths[row]). You run one batched matmul over the
    padded max length, then mask each row down to its own valid prefix so padding
    positions contribute zero attention weight. (Do NOT use causal masking here:
    with Tq=1 the only constraint is the padding mask; the new token may see all
    valid cached positions.)

    Stuck? See HINTS.md (batching.batched_decode_step).
    """
    raise NotImplementedError


class ContinuousBatchingScheduler:
    """Iteration-level scheduler. You implement `step` and `run`."""

    def __init__(self, model: GPT, config: GPTConfig, max_batch_size: int,
                 max_seq_len: int):
        self.model = model
        self.config = config
        self.max_batch_size = max_batch_size
        self.cache = BatchedKVCache(config, max_batch_size, max_seq_len)
        self.waiting: list[Request] = []     # not yet started (FIFO)
        self.running: list[Request] = []      # currently decoding
        self.finished: list[Request] = []

    def add_request(self, req: Request) -> None:
        self.waiting.append(req)

    def _admit(self) -> None:
        """Move waiting requests into free slots and prefill them.

        This is part of the exercise — it's where prefill lives.

        Stuck? See HINTS.md (batching.ContinuousBatchingScheduler._admit).
        """
        raise NotImplementedError

    def step(self) -> None:
        """Advance every running request by exactly one decoded token, retire
        any that hit max_new_tokens, and backfill freed slots via self._admit().

        The invariant the tests check: len(running) is never > max_batch_size,
        and a finished request's tokens are independent of what else was batched
        alongside it.

        Stuck? See HINTS.md (batching.ContinuousBatchingScheduler.step).
        """
        raise NotImplementedError

    def run(self) -> dict[int, torch.Tensor]:
        """Drive the scheduler until all requests finish.

        Returns: {req_id: full output sequence tensor}.

        Stuck? See HINTS.md (batching.ContinuousBatchingScheduler.run).
        """
        raise NotImplementedError
