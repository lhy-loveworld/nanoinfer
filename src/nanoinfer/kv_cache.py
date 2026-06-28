"""Stage 2 — KV cache and incremental decoding.

The whole point of a KV cache: during autoregressive generation, the keys and
values for past tokens never change, so recomputing them every step is pure
waste. Cache them once; each new token does attention with Tq=1 against the
full cached K/V (Tk = cache length). That's an O(T^2) -> O(T) win per step.

You'll reuse `attention()` and `repeat_kv()` from Stage 1 unchanged — they
already support Tq != Tk, which is exactly the prefill (Tq=T) vs decode (Tq=1)
distinction.

Run the tests with:
    pytest tests/test_kv_cache.py
"""
from __future__ import annotations

import torch

from .config import GPTConfig
from .model import GPT, attention, repeat_kv


class KVCache:
    """Pre-allocated key/value cache for every layer of a model.

    Real serving stacks pre-allocate to a max length (and later page it) rather
    than growing tensors per step — so we pre-allocate here too. You fill it in
    place and track how many positions are currently valid.
    """

    def __init__(self, config: GPTConfig, batch_size: int, max_seq_len: int,
                 device=None, dtype=torch.float32):
        self.config = config
        self.max_seq_len = max_seq_len
        # (n_layer, B, nkv, max_seq_len, hd) for keys and for values.
        shape = (config.n_layer, batch_size, config.n_kv_head, max_seq_len, config.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.length = 0  # number of valid positions currently cached

    def reset(self) -> None:
        self.length = 0

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Write new keys/values for one layer and return the full cached K/V.

        Args:
            layer_idx: which layer's cache to update
            k: (B, nkv, Tnew, hd) new keys for this step/chunk
            v: (B, nkv, Tnew, hd) new values

        Returns:
            (k_full, v_full) each of shape (B, nkv, length + Tnew, hd): the keys
            and values for *all* positions seen so far (including the new ones).

        The one subtlety: advance self.length exactly once per decode step, not
        once per layer, or repeated per-layer appends will over-count.

        Stuck? See HINTS.md (kv_cache.KVCache.append).
        """
        raise NotImplementedError


@torch.no_grad()
def decode_step(model: GPT, idx: torch.Tensor, cache: KVCache, start_pos: int) -> torch.Tensor:
    """Run a chunk of tokens through the model using the KV cache.

    Works for both prefill (idx is the whole prompt, Tq = prompt length) and
    decode (idx is a single new token, Tq = 1). The only difference is the size
    of the query axis; the cache handles the rest.

    Args:
        model: a GPT whose Stage 1 forward already works
        idx: (B, Tq) new token ids to process
        cache: the KVCache to read from and write into
        start_pos: position index of the FIRST token in idx (i.e. how many
            tokens precede it). Equals cache.length before this call. Used for
            position embeddings — the new tokens are at positions
            [start_pos, start_pos + Tq).

    Returns:
        logits: (B, Tq, vocab_size). For pure decode you typically only use the
        last position, but returning all of them keeps prefill symmetric.

    You implement this by walking the model's submodules manually (you can't call
    model.forward — it has no cache). The causal mask in `attention` with Tq=1,
    Tk=cache_len lets the new token attend to everything cached — verify you
    understand why before moving on.

    Stuck? See HINTS.md (kv_cache.decode_step).
    """
    raise NotImplementedError
