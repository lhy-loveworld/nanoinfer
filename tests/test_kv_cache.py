"""Stage 2 — KV cache correctness.

Oracle: the Stage 1 cacheless forward. Incremental decoding through the cache
must produce *identical* logits to a full recompute over the same prefix. This
is the single most important invariant in inference engineering — if cached and
uncached disagree, generation silently corrupts.
"""
import pytest
import torch

from nanoinfer.config import GPTConfig
from nanoinfer.kv_cache import KVCache, decode_step
from nanoinfer.model import GPT


@pytest.fixture
def model_and_config():
    cfg = GPTConfig()
    model = GPT(cfg).eval()
    return model, cfg


def test_prefill_matches_full_forward(model_and_config):
    model, cfg = model_and_config
    idx = torch.randint(0, cfg.vocab_size, (2, 9))
    with torch.no_grad():
        full = model(idx)
    cache = KVCache(cfg, batch_size=2, max_seq_len=cfg.block_size)
    cached = decode_step(model, idx, cache, start_pos=0)
    torch.testing.assert_close(cached, full, rtol=1e-4, atol=1e-5)


def test_token_by_token_matches_full_forward(model_and_config):
    model, cfg = model_and_config
    B, T = 2, 10
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        full = model(idx)  # (B, T, vocab)

    cache = KVCache(cfg, batch_size=B, max_seq_len=cfg.block_size)
    collected = []
    for t in range(T):
        step = decode_step(model, idx[:, t:t + 1], cache, start_pos=t)
        assert step.shape[1] == 1
        collected.append(step[:, -1])  # (B, vocab)
    cached = torch.stack(collected, dim=1)  # (B, T, vocab)
    torch.testing.assert_close(cached, full, rtol=1e-4, atol=1e-5)


def test_prefill_then_decode(model_and_config):
    """Realistic path: prefill a prompt in one shot, then decode one at a time."""
    model, cfg = model_and_config
    B, P, D = 1, 5, 4  # prompt len 5, then 4 decode steps
    idx = torch.randint(0, cfg.vocab_size, (B, P + D))
    with torch.no_grad():
        full = model(idx)

    cache = KVCache(cfg, batch_size=B, max_seq_len=cfg.block_size)
    out_prefill = decode_step(model, idx[:, :P], cache, start_pos=0)
    torch.testing.assert_close(out_prefill, full[:, :P], rtol=1e-4, atol=1e-5)
    for t in range(P, P + D):
        step = decode_step(model, idx[:, t:t + 1], cache, start_pos=t)
        torch.testing.assert_close(step[:, -1], full[:, t], rtol=1e-4, atol=1e-5)


def test_cache_length_tracks(model_and_config):
    model, cfg = model_and_config
    cache = KVCache(cfg, batch_size=1, max_seq_len=cfg.block_size)
    decode_step(model, torch.randint(0, cfg.vocab_size, (1, 3)), cache, start_pos=0)
    assert cache.length == 3
    decode_step(model, torch.randint(0, cfg.vocab_size, (1, 1)), cache, start_pos=3)
    assert cache.length == 4
