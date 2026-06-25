"""Stage 3 — sampling + decode loop.

These tests pin down the *semantics* of each filter (not numerics against a
black box), because that's where sampling bugs hide.
"""
import pytest
import torch
import torch.nn.functional as F

from nanoinfer.config import GPTConfig
from nanoinfer.generate import (
    apply_temperature,
    generate,
    sample_next,
    top_k_filter,
    top_p_filter,
)
from nanoinfer.model import GPT


def test_apply_temperature():
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    torch.testing.assert_close(apply_temperature(logits, 2.0), logits / 2.0)


def test_top_k_keeps_exactly_k():
    logits = torch.tensor([[5.0, 1.0, 4.0, 2.0, 3.0]])
    out = top_k_filter(logits, k=2)
    finite = torch.isfinite(out)
    assert finite.sum().item() == 2
    # the kept entries must be the two largest (values 5.0 and 4.0 -> idx 0, 2)
    assert finite[0, 0] and finite[0, 2]
    assert out[0, 1] == float("-inf")


def test_top_k_larger_than_vocab_is_noop():
    logits = torch.randn(1, 4)
    torch.testing.assert_close(top_k_filter(logits, k=10), logits)


def test_top_p_basic_nucleus():
    # probs after softmax: pick logits so the top-1 mass is ~0.7
    logits = torch.log(torch.tensor([[0.7, 0.2, 0.07, 0.03]]))
    out = top_p_filter(logits, p=0.8)
    finite = torch.isfinite(out)
    # cumulative: 0.7 (<0.8) -> add next -> 0.9 (>=0.8). So keep 2 tokens.
    assert finite.sum().item() == 2
    assert finite[0, 0] and finite[0, 1]


def test_top_p_always_keeps_one():
    # top token alone exceeds p; must still keep exactly it (never zero tokens)
    logits = torch.log(torch.tensor([[0.95, 0.03, 0.02]]))
    out = top_p_filter(logits, p=0.5)
    assert torch.isfinite(out).sum().item() == 1
    assert torch.isfinite(out[0, 0])


def test_sample_next_greedy_is_argmax():
    logits = torch.randn(4, 50)
    got = sample_next(logits, temperature=0.0)
    torch.testing.assert_close(got, logits.argmax(dim=-1))


def test_sample_next_respects_top_k_support():
    # with top_k=1 the only legal sample is the argmax, regardless of seed
    logits = torch.randn(8, 30)
    g = torch.Generator().manual_seed(123)
    got = sample_next(logits, temperature=1.0, top_k=1, generator=g)
    torch.testing.assert_close(got, logits.argmax(dim=-1))


def test_sample_next_is_reproducible():
    logits = torch.randn(4, 100)
    a = sample_next(logits, temperature=1.0, generator=torch.Generator().manual_seed(7))
    b = sample_next(logits, temperature=1.0, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(a, b)


def test_generate_shape_and_vocab():
    cfg = GPTConfig()
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (3, 4))
    out = generate(model, idx, max_new_tokens=6, temperature=1.0,
                   generator=torch.Generator().manual_seed(0))
    assert out.shape == (3, 4 + 6)
    assert (out >= 0).all() and (out < cfg.vocab_size).all()
    # the prompt prefix must be preserved
    torch.testing.assert_close(out[:, :4], idx)


def test_generate_greedy_matches_manual_loop():
    """Greedy generate must equal a hand-rolled argmax loop using the plain
    (cacheless) forward — ties the whole stack together."""
    cfg = GPTConfig()
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 5))
    out = generate(model, idx, max_new_tokens=5, temperature=0.0)

    seq = idx.clone()
    with torch.no_grad():
        for _ in range(5):
            logits = model(seq)
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, nxt], dim=1)
    torch.testing.assert_close(out, seq)
