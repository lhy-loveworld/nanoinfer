"""Stage 5a — speculative decoding correctness.

The invariant that makes greedy speculation testable: regardless of the draft
model's quality (even a random draft), the *output* equals plain greedy target
decoding. A good draft only changes speed (acceptance rate), never the result.
"""
import pytest
import torch

from nanoinfer.config import GPTConfig
from nanoinfer.model import GPT
from nanoinfer.speculative import draft_k, speculative_generate, verify


def _greedy_reference(model, idx, n):
    seq = idx.clone()
    with torch.no_grad():
        for _ in range(n):
            nxt = model(seq)[:, -1].argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], dim=1)
    return seq


@pytest.fixture
def models():
    cfg = GPTConfig()
    target = GPT(cfg).eval()
    draft = GPT(cfg).eval()  # different init -> imperfect draft, still must be exact
    return target, draft, cfg


def test_draft_k_shape(models):
    _, draft, cfg = models
    idx = torch.randint(0, cfg.vocab_size, (1, 5))
    out = draft_k(draft, idx, k=4)
    assert out.shape == (1, 4)


def test_verify_accepts_self_proposals(models):
    """If the proposal already equals the target's own greedy continuation,
    verify must accept all k and add a bonus -> k+1 tokens."""
    target, _, cfg = models
    idx = torch.randint(0, cfg.vocab_size, (1, 5))
    target_cont = _greedy_reference(target, idx, 3)[:, idx.shape[1]:]  # target's own next 3
    accepted, n = verify(target, idx, target_cont)
    assert n == 3
    assert accepted.shape == (1, 4)
    torch.testing.assert_close(accepted[:, :3], target_cont)


def test_speculative_equals_greedy(models):
    target, draft, cfg = models
    idx = torch.randint(0, cfg.vocab_size, (1, 6))
    spec = speculative_generate(target, draft, idx, max_new_tokens=12, k=4)
    ref = _greedy_reference(target, idx, 12)
    assert spec.shape == ref.shape
    torch.testing.assert_close(spec, ref)


def test_speculative_exact_even_with_identical_models(models):
    """Draft == target -> every proposal accepted, but output is still the plain
    greedy sequence (fast path stress test)."""
    target, _, cfg = models
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    spec = speculative_generate(target, target, idx, max_new_tokens=10, k=5)
    ref = _greedy_reference(target, idx, 10)
    torch.testing.assert_close(spec, ref)
