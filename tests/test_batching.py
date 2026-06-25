"""Stage 4 — continuous batching.

Core contract: a request's output must be identical whether it runs alone or
batched alongside others of different lengths. We get the per-request reference
from Stage 3's `generate` (same prompt, same seed, same sampling params).
"""
import pytest
import torch

from nanoinfer.batching import ContinuousBatchingScheduler, Request
from nanoinfer.config import GPTConfig
from nanoinfer.generate import generate
from nanoinfer.model import GPT


def _reference(model, req: Request) -> torch.Tensor:
    g = torch.Generator().manual_seed(req.seed)
    out = generate(
        model, req.prompt[None, :], req.max_new_tokens,
        temperature=req.temperature, top_k=req.top_k, top_p=req.top_p, generator=g,
    )
    return out[0]


@pytest.fixture
def model_cfg():
    cfg = GPTConfig()
    return GPT(cfg).eval(), cfg


def test_single_request_matches_generate(model_cfg):
    model, cfg = model_cfg
    req = Request(req_id=0, prompt=torch.randint(0, cfg.vocab_size, (4,)),
                  max_new_tokens=6, temperature=0.0, seed=1)
    sched = ContinuousBatchingScheduler(model, cfg, max_batch_size=2, max_seq_len=cfg.block_size)
    sched.add_request(req)
    out = sched.run()
    torch.testing.assert_close(out[0], _reference(model, req))


def test_batched_varied_lengths_match_reference(model_cfg):
    """Many requests, varied prompt lengths and max_new_tokens, batch smaller
    than the number of requests so slots churn (admission + retirement)."""
    model, cfg = model_cfg
    reqs = [
        Request(0, torch.randint(0, cfg.vocab_size, (3,)), max_new_tokens=8, temperature=0.0, seed=10),
        Request(1, torch.randint(0, cfg.vocab_size, (6,)), max_new_tokens=2, temperature=0.0, seed=11),
        Request(2, torch.randint(0, cfg.vocab_size, (2,)), max_new_tokens=10, temperature=0.0, seed=12),
        Request(3, torch.randint(0, cfg.vocab_size, (5,)), max_new_tokens=4, temperature=0.0, seed=13),
        Request(4, torch.randint(0, cfg.vocab_size, (4,)), max_new_tokens=7, temperature=0.0, seed=14),
    ]
    refs = {r.req_id: _reference(model, r) for r in reqs}

    sched = ContinuousBatchingScheduler(model, cfg, max_batch_size=2, max_seq_len=cfg.block_size)
    for r in reqs:
        sched.add_request(r)
    out = sched.run()

    assert set(out.keys()) == set(refs.keys())
    for rid, ref in refs.items():
        torch.testing.assert_close(out[rid], ref, msg=f"request {rid} diverged from solo generate")


def test_sampling_requests_are_reproducible(model_cfg):
    """With temperature > 0, per-request generators must make batched output
    match solo output exactly (sampling independence across the batch)."""
    model, cfg = model_cfg
    reqs = [
        Request(0, torch.randint(0, cfg.vocab_size, (3,)), max_new_tokens=6, temperature=0.8, top_k=10, seed=1),
        Request(1, torch.randint(0, cfg.vocab_size, (5,)), max_new_tokens=6, temperature=0.8, top_k=10, seed=2),
        Request(2, torch.randint(0, cfg.vocab_size, (4,)), max_new_tokens=6, temperature=0.8, top_k=10, seed=3),
    ]
    refs = {r.req_id: _reference(model, r) for r in reqs}
    sched = ContinuousBatchingScheduler(model, cfg, max_batch_size=2, max_seq_len=cfg.block_size)
    for r in reqs:
        sched.add_request(r)
    out = sched.run()
    for rid, ref in refs.items():
        torch.testing.assert_close(out[rid], ref)


def test_never_exceeds_max_batch(model_cfg):
    """Instrument step() via the running list: the scheduler must never hold
    more concurrent sequences than max_batch_size."""
    model, cfg = model_cfg
    reqs = [Request(i, torch.randint(0, cfg.vocab_size, (3,)), max_new_tokens=5, temperature=0.0, seed=i)
            for i in range(6)]
    sched = ContinuousBatchingScheduler(model, cfg, max_batch_size=2, max_seq_len=cfg.block_size)
    for r in reqs:
        sched.add_request(r)

    max_seen = 0
    orig_step = sched.step

    def wrapped():
        nonlocal max_seen
        max_seen = max(max_seen, len(sched.running))
        return orig_step()

    sched.step = wrapped
    sched.run()
    assert max_seen <= 2
