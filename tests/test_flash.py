"""Stage 5c — tiled attention.

Correctness oracle: F.scaled_dot_product_attention. The perf test is GPU-only and
marked `perf` so it's opt-in (pytest -m perf --device cuda).
"""
import time

import pytest
import torch
import torch.nn.functional as F

from nanoinfer.flash import flash_attention


@pytest.mark.parametrize("T,block", [(64, 16), (128, 32), (100, 64)])
def test_flash_matches_sdpa_causal(T, block):
    B, nh, hd = 2, 4, 32
    q, k, v = (torch.randn(B, nh, T, hd) for _ in range(3))
    got = flash_attention(q, k, v, causal=True, block_size=block)
    want = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_flash_matches_sdpa_noncausal():
    B, nh, T, hd = 2, 4, 96, 32
    q, k, v = (torch.randn(B, nh, T, hd) for _ in range(3))
    got = flash_attention(q, k, v, causal=False, block_size=32)
    want = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_flash_handles_ragged_block_size():
    # T not divisible by block_size must still be correct (tail block).
    B, nh, T, hd = 1, 2, 70, 16
    q, k, v = (torch.randn(B, nh, T, hd) for _ in range(3))
    got = flash_attention(q, k, v, causal=True, block_size=32)
    want = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.perf
def test_flash_correct_on_gpu_long_seq(device):
    if device.type != "cuda":
        pytest.skip("pass --device cuda")
    B, nh, T, hd = 4, 8, 2048, 64
    q, k, v = (torch.randn(B, nh, T, hd, device=device) for _ in range(3))
    got = flash_attention(q, k, v, causal=True, block_size=128)
    want = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(got, want, rtol=2e-3, atol=2e-3)
    # informational timing (not asserted — your pure-PyTorch loop won't beat the
    # fused kernel; the point is to feel the memory behavior):
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(5):
        flash_attention(q, k, v, causal=True, block_size=128)
    torch.cuda.synchronize()
    print(f"\nflash_attention: {(time.time()-t0)/5*1e3:.2f} ms/iter on {torch.cuda.get_device_name(0)}")
