"""Stage 1+ — Rotary Position Embeddings.

The defining property of RoPE is *relative-position invariance*: the attention
score between a rotated query and key depends only on their offset (m - n), not
their absolute positions. The tests pin that down, plus the structural facts
(norm preservation, position-0 identity) and an independent numeric oracle.
"""
import pytest
import torch

from nanoinfer.rope import apply_rope, build_rope_cache, rotate_half


def test_cache_shapes():
    cos, sin = build_rope_cache(16, 8)
    assert cos.shape == (16, 8)
    assert sin.shape == (16, 8)


def test_rotate_half_twice_is_negation():
    # a 90-degree rotation applied twice is a 180-degree rotation == negation
    x = torch.randn(2, 3, 8)
    torch.testing.assert_close(rotate_half(rotate_half(x)), -x)


def test_position_zero_is_identity():
    hd = 8
    cos, sin = build_rope_cache(4, hd)
    x = torch.randn(1, 2, 4, hd)
    out = apply_rope(x, cos, sin)
    # angle at position 0 is 0 -> cos=1, sin=0 -> no change at that position
    torch.testing.assert_close(out[:, :, 0], x[:, :, 0])


def test_preserves_norm():
    # rotation is orthogonal, so each per-position vector keeps its L2 norm
    hd, T = 16, 10
    cos, sin = build_rope_cache(T, hd)
    x = torch.randn(2, 4, T, hd)
    out = apply_rope(x, cos, sin)
    torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1), rtol=1e-5, atol=1e-5)


def test_relative_position_property():
    """<rope(q, m), rope(k, n)> depends only on (m - n): the score matrix formed
    by rotating the SAME q,k to every position is Toeplitz (constant diagonals)."""
    hd, T = 16, 12
    cos, sin = build_rope_cache(T, hd)
    qv, kv = torch.randn(hd), torch.randn(hd)
    Q = apply_rope(qv.expand(T, hd), cos, sin)   # row m = qv rotated to position m
    K = apply_rope(kv.expand(T, hd), cos, sin)
    S = Q @ K.t()                                # S[m, n] = <rope(qv,m), rope(kv,n)>
    # constant along diagonals -> S[m, n] == S[m+1, n+1]
    torch.testing.assert_close(S[:-1, :-1], S[1:, 1:], rtol=1e-4, atol=1e-4)


def test_matches_explicit_rotation():
    """Independent oracle: rotate each (x[i], x[i+hd/2]) pair by angle i*pos,
    recomputed from scratch (does not go through apply_rope's code path)."""
    hd, T = 8, 5
    cos, sin = build_rope_cache(T, hd)
    x = torch.randn(1, 1, T, hd)
    got = apply_rope(x, cos, sin)

    half = hd // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, hd, 2).float() / hd))  # (half,)
    angle = torch.outer(torch.arange(T).float(), inv_freq)               # (T, half)
    c, s = angle.cos(), angle.sin()
    x1, x2 = x[..., :half], x[..., half:]
    want = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_odd_head_dim_rejected():
    with pytest.raises(AssertionError):
        build_rope_cache(8, head_dim=7)


@pytest.mark.gpu
def test_rope_on_gpu(device):
    if device.type != "cuda":
        pytest.skip("pass --device cuda")
    hd, T = 16, 32
    cos, sin = build_rope_cache(T, hd, device=device)
    x = torch.randn(2, 4, T, hd, device=device)
    out = apply_rope(x, cos, sin)
    assert out.shape == x.shape and out.device.type == "cuda"
    torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1), rtol=1e-4, atol=1e-4)
