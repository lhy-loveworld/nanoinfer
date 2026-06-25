"""Stage 1a — attention math.

Oracle: torch.nn.functional.scaled_dot_product_attention. If your hand-written
`attention` matches it (and repeat_kv expands heads correctly), the math is right.
"""
import pytest
import torch
import torch.nn.functional as F

from nanoinfer.config import GPTConfig
from nanoinfer.model import attention, repeat_kv


def test_repeat_kv_shapes_and_values():
    B, nkv, T, hd = 2, 2, 5, 8
    n_rep = 3
    x = torch.randn(B, nkv, T, hd)
    out = repeat_kv(x, n_rep)
    assert out.shape == (B, nkv * n_rep, T, hd)
    # head group g must be repeated contiguously: out heads [0,1,2] == x head 0
    for g in range(nkv):
        for r in range(n_rep):
            torch.testing.assert_close(out[:, g * n_rep + r], x[:, g])


def test_repeat_kv_noop_when_one():
    x = torch.randn(2, 4, 3, 8)
    torch.testing.assert_close(repeat_kv(x, 1), x)


@pytest.mark.parametrize("Tq,Tk", [(6, 6), (1, 6), (3, 10)])
def test_attention_matches_sdpa_causal(Tq, Tk):
    B, nh, hd = 2, 4, 16
    q = torch.randn(B, nh, Tq, hd)
    k = torch.randn(B, nh, Tk, hd)
    v = torch.randn(B, nh, Tk, hd)
    got = attention(q, k, v, causal=True)
    # Bottom-right aligned causal mask (j0 = Tk - Tq): the last query sees the
    # last key, so a single decode query (Tq=1) attends the whole cache. Note
    # F.sdpa's is_causal=True uses *top-left* tril and is only equivalent when
    # Tq == Tk — so we build the mask explicitly to pin the decode convention.
    q_idx = torch.arange(Tq).view(Tq, 1)
    k_idx = torch.arange(Tk).view(1, Tk)
    mask = k_idx <= (Tk - Tq) + q_idx  # True = attend
    want = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert got.shape == (B, nh, Tq, hd)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_attention_non_causal_matches_sdpa():
    B, nh, T, hd = 2, 4, 7, 16
    q, k, v = (torch.randn(B, nh, T, hd) for _ in range(3))
    got = attention(q, k, v, causal=False)
    want = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_attention_is_causal_in_practice():
    # Changing a *future* key/value must not change an earlier query's output.
    B, nh, T, hd = 1, 2, 6, 8
    q, k, v = (torch.randn(B, nh, T, hd) for _ in range(3))
    out1 = attention(q, k, v, causal=True)
    v2 = v.clone()
    v2[:, :, -1] += 100.0  # perturb the last (future-most) value
    out2 = attention(q, k, v2, causal=True)
    # all query positions except the last must be unchanged
    torch.testing.assert_close(out1[:, :, :-1], out2[:, :, :-1])
    assert not torch.allclose(out1[:, :, -1], out2[:, :, -1])
