"""Stage 5b — quantization.

Quantization is lossy, so tests assert *bounded* error rather than exactness:
round-trips stay within the quantization step, and a QuantizedLinear stays close
to its float original.
"""
import pytest
import torch
import torch.nn as nn

from nanoinfer.quant import (
    QuantizedLinear,
    dequantize_per_channel_int8,
    fp8_roundtrip,
    quantize_per_channel_int8,
)


def test_int8_roundtrip_bounded_error():
    w = torch.randn(16, 32)
    q, scale = quantize_per_channel_int8(w)
    assert q.dtype == torch.int8
    assert q.abs().max() <= 127
    assert scale.shape == (16,)
    recon = dequantize_per_channel_int8(q, scale)
    # error per element is at most half a quantization step = scale/2 (+fp slack)
    assert torch.all((w - recon).abs() <= scale[:, None] / 2 + 1e-6)


def test_int8_zero_row_is_safe():
    w = torch.randn(4, 8)
    w[2] = 0.0
    q, scale = quantize_per_channel_int8(w)
    recon = dequantize_per_channel_int8(q, scale)
    assert torch.isfinite(recon).all()
    torch.testing.assert_close(recon[2], torch.zeros(8))


def test_quantized_linear_close_to_float():
    lin = nn.Linear(64, 32)
    qlin = QuantizedLinear.from_linear(lin)
    x = torch.randn(8, 64)
    ref = lin(x)
    got = qlin(x)
    assert got.shape == ref.shape
    # per-channel int8 on well-conditioned weights: small relative error
    rel = (got - ref).norm() / ref.norm()
    assert rel < 0.02, f"relative error too high: {rel.item():.4f}"


def test_quantized_linear_preserves_bias():
    lin = nn.Linear(16, 16, bias=True)
    qlin = QuantizedLinear.from_linear(lin)
    torch.testing.assert_close(qlin.bias, lin.bias.data)


@pytest.mark.gpu
def test_fp8_roundtrip_error_small(device):
    if device.type != "cuda":
        pytest.skip("pass --device cuda to exercise FP8 on the GPU")
    w = torch.randn(128, 128, device=device)
    recon = fp8_roundtrip(w)
    rel = (w - recon).abs().norm() / w.norm()
    # e4m3 has ~3 mantissa bits -> coarse but bounded
    assert rel < 0.1, f"fp8 round-trip error unexpectedly high: {rel.item():.4f}"
