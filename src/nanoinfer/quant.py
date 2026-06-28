"""Stage 5b — weight quantization.

Inference is usually memory-bandwidth bound: smaller weights = faster decode and
more model per GPU. INT8 (and FP8 on Blackwell) are the workhorses. You'll
implement symmetric quantization with per-channel scales — the scheme that keeps
accuracy because each output channel gets its own dynamic range.

Run the tests with:
    pytest tests/test_quant.py
    pytest tests/test_quant.py --device cuda    # exercises the FP8 path on the 5080
"""
from __future__ import annotations

import torch
import torch.nn as nn


def quantize_per_channel_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-(output-)channel INT8 quantization of a weight matrix.

    Args:
        w: (out_features, in_features) float weights.
    Returns:
        (q, scale) where
            q: (out_features, in_features) int8 in [-127, 127]
            scale: (out_features,) float, scale[i] = max(|w[i]|) / 127
        such that w ≈ q.float() * scale[:, None].

    Use 127 (not 128) so the range is symmetric. Guard against a zero row
    (all-zero channel) producing a divide-by-zero scale.
    """
    raise NotImplementedError


def dequantize_per_channel_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of quantize_per_channel_int8: returns q.float() * scale[:, None]."""
    raise NotImplementedError


class QuantizedLinear(nn.Module):
    """A drop-in replacement for nn.Linear that stores INT8 weights + per-channel
    scales and dequantizes on the fly.

    In a production kernel you'd matmul in INT8 and accumulate in INT32; here we
    dequantize then matmul (simpler, and enough to study the accuracy impact).
    """

    def __init__(self, q: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.register_buffer("q", q)
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias if bias is not None else None)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "QuantizedLinear":
        """Quantize an existing nn.Linear's weight into a QuantizedLinear.

        Stuck? See HINTS.md (quant.QuantizedLinear.from_linear).
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """y = x @ W_dequant^T + bias, where W_dequant = dequantize(q, scale).

        Args:
            x: (..., in_features)
        Returns:
            (..., out_features)
        """
        raise NotImplementedError


def fp8_roundtrip(w: torch.Tensor) -> torch.Tensor:
    """Cast w to FP8 (e4m3) and back to float — to measure FP8 representation error.

    Args:
        w: float tensor.
    Returns:
        w cast to torch.float8_e4m3fn then back to w.dtype.

    Note: FP8 dtypes exist on CPU for storage/casting in recent PyTorch, but the
    test that *uses* this is GPU-gated since FP8 compute targets Blackwell. Keep
    the implementation a straight `.to(torch.float8_e4m3fn).to(w.dtype)`.
    """
    raise NotImplementedError
