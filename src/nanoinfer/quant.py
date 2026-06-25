"""Stage 5b — weight quantization. [REFERENCE SOLUTION]"""
from __future__ import annotations

import torch
import torch.nn as nn


def quantize_per_channel_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = w.abs().amax(dim=1)                 # (out,)
    scale = amax / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)  # zero-row safe
    q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale


def dequantize_per_channel_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale[:, None]


class QuantizedLinear(nn.Module):
    def __init__(self, q: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.register_buffer("q", q)
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias if bias is not None else None)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "QuantizedLinear":
        q, scale = quantize_per_channel_int8(linear.weight.data)
        bias = linear.bias.data.clone() if linear.bias is not None else None
        return cls(q, scale, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = dequantize_per_channel_int8(self.q, self.scale)  # (out, in)
        y = x @ w.t()
        if self.bias is not None:
            y = y + self.bias
        return y


def fp8_roundtrip(w: torch.Tensor) -> torch.Tensor:
    return w.to(torch.float8_e4m3fn).to(w.dtype)
