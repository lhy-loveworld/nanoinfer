"""Stage 2 — KV cache and incremental decoding. [REFERENCE SOLUTION]"""
from __future__ import annotations

import torch

from .config import GPTConfig
from .model import GPT, attention, repeat_kv


class KVCache:
    def __init__(self, config: GPTConfig, batch_size: int, max_seq_len: int,
                 device=None, dtype=torch.float32):
        self.config = config
        self.max_seq_len = max_seq_len
        shape = (config.n_layer, batch_size, config.n_kv_head, max_seq_len, config.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.length = 0

    def reset(self) -> None:
        self.length = 0

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Write new K/V at the current length offset; return the full K/V so far.

        Uses self.length as the write offset (it is the same for every layer in a
        single decode step). `decode_step` advances self.length once, after all
        layers have appended.
        """
        Tnew = k.size(2)
        off = self.length
        self.k[layer_idx, :, :, off:off + Tnew] = k
        self.v[layer_idx, :, :, off:off + Tnew] = v
        k_full = self.k[layer_idx, :, :, :off + Tnew]
        v_full = self.v[layer_idx, :, :, :off + Tnew]
        return k_full, v_full


@torch.no_grad()
def decode_step(model: GPT, idx: torch.Tensor, cache: KVCache, start_pos: int) -> torch.Tensor:
    B, Tq = idx.shape
    cfg = model.config
    pos = torch.arange(start_pos, start_pos + Tq, device=idx.device)
    x = model.transformer.wte(idx) + model.transformer.wpe(pos)

    for i, block in enumerate(model.transformer.h):
        attn = block.attn
        h = block.ln_1(x)
        q_dim = attn.nh * attn.hd
        kv_dim = attn.nkv * attn.hd
        q, k, v = attn.c_attn(h).split([q_dim, kv_dim, kv_dim], dim=-1)
        q = q.view(B, Tq, attn.nh, attn.hd).transpose(1, 2)    # (B, nh, Tq, hd)
        k = k.view(B, Tq, attn.nkv, attn.hd).transpose(1, 2)   # (B, nkv, Tq, hd)
        v = v.view(B, Tq, attn.nkv, attn.hd).transpose(1, 2)
        k_full, v_full = cache.append(i, k, v)                 # (B, nkv, start_pos+Tq, hd)
        k_full = repeat_kv(k_full, cfg.n_rep)
        v_full = repeat_kv(v_full, cfg.n_rep)
        # Tk = start_pos+Tq, Tq queries -> j0 = start_pos, so query i (abs pos
        # start_pos+i) attends keys 0..start_pos+i. Exactly what we want.
        y = attention(q, k_full, v_full, causal=True)
        y = y.transpose(1, 2).contiguous().view(B, Tq, cfg.n_embd)
        x = x + attn.c_proj(y)
        x = x + block.mlp(block.ln_2(x))

    cache.length += Tq
    x = model.transformer.ln_f(x)
    return model.lm_head(x)
