"""Stage 4 — continuous (iteration-level) batching. [REFERENCE SOLUTION]"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .config import GPTConfig
from .generate import sample_next
from .model import GPT, repeat_kv


@dataclass
class Request:
    req_id: int
    prompt: torch.Tensor
    max_new_tokens: int
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    seed: int = 0
    generated: list[int] = field(default_factory=list)
    slot: int | None = None
    done: bool = False
    _gen: torch.Generator | None = field(default=None, repr=False)

    def output(self) -> torch.Tensor:
        gen = torch.tensor(self.generated, dtype=torch.long, device=self.prompt.device)
        return torch.cat([self.prompt, gen])


class BatchedKVCache:
    def __init__(self, config: GPTConfig, n_slots: int, max_seq_len: int,
                 device=None, dtype=torch.float32):
        self.config = config
        self.n_slots = n_slots
        self.max_seq_len = max_seq_len
        shape = (config.n_layer, n_slots, config.n_kv_head, max_seq_len, config.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.lengths = torch.zeros(n_slots, dtype=torch.long, device=device)
        self.free_slots = list(range(n_slots))

    def allocate(self) -> int:
        if not self.free_slots:
            raise RuntimeError("no free slots")
        s = self.free_slots.pop(0)
        self.lengths[s] = 0
        return s

    def release(self, slot: int) -> None:
        self.lengths[slot] = 0
        if slot not in self.free_slots:
            self.free_slots.append(slot)
            self.free_slots.sort()

    def write(self, layer_idx: int, rows: list[int], k: torch.Tensor, v: torch.Tensor) -> None:
        for i, s in enumerate(rows):
            pos = int(self.lengths[s])
            self.k[layer_idx, s, :, pos:pos + 1] = k[i]
            self.v[layer_idx, s, :, pos:pos + 1] = v[i]


@torch.no_grad()
def batched_decode_step(model: GPT, idx: torch.Tensor, cache: BatchedKVCache,
                        rows: list[int]) -> torch.Tensor:
    n = len(rows)
    cfg = model.config
    device = idx.device
    lengths = cache.lengths[rows]                       # (n,) length BEFORE this token
    Lmax = int(lengths.max().item()) + 1                # +1 for the new token

    pos_emb = model.transformer.wpe(lengths).unsqueeze(1)   # new token sits at `length`
    x = model.transformer.wte(idx) + pos_emb            # (n, 1, C)

    # key-padding mask: row r may attend key positions 0..lengths[r] (inclusive)
    key_idx = torch.arange(Lmax, device=device)[None, :]    # (1, Lmax)
    valid = key_idx <= lengths[:, None]                     # (n, Lmax) bool
    attn_mask = valid[:, None, None, :]                     # (n, 1, 1, Lmax)

    for i, block in enumerate(model.transformer.h):
        attn = block.attn
        h = block.ln_1(x)
        q_dim = attn.nh * attn.hd
        kv_dim = attn.nkv * attn.hd
        q, k, v = attn.c_attn(h).split([q_dim, kv_dim, kv_dim], dim=-1)
        q = q.view(n, 1, attn.nh, attn.hd).transpose(1, 2)   # (n, nh, 1, hd)
        k = k.view(n, 1, attn.nkv, attn.hd).transpose(1, 2)  # (n, nkv, 1, hd)
        v = v.view(n, 1, attn.nkv, attn.hd).transpose(1, 2)
        cache.write(i, rows, k, v)
        k_full = cache.k[i, rows][:, :, :Lmax]               # (n, nkv, Lmax, hd)
        v_full = cache.v[i, rows][:, :, :Lmax]
        k_full = repeat_kv(k_full, cfg.n_rep)
        v_full = repeat_kv(v_full, cfg.n_rep)
        att = (q @ k_full.transpose(-2, -1)) / math.sqrt(attn.hd)  # (n, nh, 1, Lmax)
        att = att.masked_fill(~attn_mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v_full).transpose(1, 2).contiguous().view(n, 1, cfg.n_embd)
        x = x + attn.c_proj(y)
        x = x + block.mlp(block.ln_2(x))

    cache.lengths[rows] += 1
    x = model.transformer.ln_f(x)
    return model.lm_head(x)[:, -1]                           # (n, vocab)


class ContinuousBatchingScheduler:
    def __init__(self, model: GPT, config: GPTConfig, max_batch_size: int,
                 max_seq_len: int):
        self.model = model
        self.config = config
        self.max_batch_size = max_batch_size
        self.device = next(model.parameters()).device
        self.cache = BatchedKVCache(config, max_batch_size, max_seq_len, device=self.device,
                                    dtype=next(model.parameters()).dtype)
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.finished: list[Request] = []

    def add_request(self, req: Request) -> None:
        self.waiting.append(req)

    def _sample(self, req: Request, logits_row: torch.Tensor) -> None:
        nxt = sample_next(
            logits_row, temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
            generator=req._gen,
        )
        req.generated.append(int(nxt))

    def _retire(self, req: Request) -> None:
        req.done = True
        self.cache.release(req.slot)
        if req in self.running:
            self.running.remove(req)
        self.finished.append(req)

    def _admit(self) -> None:
        while self.cache.free_slots and self.waiting:
            req = self.waiting.pop(0)
            slot = self.cache.allocate()
            req.slot = slot
            req._gen = torch.Generator(device=self.device).manual_seed(req.seed)
            prompt = req.prompt.to(self.device)
            # prefill one token at a time (reuses the decode primitive)
            last = None
            for t in range(prompt.shape[0]):
                last = batched_decode_step(self.model, prompt[t].view(1, 1), self.cache, [slot])
            self._sample(req, last)  # first generated token
            if len(req.generated) >= req.max_new_tokens:
                self._retire(req)
            else:
                self.running.append(req)

    def step(self) -> None:
        if not self.running:
            return
        rows = [r.slot for r in self.running]
        idx = torch.tensor([[r.generated[-1]] for r in self.running],
                           dtype=torch.long, device=self.device)
        logits = batched_decode_step(self.model, idx, self.cache, rows)  # (n, vocab)
        finished_now = []
        for j, r in enumerate(self.running):
            self._sample(r, logits[j:j + 1])
            if len(r.generated) >= r.max_new_tokens:
                finished_now.append(r)
        for r in finished_now:
            self._retire(r)
        self._admit()

    def run(self) -> dict[int, torch.Tensor]:
        self._admit()
        while self.running:
            self.step()
        return {r.req_id: r.output() for r in self.finished}
