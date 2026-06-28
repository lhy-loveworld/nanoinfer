# Hints

Stuck on a function? The skeletons in `src/nanoinfer/` deliberately keep only the
*spec* — what each function must do, its shapes, and the behavioral contract the
tests enforce. This file holds the **implementation sketches**: the step-by-step
recipes and algorithm pseudocode that would otherwise give the answer away.

Try to derive the approach yourself first (that's the point of the exercise). If
you get stuck, find the function below. Each docstring points here with a
`See HINTS.md (<module>.<name>)` line.

---

## Stage 1 — `model.py`

### model.repeat_kv

Implement without allocating more memory than necessary where you can:
`torch.expand` + `reshape`, or `repeat_interleave` on the head dim. When
`n_rep == 1` (standard multi-head attention) this should be a no-op.

### model.attention

```
scores = q @ k^T / sqrt(hd)   # (B, nh, Tq, Tk)
apply causal mask             # query i attends to key j <= j0 + i, j0 = Tk - Tq
softmax over the key axis
weighted sum with v           # -> (B, nh, Tq, hd)
```

### model.CausalSelfAttention.forward

```
1. project x with c_attn, split into q (nh*hd), k (nkv*hd), v (nkv*hd)
2. reshape each into (B, heads, T, hd) — note q has nh heads, k/v nkv
3. if self.config.rope: apply RoPE to q and k using self.rope_cos[:T] and
   self.rope_sin[:T] — do this BEFORE repeat_kv (rotate the nkv k-heads, not the
   expanded nh) and NEVER rotate v
4. expand k, v to nh heads with repeat_kv
5. y = attention(q, k, v, causal=True)
6. reshape y back to (B, T, C) and apply c_proj
```

### model.GPT.forward

```
1. x = wte(idx). If NOT config.rope, also add wpe(positions) where
   positions = [0, 1, ..., T-1]. If config.rope, add nothing here — position
   enters inside attention via the rotary q/k rotation.
2. run through each block in self.transformer.h
3. final layernorm ln_f
4. project to vocab with lm_head
```

Position handling becomes interesting in Stage 2 when you decode one token at a
time — the position of the new token is the cache length, not zero. Write this
now in a way you'll be able to generalize.

---

## Stage 2 — `kv_cache.py`

### kv_cache.KVCache.append

Write into `self.k[layer_idx]` / `self.v[layer_idx]` at the slice
`[:, :, self.length : self.length + Tnew]`. Only advance `self.length` once
(after the LAST layer), or track it so repeated per-layer appends in one step
don't over-count. A clean approach: advance `length` in `decode_step` after all
layers, and have `append` use a passed-in write offset. Choose a scheme and keep
it consistent — the tests only check the returned tensors and final logits, not
your bookkeeping style.

### kv_cache.decode_step

Walk the model's submodules manually (you can't call `model.forward` — it has no
cache):

```
1. tok_emb = wte(idx); pos_emb = wpe(arange(start_pos, start_pos+Tq))
2. x = tok_emb + pos_emb
3. for each layer i / Block b:
     a. h = b.ln_1(x); project with b.attn.c_attn; split into q,k,v
     b. reshape q -> (B,nh,Tq,hd); k,v -> (B,nkv,Tq,hd)
     c. k_full, v_full = cache.append(i, k, v)
     d. expand k_full,v_full with repeat_kv; y = attention(q, k_full, v_full, causal=True)
     e. reshape y, apply b.attn.c_proj, add residual
     f. x = x + b.mlp(b.ln_2(x))
4. x = ln_f(x); logits = lm_head(x)
```

The causal mask in `attention` with `Tq=1`, `Tk=cache_len` lets the new token
attend to everything cached — verify you understand why before moving on.

---

## Stage 3 — `generate.py`

### generate.generate

```
1. allocate a KVCache sized for T0 + max_new_tokens
2. prefill: logits = decode_step(model, idx, cache, start_pos=0)
3. loop max_new_tokens times:
     - next_tok = sample_next(logits[:, -1], ...)   # (B,)
     - append next_tok to the running sequence
     - logits = decode_step(model, next_tok[:, None], cache, start_pos=cache.length)
4. return the full sequence
```

---

## Stage 4 — `batching.py`

### batching.batched_decode_step

```
1. positions = cache.lengths[rows]                      # (n_active,)
   tok_emb = wte(idx); pos_emb = wpe(positions)[:, None]; x = tok_emb + pos_emb
2. Lmax = max length among rows. Build a key-padding mask of shape
   (n_active, Lmax): True where key position < that row's length.
3. for each layer / Block b:
     a. project ln_1(x) with b.attn.c_attn -> q,k,v ; reshape:
        q -> (n_active, nh, 1, hd); k,v -> (n_active, nkv, 1, hd)
     b. cache.write(layer, rows, k, v)                  # store new K/V
     c. gather full K/V for these rows up to Lmax:
        k_full = cache.k[layer, rows, :, :Lmax] ; same for v   # (n_active, nkv, Lmax, hd)
     d. expand with repeat_kv -> (n_active, nh, Lmax, hd)
     e. attention with the padding mask. Two ok options:
          - F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=mask4d)
          - or your own scores -> masked_fill(-inf) -> softmax -> @v
        (Do NOT use causal=True here: with Tq=1 the only constraint is the
         padding mask; the new token may see all valid cached positions.)
     f. reshape, c_proj, residual; then x = x + b.mlp(b.ln_2(x))
4. x = ln_f(x); logits = lm_head(x)[:, -1]              # (n_active, vocab)
5. cache.lengths[rows] += 1
```

### batching.ContinuousBatchingScheduler._admit

For each free slot while `waiting` is non-empty:

```
- pop a Request, allocate a slot, set req.slot
- prefill: feed the prompt through the cache so the slot holds the prompt's K/V
  and you have logits for the first generated token. You can prefill one token at
  a time with batched_decode_step (rows=[slot]) over the prompt, keeping the LAST
  logits; that reuses your decode primitive and keeps this simple.
- sample the first token from those logits (use the request's own
  generator/sampling params), append it to req.generated
- move req from waiting to running; mark done if it hit a limit
```

### batching.ContinuousBatchingScheduler.step

```
- Build idx (n_running, 1) from each running request's LAST token.
- logits = batched_decode_step(model, idx, cache, rows=[r.slot ...])
- For each running request, sample its next token from its row of logits using
  that request's own generator + sampling params, append it.
- Retire requests that reached max_new_tokens: mark done, release their slot,
  move to finished.
- Call self._admit() to backfill freed slots with waiting requests.
```

### batching.ContinuousBatchingScheduler.run

Typical loop: admit, then `while running: step()`.

---

## Stage 5b — `quant.py`

### quant.QuantizedLinear.from_linear

`q, scale = quantize_per_channel_int8(linear.weight.data)`; copy bias (if any);
return `cls(q, scale, bias)`.

---

## Stage 5c — `flash.py`

### flash.flash_attention

Per query block, streaming over key blocks. Maintain for each query row:
`m` (running max logit, init `-inf`), `l` (running sum of exp, init 0),
`acc` (running output accumulator, init 0).

```
For each key/value block (Kj, Vj):
    S   = (Qi @ Kj^T) / sqrt(hd)              # (block_q, block_k)
    apply causal mask to S where needed (skip fully-masked future blocks)
    m_new = max(m, rowmax(S))
    p   = exp(S - m_new)                      # rescaled probabilities
    l   = l * exp(m - m_new) + rowsum(p)
    acc = acc * exp(m - m_new)[:, None] + p @ Vj
    m   = m_new
Output_i = acc / l[:, None]
```

The `exp(m_old - m_new)` rescaling is the crux — it retroactively corrects the
earlier partial sums when a later block raises the running max. Get that right
and your output matches a single full softmax exactly.

---

## Stage 1+ — `rope.py`

### rope.build_rope_cache

```
1. inv_freq = 1 / base ** (arange(0, head_dim, 2) / head_dim)    # (hd/2,)
2. angles   = outer(arange(seq_len), inv_freq)                   # (seq_len, hd/2)
3. emb      = cat([angles, angles], dim=-1)                      # (seq_len, hd)
   (duplicated so dim i and dim i+hd/2 share a frequency — that's the half-split
    pairing rotate_half assumes)
4. return emb.cos(), emb.sin()
```

Compute angles in float32 for precision, then cast to `dtype` at the end.

### Wiring RoPE into the model

Not needed to pass `tests/test_rope.py`, and intentionally not wired into
`model.py` so you can do it yourself:

- drop `wpe` from GPT; build ONE cache of length `block_size` up front
- inside attention, after the head reshape and BEFORE the `q@k^T` score, do
  `q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)` (v is NOT rotated)
- with a KV cache (Stage 2), slice cos/sin at the token's ABSOLUTE position
  (`start_pos .. start_pos+T`) — same offset logic as the position ids.
