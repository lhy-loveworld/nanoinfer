# nanoinfer — LLM inference from scratch

A hands-on, test-driven path through the parts of an LLM serving stack that an
**inference / model-serving** engineer actually owns. Every component ships as a
skeleton with a thorough spec and a failing test suite — you implement the
bodies, the tests tell you when each piece is correct.

The emphasis is deliberately on **inference**, not training: a thin modeling
substrate, then KV caching, sampling, continuous batching, and serving-grade
optimizations (speculative decoding, quantization, tiled attention).

> Built as interview prep for an inference-focused research-engineering role.
> The tests are the spec; the docstrings are the hints.

## Layout

```
src/nanoinfer/
  config.py        GPTConfig (given) — note n_kv_head < n_head enables GQA
  model.py         Stage 1: attention (GQA), MLP, Block, GPT.forward
  kv_cache.py      Stage 2: KVCache + incremental decode_step
  generate.py      Stage 3: temperature / top-k / top-p / sampling loop
  batching.py      Stage 4: continuous (iteration-level) batching scheduler
  speculative.py   Stage 5a: draft + verify speculative decoding
  quant.py         Stage 5b: INT8 per-channel + FP8 quantization
  flash.py         Stage 5c: tiled attention with online softmax
  rope.py          Stage 1+: rotary position embeddings (RoPE)
tests/             one test module per source module
```

## The progression

Each stage builds on the previous one — later stages reuse, not rewrite, your
earlier code (e.g. KV-cache decode reuses your hand-written `attention`; the
batching contract is checked against your `generate`).

| Stage | You implement | The test pins down | Key inference idea |
|------:|---------------|--------------------|--------------------|
| **1** | `attention`, `repeat_kv`, `MLP/Block/GPT.forward` | matches `F.scaled_dot_product_attention`; causality; grad flow | GQA shrinks the KV cache; attention with `Tq != Tk` |
| **2** | `KVCache.append`, `decode_step` | cached decode == full recompute, token-for-token | the cache *is* the inference speedup; position offsets |
| **3** | `apply_temperature`, `top_k_filter`, `top_p_filter`, `sample_next`, `generate` | exact filter semantics; greedy == manual loop | nucleus boundary bugs; reproducible sampling |
| **4** | `batched_decode_step`, `Scheduler.step/run/_admit` | batched output == solo `generate`; never exceeds `max_batch` | iteration-level scheduling; no head-of-line blocking |
| **5a** | `draft_k`, `verify`, `speculative_generate` | speculative-greedy == plain greedy target | accept/reject; K+1 tokens per target forward |
| **5b** | `quantize_per_channel_int8`, `QuantizedLinear`, `fp8_roundtrip` | bounded round-trip + linear error | memory-bandwidth-bound decode; per-channel scales |
| **5c** | `flash_attention` | matches SDPA incl. ragged tail | online softmax; O(T) memory streaming |
| **1+** | `rotate_half`, `build_rope_cache`, `apply_rope` | relative-position invariance; norm preserved | rotary embeddings; lifts the `block_size` cap |

Suggested order: **1 → 2 → 3 → 4**, then pick among **5a/5b/5c** in any order.
RoPE (**1+**) is a self-contained side quest — do it any time after Stage 1.

## Setup

Requires a recent PyTorch. For an RTX 50-series (Blackwell, `sm_120`) GPU use the
CUDA 12.8 build:

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or plain `pip install torch` on CPU/older GPUs
pip install -e ".[dev]"
```

## Running the tests

```bash
pytest                              # whole suite on CPU (fast, deterministic)
pytest tests/test_attention.py      # one stage at a time
pytest --device cuda                # run on your GPU
pytest -m "not perf"                # skip benchmarks
pytest -m perf --device cuda        # only the GPU timing tests
```

Everything starts **red** (`NotImplementedError`). Work one module top-to-bottom;
green means you matched the reference behavior.

## Watch it learn real text

Once Stages 1–3 are green, `train.py` trains a byte-level GPT and prints samples —
an end-to-end smoke test of your forward pass, KV-cache decode, and sampling:

```bash
python train.py --device cuda                 # downloads tinyshakespeare (offline fallback bundled)
python train.py --device cuda --max-iters 3000 --n-embd 256
```

A 0.8M-param model reaches val loss ~2.2 in ~12s on an RTX 5080 and starts
emitting Shakespeare-shaped text. (Generation is capped at `block_size` — learned
absolute position embeddings can't extrapolate; RoPE/sliding-window is the fix.)

## Reference solutions

A complete implementation lives on the **`reference-solution`** branch. Attempt a
stage yourself first, then diff:

```bash
git diff master reference-solution -- src/nanoinfer/model.py    # one module
git diff master reference-solution                              # everything
```

The branch differs from `master` only in `src/nanoinfer/*.py` (the bodies) — tests
and tooling are identical, so the diff is purely "your job vs. one correct answer."

## How to use this for interview prep

- Treat each docstring as the problem statement and the test as the acceptance
  criteria — implement without peeking at reference repos first.
- After a stage is green, re-derive the *why*: why does GQA cut cache size by
  `n_head / n_kv_head`? why must speculative-greedy be exact regardless of draft
  quality? why does the online-softmax rescale term exist?
- Stretch goals worth adding once the suite is green: paged attention, prefix/KV
  reuse across requests, stochastic (distribution-preserving) speculative
  decoding, chunked prefill, and an INT8 matmul that accumulates in INT32.

## Reference material

If you want to check your approach against polished implementations:
[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT),
[rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch), and the
[vLLM](https://github.com/vllm-project/vllm) internals for batching/paging.
