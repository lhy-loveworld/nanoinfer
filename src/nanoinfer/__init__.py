"""nanoinfer — LLM inference from scratch.

A progression of skeletons + tests. You implement the bodies marked with
`raise NotImplementedError`; the tests tell you when each piece is correct.

Stages:
    1. model        — attention (GQA), MLP, Block, GPT forward pass
    2. kv_cache     — incremental decoding with a key/value cache
    3. generate     — sampling (greedy/temperature/top-k/top-p) + the decode loop
    4. batching     — continuous batching scheduler
    5. speculative  — draft+verify speculative decoding
       quant        — INT8/FP8 weight quantization
       flash        — tiled, memory-aware attention
"""

from .config import GPTConfig

__all__ = ["GPTConfig"]
