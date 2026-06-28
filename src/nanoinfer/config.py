"""Model configuration. This file is complete — nothing to implement here."""
from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Hyperparameters for the GPT model.

    Defaults describe a tiny model that trains/runs in milliseconds on CPU, so
    the test suite stays fast. `n_kv_head < n_head` enables grouped-query
    attention (GQA), which is the lever that shrinks the KV cache in real
    serving systems — keep it in mind throughout the inference stages.
    """

    vocab_size: int = 256       # byte-level vocab keeps things self-contained
    block_size: int = 64        # maximum context length
    n_layer: int = 2
    n_head: int = 4             # number of query heads
    n_kv_head: int = 2          # number of key/value heads (GQA when < n_head)
    n_embd: int = 32            # embedding / residual stream width
    dropout: float = 0.0
    # No bias anywhere (linears + norms): modern decoder LLMs (LLaMA, Mistral,
    # ...) drop it — it costs params/bandwidth for negligible quality.
    # Tie the token embedding (wte) and the output projection (lm_head) to share
    # one weight matrix. Saves vocab*n_embd params and is standard in smaller
    # models (GPT-2, Gemma); large models (LLaMA) leave them untied, so default off.
    tie_weights: bool = False
    # Use rotary position embeddings instead of a learned wpe table. When True,
    # GPT drops wpe entirely and attention rotates q/k by position (see rope.py).
    # This is what modern decoders (LLaMA, Mistral, ...) use, and it lifts the
    # block_size generation cap — context is then bounded by max_seq_len.
    rope: bool = False
    max_seq_len: int = 1024     # length of the precomputed rotary cache (rope only)

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        if self.rope:
            assert self.head_dim % 2 == 0, "RoPE requires an even head_dim"

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def n_rep(self) -> int:
        """How many query heads share each KV head (the GQA group size)."""
        return self.n_head // self.n_kv_head
