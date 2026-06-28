"""Stage 1b — assembled model.

No external weights to compare against, so these tests check the structural
invariants a correct GPT forward pass must satisfy: shapes, causality (a future
token can't change an earlier position's logits), weight tying, and grad flow.
"""
import pytest
import torch

from nanoinfer.config import GPTConfig
from nanoinfer.model import GPT, MLP, Block, CausalSelfAttention


@pytest.fixture
def config():
    return GPTConfig()


def test_mlp_shape(config):
    x = torch.randn(2, 5, config.n_embd)
    assert MLP(config)(x).shape == x.shape


def test_attention_module_shape(config):
    x = torch.randn(2, 5, config.n_embd)
    assert CausalSelfAttention(config)(x).shape == x.shape


def test_block_shape_and_residual(config):
    x = torch.randn(2, 5, config.n_embd)
    out = Block(config)(x)
    assert out.shape == x.shape
    # residual stream means output should not equal the sublayer output alone
    assert not torch.allclose(out, x)


def test_gpt_forward_shape(config):
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (3, 7))
    logits = model(idx)
    assert logits.shape == (3, 7, config.vocab_size)


def test_weights_untied_by_default(config):
    # default config has tie_weights=False -> wte and lm_head are separate params
    model = GPT(config)
    assert model.transformer.wte.weight is not model.lm_head.weight


def test_weight_tying_when_enabled():
    model = GPT(GPTConfig(tie_weights=True))
    assert model.transformer.wte.weight is model.lm_head.weight


def test_gpt_is_causal(config):
    """Changing token at position t must not affect logits at positions < t."""
    model = GPT(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        base = model(idx)
        idx2 = idx.clone()
        idx2[0, -1] = (idx2[0, -1] + 1) % config.vocab_size  # change last token
        perturbed = model(idx2)
    torch.testing.assert_close(base[:, :-1], perturbed[:, :-1])


def test_gpt_backward(config):
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (2, 6))
    logits = model(idx)
    logits.sum().backward()
    # every parameter that requires grad should receive one
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no grad for {name}"


@pytest.mark.gpu
def test_gpt_runs_on_gpu(config, device):
    if device.type != "cuda":
        pytest.skip("pass --device cuda to run")
    model = GPT(config).to(device)
    idx = torch.randint(0, config.vocab_size, (4, 16), device=device)
    logits = model(idx)
    assert logits.device.type == "cuda"
    assert logits.shape == (4, 16, config.vocab_size)


# --- RoPE integration (config.rope=True) -----------------------------------
# These check the *wiring*, not the rotary math itself (that's test_rope.py):
# the model runs, stays causal, drops wpe, and isn't capped at block_size.

def test_rope_model_forward_shape():
    cfg = GPTConfig(rope=True)
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 7))
    assert model(idx).shape == (2, 7, cfg.vocab_size)


def test_rope_model_has_no_wpe():
    assert "wpe" not in GPT(GPTConfig(rope=True)).transformer
    assert "wpe" in GPT(GPTConfig(rope=False)).transformer


def test_rope_model_is_causal():
    cfg = GPTConfig(rope=True)
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        base = model(idx)
        idx2 = idx.clone()
        idx2[0, -1] = (idx2[0, -1] + 1) % cfg.vocab_size
        perturbed = model(idx2)
    torch.testing.assert_close(base[:, :-1], perturbed[:, :-1])


def test_rope_lifts_block_size_cap():
    # learned wpe caps sequences at block_size; RoPE only at max_seq_len, so a
    # sequence longer than block_size must still run.
    cfg = GPTConfig(rope=True, block_size=8, max_seq_len=64)
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 32))  # 32 > block_size=8
    assert model(idx).shape == (1, 32, cfg.vocab_size)


@pytest.mark.gpu
def test_rope_model_on_gpu(device):
    # the real point of registering the rope cache as a BUFFER: it must move to
    # the GPU with the model. A plain-attribute cache stays on CPU and this fails.
    if device.type != "cuda":
        pytest.skip("pass --device cuda to run")
    cfg = GPTConfig(rope=True)
    model = GPT(cfg).eval().to(device)
    idx = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
    logits = model(idx)
    assert logits.device.type == "cuda"
    assert logits.shape == (2, 16, cfg.vocab_size)
