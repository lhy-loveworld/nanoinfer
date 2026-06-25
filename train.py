"""Tiny byte-level trainer + sampler for nanoinfer.

Once Stages 1–3 are implemented (GPT.forward + generate), this trains the model
on a text file and prints samples so you can watch it learn real text. It uses
only the public API (GPT, GPTConfig, generate), so it doubles as an end-to-end
smoke test of your implementation.

Examples:
    python train.py                          # train on bundled sample text (CPU/GPU auto)
    python train.py --data path/to/corpus.txt --max-iters 3000 --device cuda
    python train.py --device cuda --n-layer 6 --n-embd 256 --block-size 256

Vocabulary is byte-level (0..255), which matches GPTConfig.vocab_size=256, so any
UTF-8 text works with no tokenizer to build.
"""
from __future__ import annotations

import argparse
import time
import urllib.request

import torch
import torch.nn.functional as F

from nanoinfer.config import GPTConfig
from nanoinfer.generate import generate
from nanoinfer.model import GPT

# Bundled fallback text so the script runs fully offline. A model will happily
# overfit this and reproduce it — proof the train/generate loop works end to end.
FALLBACK_TEXT = (
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles\n"
    "And by opposing end them. To die—to sleep,\n"
    "No more; and by a sleep to say we end\n"
    "The heart-ache and the thousand natural shocks\n"
    "That flesh is heir to: 'tis a consummation\n"
    "Devoutly to be wish'd. To die, to sleep;\n"
    "To sleep, perchance to dream—ay, there's the rub:\n"
    "For in that sleep of death what dreams may come,\n"
    "When we have shuffled off this mortal coil,\n"
    "Must give us pause.\n"
) * 8

TINY_SHAKESPEARE = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def load_text(path: str | None) -> str:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    # try a small public-domain corpus; fall back to the bundled text offline
    try:
        with urllib.request.urlopen(TINY_SHAKESPEARE, timeout=5) as r:
            print("downloaded tinyshakespeare")
            return r.read().decode("utf-8")
    except Exception as e:  # noqa: BLE001
        print(f"download failed ({e.__class__.__name__}); using bundled sample text")
        return FALLBACK_TEXT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None, help="path to a UTF-8 text file")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-iters", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-kv-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    text = load_text(args.data)
    data = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"corpus: {len(data)} bytes | train {len(train_data)} | val {len(val_data)}")

    cfg = GPTConfig(
        vocab_size=256,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_embd=args.n_embd,
    )
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params on {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def get_batch(split: str):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - cfg.block_size - 1, (args.batch_size,))
        x = torch.stack([d[i:i + cfg.block_size] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + cfg.block_size] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss(split: str, iters: int = 50) -> float:
        model.eval()
        losses = []
        for _ in range(iters):
            x, y = get_batch(split)
            logits = model(x)
            losses.append(F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1)).item())
        model.train()
        return sum(losses) / len(losses)

    @torch.no_grad()
    def sample(n_tokens: int = 200) -> str:
        model.eval()
        prompt = torch.tensor([[ord("\n")]], dtype=torch.long, device=device)
        # learned absolute position embeddings cap the sequence at block_size,
        # so we can't generate past it (use RoPE / sliding window to go further)
        n_tokens = min(n_tokens, cfg.block_size - prompt.shape[1])
        g = torch.Generator(device=device).manual_seed(args.seed)
        out = generate(model, prompt, n_tokens, temperature=0.8, top_k=40, generator=g)
        model.train()
        return bytes(out[0].tolist()).decode("utf-8", errors="replace")

    t0 = time.time()
    for it in range(1, args.max_iters + 1):
        x, y = get_batch("train")
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if it % args.eval_every == 0 or it == 1:
            vl = estimate_loss("val")
            dt = time.time() - t0
            print(f"iter {it:5d} | train {loss.item():.3f} | val {vl:.3f} | {dt:.1f}s")

    print("\n=== sample ===")
    print(sample())


if __name__ == "__main__":
    main()
