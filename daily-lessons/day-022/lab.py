#!/usr/bin/env python3
"""Session 022: a tiny character-level GPT training and decoding lab.

Educational CPU-friendly implementation. Character tokenization keeps the
training loop visible; Session 021 covers production subword tokenizer contracts.
"""
from __future__ import annotations

import argparse
import math
import random
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 22022
BLOCK_SIZE = 64
BATCH_SIZE = 32
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.0
MAX_STEPS = 220
ACCUM_STEPS = 2
WARMUP_STEPS = 20
BASE_LR = 3e-3
MIN_LR_RATIO = 0.10


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_corpus():
    """Create deterministic diagnostic-style documents with varied values."""
    templates = [
        "soundwire port {port} retry count {trend} after clock {clock}.\n",
        "dma queue depth {depth} and latency {latency} under {mode} traffic.\n",
        "pcie replay counter {trend} while link speed remains {speed}.\n",
        "codec audio frames are {state} at sample rate {rate} khz.\n",
        "axi response errors {trend} when master is {master}.\n",
        "normal traffic keeps retry counters {state} and queue depth {depth}.\n",
        "high queue depth can increase dma latency during {mode} traffic.\n",
        "soundwire frame status is {state} after bank {bank} switch.\n",
    ]
    values = {
        "port": [1, 2, 3, 4],
        "trend": ["low", "stable", "rising", "high"],
        "clock": ["normal", "reduced", "stable"],
        "depth": ["low", "medium", "high"],
        "latency": ["normal", "elevated", "high"],
        "mode": ["normal", "stress", "burst"],
        "speed": ["gen3", "gen4", "gen5"],
        "state": ["stable", "clean", "noisy"],
        "rate": [44, 48, 96],
        "master": ["active", "stalled", "idle"],
        "bank": [0, 1],
    }
    docs = []
    for i in range(240):
        template = templates[i % len(templates)]
        row = {
            key: choices[(i * (len(key) + 3) + (i // 7)) % len(choices)]
            for key, choices in values.items()
        }
        docs.append(template.format(**row))
    rng = random.Random(SEED)
    rng.shuffle(docs)
    # Split by document before building streams; no window crosses split sets.
    return docs[:192], docs[192:216], docs[216:]


class CharCodec:
    def __init__(self, texts):
        chars = sorted(set("".join(texts)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = chars

    def encode(self, text):
        unknown = sorted(set(text) - set(self.stoi))
        if unknown:
            raise ValueError(f"unseen characters in held-out text: {unknown}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def stream_docs(docs, codec):
    return torch.tensor(codec.encode("".join(docs)), dtype=torch.long)


class MultiHeadCausalAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size=BLOCK_SIZE):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(block_size, D_MODEL)
        self.blocks = nn.Sequential(
            *[Block(D_MODEL, N_HEADS, DROPOUT) for _ in range(N_LAYERS)]
        )
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, vocab_size, bias=False)
        # Tie input embeddings and output projection weights.
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        if t > self.block_size:
            raise ValueError(f"sequence length {t} exceeds block size {self.block_size}")
        positions = torch.arange(t, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)[None, :, :]
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0,
                 top_k=None, top_p=1.0, greedy=False):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            next_logits = logits[:, -1, :] / temperature
            if greedy:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                if top_k is not None:
                    k = min(int(top_k), next_logits.size(-1))
                    values, _ = torch.topk(next_logits, k)
                    cutoff = values[:, [-1]]
                    next_logits = next_logits.masked_fill(
                        next_logits < cutoff, -float("inf")
                    )
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_logits, descending=True
                    )
                    sorted_probs = torch.softmax(sorted_logits, dim=-1)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    remove = cumulative - sorted_probs > top_p
                    sorted_logits = sorted_logits.masked_fill(
                        remove, -float("inf")
                    )
                    filtered = torch.full_like(next_logits, -float("inf"))
                    filtered.scatter_(1, sorted_indices, sorted_logits)
                    next_logits = filtered
                probs = torch.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


def get_batch(data, batch_size, block_size, device, generator):
    max_start = data.numel() - block_size - 1
    if max_start <= 0:
        raise ValueError("stream is shorter than block_size + 1")
    starts = torch.randint(max_start, (batch_size,), generator=generator)
    x = torch.stack([data[s:s + block_size] for s in starts])
    y = torch.stack([data[s + 1:s + block_size + 1] for s in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, device, seed=123, batches=12):
    model.eval()
    out = {}
    for name, data in (("train", train_data), ("validation", val_data)):
        gen = torch.Generator().manual_seed(seed)
        losses = []
        for _ in range(batches):
            x, y = get_batch(data, BATCH_SIZE, BLOCK_SIZE, device, gen)
            _, loss = model(x, y)
            losses.append(float(loss))
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def lr_factor(step):
    if step < WARMUP_STEPS:
        return float(step + 1) / float(WARMUP_STEPS)
    progress = min(
        1.0,
        (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS),
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine


def train(device):
    set_seed(SEED)
    train_docs, val_docs, test_docs = make_corpus()
    codec = CharCodec(train_docs)
    train_data = stream_docs(train_docs, codec)
    val_data = stream_docs(val_docs, codec)
    test_data = stream_docs(test_docs, codec)

    model = TinyGPT(len(codec.itos)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=BASE_LR, weight_decay=0.1
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    generator = torch.Generator().manual_seed(SEED + 1)

    initial = estimate_loss(model, train_data, val_data, device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    last_grad_norm = None

    for step in range(MAX_STEPS):
        for _ in range(ACCUM_STEPS):
            x, y = get_batch(
                train_data, BATCH_SIZE, BLOCK_SIZE, device, generator
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                _, loss = model(x, y)
                scaled_loss = loss / ACCUM_STEPS
            scaler.scale(scaled_loss).backward()

        scaler.unscale_(optimizer)
        last_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        )
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        # Do not advance the LR schedule if GradScaler skipped the optimizer step.
        if scaler.get_scale() >= old_scale:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if (step + 1) % 55 == 0:
            metrics = estimate_loss(model, train_data, val_data, device)
            print(
                f"step={step+1:3d} train={metrics['train']:.4f} "
                f"val={metrics['validation']:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.6f}"
            )

    final = estimate_loss(model, train_data, val_data, device)
    model.eval()
    with torch.inference_mode():
        test_gen = torch.Generator().manual_seed(SEED + 2)
        test_losses = []
        for _ in range(16):
            x, y = get_batch(test_data, BATCH_SIZE, BLOCK_SIZE, device, test_gen)
            _, loss = model(x, y)
            test_losses.append(float(loss))
    test_loss = sum(test_losses) / len(test_losses)
    perplexity = math.exp(min(test_loss, 20.0))

    # Checkpoint round-trip. The demo uses a self-created, trusted local artifact.
    with tempfile.TemporaryDirectory() as td:
        ckpt_path = Path(td) / "tiny_gpt.pt"
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": MAX_STEPS,
            "vocab": codec.itos,
            "config": {
                "block_size": BLOCK_SIZE,
                "d_model": D_MODEL,
                "n_heads": N_HEADS,
                "n_layers": N_LAYERS,
            },
        }
        torch.save(payload, ckpt_path)
        restored = TinyGPT(len(codec.itos)).to(device)
        restored_payload = torch.load(
            ckpt_path, map_location=device, weights_only=True
        )
        restored.load_state_dict(restored_payload["model"])
        restored.eval()
        sample = torch.tensor([codec.encode("dma queue")], device=device)
        with torch.inference_mode():
            original_logits, _ = model(sample)
            restored_logits, _ = restored(sample)
        checkpoint_error = float(
            (original_logits - restored_logits).abs().max()
        )

    prompt = torch.tensor([codec.encode("dma ")], device=device)
    greedy_ids = model.generate(prompt, 48, greedy=True)
    sampled_ids = model.generate(
        prompt, 48, temperature=0.8, top_k=12, top_p=0.9
    )

    parameter_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    assert parameter_count == 106560
    assert final["train"] < initial["train"], (initial, final)
    assert final["validation"] < initial["validation"], (initial, final)
    assert checkpoint_error < 1e-6, checkpoint_error
    assert len(greedy_ids[0]) == len(prompt[0]) + 48
    assert len(sampled_ids[0]) == len(prompt[0]) + 48
    assert math.isfinite(test_loss) and test_loss > 0

    return {
        "vocab_size": len(codec.itos),
        "parameters": parameter_count,
        "train_docs": len(train_docs),
        "validation_docs": len(val_docs),
        "test_docs": len(test_docs),
        "train_tokens": train_data.numel(),
        "initial": initial,
        "final": final,
        "test_loss": test_loss,
        "perplexity": perplexity,
        "checkpoint_error": checkpoint_error,
        "last_grad_norm": last_grad_norm,
        "greedy_text": codec.decode(greedy_ids[0].tolist()),
        "sampled_text": codec.decode(sampled_ids[0].tolist()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or
        (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    print(f"device={device} torch={torch.__version__}")
    result = train(device)
    print("\nRESULTS")
    for key, value in result.items():
        print(f"{key}: {value}")
    print("ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
