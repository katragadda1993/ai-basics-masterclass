"""Session 021 lab: toy BPE, embeddings and a tiny causal language model.

This is an educational reference, not a production tokenizer or LLM.
Run:
    python lab.py
    python lab.py --device cpu
    python lab.py --device cuda
"""
import argparse
import math
import re
from collections import Counter

import torch
from torch import nn
from torch.nn import functional as F

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class ToyBPETokenizer:
    """Educational BPE over whitespace-pretokenized words with a ▁ boundary marker."""
    def __init__(self, merges, vocab):
        self.merges = list(merges)
        self.vocab = list(vocab)
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.pad_id = self.token_to_id["<pad>"]
        self.bos_id = self.token_to_id["<bos>"]
        self.eos_id = self.token_to_id["<eos>"]
        self.unk_id = self.token_to_id["<unk>"]

    @staticmethod
    def _split_words(text):
        return re.findall(r"\S+", text.strip())

    @classmethod
    def train(cls, texts, num_merges=80):
        counts, alphabet = Counter(), set()
        for text in texts:
            for word in cls._split_words(text):
                symbols = ["▁"] + list(word)
                counts[tuple(symbols)] += 1
                alphabet.update(symbols)
        sequences, merges = dict(counts), []
        for _ in range(num_merges):
            pair_counts = Counter()
            for symbols, freq in sequences.items():
                for pair in zip(symbols, symbols[1:]):
                    pair_counts[pair] += freq
            if not pair_counts:
                break
            best_count = max(pair_counts.values())
            best_pair = min(pair for pair, count in pair_counts.items()
                            if count == best_count)
            if best_count < 2:
                break
            merged = best_pair[0] + best_pair[1]
            merges.append((best_pair, merged))
            updated = Counter()
            for symbols, freq in sequences.items():
                out, i = [], 0
                while i < len(symbols):
                    if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == best_pair:
                        out.append(merged)
                        i += 2
                    else:
                        out.append(symbols[i])
                        i += 1
                updated[tuple(out)] += freq
            sequences = dict(updated)
        vocab = SPECIALS + sorted(alphabet)
        for _, merged in merges:
            if merged not in vocab:
                vocab.append(merged)
        return cls(merges, vocab)

    def _encode_word(self, word):
        symbols = ["▁"] + list(word)
        for pair, merged in self.merges:
            out, i = [], 0
            while i < len(symbols):
                if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
                    out.append(merged)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            symbols = out
        return [self.token_to_id.get(token, self.unk_id) for token in symbols]

    def encode(self, text, add_special_tokens=True):
        ids = [self.bos_id] if add_special_tokens else []
        for word in self._split_words(text):
            ids.extend(self._encode_word(word))
        if add_special_tokens:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        pieces = []
        for idx in ids:
            token = self.vocab[int(idx)]
            if token in SPECIALS:
                if not skip_special_tokens:
                    pieces.append(token)
            else:
                pieces.append(token)
        return "".join(pieces).replace("▁", " ").strip()


def build_corpus():
    devices = ["dma", "pcie", "soundwire", "codec", "axi"]
    events = ["retry", "timeout", "underrun", "parity", "stall"]
    states = ["normal", "rising", "bursty", "stable", "recovered"]
    actions = ["check queue", "inspect clock", "review trace", "measure latency"]
    lines = []
    for i in range(240):
        device = devices[i % len(devices)]
        event = events[(i // 2 + i) % len(events)]
        state = states[(i // 3 + i // 7) % len(states)]
        action = actions[(i // 5 + i) % len(actions)]
        level = ["low", "medium", "high"][(i // 4) % 3]
        lines.append(f"{device} {event} rate {level} state {state} then {action}")
    return lines


def make_stream(lines, tokenizer):
    ids = []
    for line in lines:
        ids.extend(tokenizer.encode(line))
    return torch.tensor(ids, dtype=torch.long)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        b, t, d = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(b, t, d))


class TinyBlock(nn.Module):
    def __init__(self, d_model=48, n_heads=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(),
                                 nn.Linear(4*d_model, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size, context=24, d_model=48, n_heads=4):
        super().__init__()
        self.context = context
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context, d_model)
        self.block = TinyBlock(d_model, n_heads)
        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, idx):
        b, t = idx.shape
        if t > self.context:
            raise ValueError("input exceeds configured context length")
        pos = torch.arange(t, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)[None, :, :]
        return self.lm_head(self.final_ln(self.block(x)))


def sample_batch(stream, batch_size, context, device, generator):
    max_start = len(stream) - context - 1
    if max_start <= 0:
        raise ValueError("stream is too short")
    starts = torch.randint(max_start, (batch_size,), generator=generator)
    x = torch.stack([stream[s:s+context] for s in starts])
    y = torch.stack([stream[s+1:s+context+1] for s in starts])
    return x.to(device), y.to(device)


@torch.inference_mode()
def estimate_loss(model, stream, device, batches=30, batch_size=32):
    model.eval()
    gen = torch.Generator().manual_seed(2021)
    losses = []
    for _ in range(batches):
        x, y = sample_batch(stream, batch_size, model.context, device, gen)
        logits = model(x)
        losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                      y.reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses)


def check_causal_invariance(model, device):
    model.eval()
    torch.manual_seed(77)
    a = torch.randint(0, model.token_embedding.num_embeddings,
                      (1, model.context), device=device)
    b = a.clone()
    cut = model.context // 2
    b[:, cut:] = torch.randint(0, model.token_embedding.num_embeddings,
                               b[:, cut:].shape, device=device)
    with torch.inference_mode():
        ya, yb = model(a), model(b)
    error = (ya[:, :cut] - yb[:, :cut]).abs().max().item()
    model.train()
    return error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--steps", type=int, default=180)
    args = parser.parse_args()
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(21021)
    lines = build_corpus()
    train_lines, val_lines, test_lines = lines[:192], lines[192:216], lines[216:]
    tokenizer = ToyBPETokenizer.train(train_lines, num_merges=80)
    sample = "dma retry rate high state rising then check queue"
    sample_ids = tokenizer.encode(sample)
    round_trip = tokenizer.decode(sample_ids)
    assert round_trip == sample, (round_trip, sample)
    train_stream = make_stream(train_lines, tokenizer)
    val_stream = make_stream(val_lines, tokenizer)
    test_stream = make_stream(test_lines, tokenizer)

    context = 24
    model = TinyLanguageModel(len(tokenizer.vocab), context=context).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    generator = torch.Generator().manual_seed(21022)

    x, y = sample_batch(train_stream, 4, context, device, generator)
    logits = model(x)
    assert logits.shape == (4, context, len(tokenizer.vocab))
    assert torch.equal(y[:, :-1], x[:, 1:])
    initial_val = estimate_loss(model, val_stream, device)

    model.train()
    for _ in range(args.steps):
        xb, yb = sample_batch(train_stream, 32, context, device, generator)
        pred = model(xb)
        loss = F.cross_entropy(pred.reshape(-1, pred.size(-1)), yb.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    final_val = estimate_loss(model, val_stream, device)
    test_loss = estimate_loss(model, test_stream, device)
    test_ppl = math.exp(min(test_loss, 20.0))
    causal_error = check_causal_invariance(model, device)
    assert math.isfinite(final_val) and math.isfinite(test_ppl)
    assert final_val < initial_val, (initial_val, final_val)
    assert causal_error < 1e-6, causal_error

    ids = torch.tensor([[tokenizer.bos_id, sample_ids[1], sample_ids[2]]], device=device)
    emb = model.token_embedding(ids)
    assert emb.shape == (1, 3, model.token_embedding.embedding_dim)

    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    print(f"Documents: train={len(train_lines)}, validation={len(val_lines)}, test={len(test_lines)}")
    print(f"Toy BPE merges learned: {len(tokenizer.merges)}")
    print(f"Toy vocabulary size: {len(tokenizer.vocab)}")
    print(f"Round-trip: {round_trip!r}")
    print(f"Token IDs (sample): {sample_ids}")
    print(f"Embedding lookup shape: {tuple(emb.shape)}")
    print(f"Training stream tokens: {len(train_stream)}")
    print(f"Validation loss: {initial_val:.4f} -> {final_val:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test perplexity: {test_ppl:.3f}")
    print(f"Causal future-token leakage max error: {causal_error:.3e}")
    print("PASS: all Session 021 checks completed.")


if __name__ == "__main__":
    main()
