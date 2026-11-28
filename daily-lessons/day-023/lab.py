#!/usr/bin/env python3
"""Session 023: tiny chat-template SFT with assistant-only loss masking.

Educational CPU-friendly demonstration. Uses a tiny character-level causal Transformer
and synthetic diagnostic instructions; not a production tokenizer or alignment recipe.
"""
from __future__ import annotations
import argparse, math, random, tempfile
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 23023
SPECIAL = ["<|pad|>", "<|bos|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
SYSTEM = "You are a concise silicon diagnostic assistant."
D_MODEL, N_HEADS, N_LAYERS, BLOCK_SIZE = 32, 4, 1, 192
BATCH_SIZE, PRETRAIN_STEPS, SFT_STEPS = 16, 180, 300


class LoRALinear(nn.Module):
    """Frozen base linear layer plus a trainable low-rank update."""
    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale


def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_examples():
    rows = []
    for port in range(1, 5):
        for retries in range(10):
            for latency in ["low", "high"]:
                status = "NORMAL" if retries == 0 and latency == "low" else "INVESTIGATE"
                rows.append((
                    f"Classify SoundWire port {port}: retries={retries}, latency={latency}.",
                    f"STATUS: {status}",
                ))
    for depth in ["low", "medium", "high"]:
        for latency in ["normal", "elevated", "high"]:
            for mode in ["normal", "stress"]:
                answer = f"SUMMARY: Queue depth is {depth} and latency is {latency}."
                rows.append((
                    f"Summarize DMA telemetry: queue_depth={depth}, latency={latency}, mode={mode}.",
                    answer,
                ))
    meanings = [
        ("OKAY", "The transaction completed successfully."),
        ("SLVERR", "The slave reported an error."),
        ("DECERR", "The interconnect could not decode the address."),
        ("EXOKAY", "The exclusive access completed successfully."),
    ]
    for code, meaning in meanings:
        for bus in ["AXI read", "AXI write", "DMA descriptor"]:
            rows.append((f"Explain response {code} for {bus}.", f"EXPLANATION: {meaning}"))
    for rate in [44100, 48000, 96000, 192000]:
        for channels in [1, 2, 4, 8]:
            for fmt in ["I2S", "TDM"]:
                rows.append((
                    f"Format audio configuration: rate={rate}, channels={channels}, format={fmt}.",
                    f"CONFIG: sample_rate={rate}; channels={channels}; format={fmt}",
                ))
    for count in range(16):
        rows.append((
            f"Return JSON for retry_count={count}.",
            f'{{"status":"investigate","retry_count":{count}}}',
        ))
    rng = random.Random(SEED)
    rng.shuffle(rows)
    n = len(rows)
    a, b = int(0.75 * n), int(0.875 * n)
    return rows[:a], rows[a:b], rows[b:]


class ChatCodec:
    def __init__(self, train_pairs):
        chars = set(SYSTEM)
        for prompt, answer in train_pairs:
            chars.update(prompt)
            chars.update(answer)
        # Fixed printable ASCII coverage avoids fitting a learned vocabulary on held-out data.
        alphabet = [chr(i) for i in range(32, 127)] + ["\n", "\t"]
        self.special = SPECIAL
        self.itos = list(SPECIAL) + [c for c in alphabet if c in chars]
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.special_ids = {s: self.stoi[s] for s in SPECIAL}

    def encode_text(self, text):
        out, i = [], 0
        while i < len(text):
            matched = next((tok for tok in self.special if text.startswith(tok, i)), None)
            if matched:
                out.append(self.stoi[matched])
                i += len(matched)
            else:
                ch = text[i]
                if ch not in self.stoi:
                    raise ValueError(f"OOV character: {ch!r}")
                out.append(self.stoi[ch])
                i += 1
        return out

    def encode_chat(self, prompt, answer=None):
        prefix = (
            f"<|bos|><|system|>{SYSTEM}<|end|>"
            f"<|user|>{prompt}<|end|><|assistant|>"
        )
        ids = self.encode_text(prefix)
        assistant_mask = [False] * len(ids)
        if answer is not None:
            answer_ids = self.encode_text(answer + "<|end|>")
            ids.extend(answer_ids)
            assistant_mask.extend([True] * len(answer_ids))
        return ids, assistant_mask

    def decode(self, ids):
        special_set = set(self.special)
        return "".join(
            self.itos[int(i)] for i in ids
            if self.itos[int(i)] not in special_set
        )


class CausalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def shape(z):
            return z.view(b, t, N_HEADS, c // N_HEADS).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            shape(q), shape(k), shape(v), is_causal=True
        )
        return self.proj(y.transpose(1, 2).contiguous().view(b, t, c))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = CausalAttention()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ff = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL),
            nn.GELU(),
            nn.Linear(4 * D_MODEL, D_MODEL),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class TinyChatLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok = nn.Embedding(vocab, D_MODEL)
        self.pos = nn.Embedding(BLOCK_SIZE, D_MODEL)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYERS)])
        self.ln = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, ids):
        b, t = ids.shape
        if t > BLOCK_SIZE:
            raise ValueError("sequence exceeds context")
        pos = torch.arange(t, device=ids.device)
        x = self.tok(ids) + self.pos(pos)[None, :, :]
        return self.head(self.ln(self.blocks(x)))


def collate(pairs, codec, device, assistant_only=True):
    encoded = [codec.encode_chat(prompt, answer) for prompt, answer in pairs]
    maxlen = min(BLOCK_SIZE, max(len(ids) for ids, _ in encoded))
    xs, ys = [], []
    pad = codec.special_ids["<|pad|>"]
    for ids, assistant_mask in encoded:
        ids = ids[:maxlen]
        assistant_mask = assistant_mask[:maxlen]
        x, target_ids = ids[:-1], ids[1:]
        target_mask = assistant_mask[1:]
        labels = [
            tok if (target_mask[j] or not assistant_only) else -100
            for j, tok in enumerate(target_ids)
        ]
        pad_count = (maxlen - 1) - len(x)
        xs.append(x + [pad] * pad_count)
        ys.append(labels + [-100] * pad_count)
    return torch.tensor(xs, device=device), torch.tensor(ys, device=device)


def batch_loss(model, pairs, codec, device, assistant_only=True):
    x, labels = collate(pairs, codec, device, assistant_only)
    logits = model(x)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )


def train_steps(model, pairs, codec, device, steps, lr, assistant_only):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    losses = []
    rng = random.Random(SEED + steps + (1 if assistant_only else 0))
    for _ in range(steps):
        batch = [pairs[rng.randrange(len(pairs))] for _ in range(BATCH_SIZE)]
        loss = batch_loss(model, batch, codec, device, assistant_only)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return sum(losses[-20:]) / min(20, len(losses))


@torch.no_grad()
def evaluate(model, pairs, codec, device, assistant_only=True):
    model.eval()
    losses = []
    for i in range(0, len(pairs), BATCH_SIZE):
        loss = batch_loss(
            model, pairs[i:i + BATCH_SIZE], codec, device, assistant_only
        )
        losses.append(float(loss))
    return sum(losses) / len(losses)


@torch.inference_mode()
def generate_answer(model, codec, prompt, device, max_new_tokens=96):
    model.eval()
    prefix = (
        f"<|bos|><|system|>{SYSTEM}<|end|>"
        f"<|user|>{prompt}<|end|><|assistant|>"
    )
    prefix_ids = codec.encode_text(prefix)
    ids = torch.tensor([prefix_ids], device=device)
    end_id = codec.special_ids["<|end|>"]
    for _ in range(max_new_tokens):
        logits = model(ids[:, -BLOCK_SIZE:])
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
        if int(next_id.item()) == end_id:
            break
    generated = ids[0, len(prefix_ids):].tolist()
    return codec.decode(generated)


def count_supervised_tokens(pairs, codec):
    total = 0
    for prompt, answer in pairs:
        _, mask = codec.encode_chat(prompt, answer)
        total += sum(mask[1:])
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    device = torch.device("cuda" if use_cuda else "cpu")

    set_seed()
    train_pairs, val_pairs, test_pairs = make_examples()
    codec = ChatCodec(train_pairs)
    model = TinyChatLM(len(codec.itos)).to(device)

    val_random = evaluate(model, val_pairs, codec, device, assistant_only=True)

    # Stage 1: toy base-LM objective over all serialized message tokens.
    pretrain_tail = train_steps(
        model, train_pairs, codec, device, PRETRAIN_STEPS, 0.002, False
    )
    val_pre_sft = evaluate(model, val_pairs, codec, device, assistant_only=True)
    val_full_pre_sft = evaluate(model, val_pairs, codec, device, assistant_only=False)

    # Stage 2: supervise assistant body and end marker only.
    sft_tail = train_steps(
        model, train_pairs, codec, device, SFT_STEPS, 0.001, True
    )
    val_after = evaluate(model, val_pairs, codec, device, assistant_only=True)
    val_full_after = evaluate(model, val_pairs, codec, device, assistant_only=False)
    test_after = evaluate(model, test_pairs, codec, device, assistant_only=True)

    # Verify that prompt targets are masked and assistant/end targets are retained.
    ids, mask = codec.encode_chat(train_pairs[0][0], train_pairs[0][1])
    answer_len = len(codec.encode_text(train_pairs[0][1] + "<|end|>"))
    assert not any(mask[:-answer_len])
    assert all(mask[-answer_len:])

    x, labels = collate([train_pairs[0]], codec, device, assistant_only=True)
    supervised = int((labels != -100).sum())
    assert 0 < supervised < labels.numel()

    # Template round-trip and role-boundary checks.
    decoded = codec.decode(ids)
    assert train_pairs[0][0] in decoded and train_pairs[0][1] in decoded

    # Checkpoint round-trip on a fixed input using a self-created trusted artifact.
    model.eval()
    sample, _ = collate([test_pairs[0]], codec, device, assistant_only=True)
    with torch.inference_mode():
        before = model(sample)
    with tempfile.TemporaryDirectory() as td:
        checkpoint = Path(td) / "sft_model.pt"
        torch.save({
            "model": model.state_dict(),
            "vocab": codec.itos,
            "special_tokens": SPECIAL,
            "step": SFT_STEPS,
        }, checkpoint)
        payload = torch.load(
            checkpoint, map_location=device, weights_only=True
        )
        restored = TinyChatLM(len(codec.itos)).to(device)
        restored.load_state_dict(payload["model"])
        restored.eval()
        with torch.inference_mode():
            after = restored(sample)
        restore_error = float((before - after).abs().max())

    # LoRA sanity check: zero-initialized B makes the adapter initially a no-op.
    base = nn.Linear(64, 64).to(device)
    adapter = LoRALinear(base, rank=4, alpha=8).to(device)
    probe = torch.randn(3, 64, device=device)
    with torch.no_grad():
        lora_init_error = float((base(probe) - adapter(probe)).abs().max())
    lora_trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    lora_full = 64 * 64 + 64

    sample_prompt = train_pairs[0][0]
    generated = generate_answer(model, codec, sample_prompt, device)
    parameters = sum(p.numel() for p in model.parameters())
    heldout_json_examples = sum(answer.startswith("{") for _, answer in test_pairs)

    assert val_after < val_pre_sft, (val_pre_sft, val_after)
    assert math.isfinite(test_after) and test_after > 0
    assert restore_error == 0.0
    assert lora_init_error < 1e-6 and lora_trainable < lora_full
    assert heldout_json_examples >= 1

    print(f"device={device} torch={torch.__version__}")
    print(f"train/validation/test examples={len(train_pairs)}/{len(val_pairs)}/{len(test_pairs)}")
    print(f"vocabulary={len(codec.itos)} parameters={parameters}")
    print(f"assistant_supervised_tokens_train={count_supervised_tokens(train_pairs, codec)}")
    print(
        f"validation_assistant_loss random={val_random:.4f} "
        f"pretrained={val_pre_sft:.4f} sft={val_after:.4f}"
    )
    print(
        f"validation_full_sequence_loss after_pretrain={val_full_pre_sft:.4f} "
        f"after_sft={val_full_after:.4f}"
    )
    print(f"test_assistant_loss={test_after:.4f}")
    print(f"pretrain_tail_loss={pretrain_tail:.4f} sft_tail_loss={sft_tail:.4f}")
    print(f"heldout_json_schema_examples={heldout_json_examples} (labels checked; not model-output accuracy)")
    print(f"checkpoint_restore_max_logit_error={restore_error}")
    print(
        f"LoRA rank=4 trainable={lora_trainable} vs full_linear={lora_full} "
        f"initial_delta_error={lora_init_error}"
    )
    print(f"sample_prompt={sample_prompt}")
    print(f"greedy_generated_answer={generated!r} (seen training prompt)")
    print(f"sample_supervised_targets={supervised} total_shifted_positions={labels.numel()}")
    print("ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
