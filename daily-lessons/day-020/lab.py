"""Session 020 lab: implement scaled dot-product attention and a tiny transformer block.

Run:
    python lab.py
    python lab.py --device cpu
    python lab.py --device cuda

The lab checks:
- manual attention agrees with PyTorch SDPA
- causal masking prevents future-token leakage
- multi-head output and parameter shapes are correct
- a residual transformer block returns finite values
"""
import argparse
import math
import torch
from torch import nn
from torch.nn import functional as F


def manual_attention(q, k, v, causal=False):
    """q/k/v: [batch, heads, sequence, head_dim]."""
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

    if causal:
        query_len, key_len = scores.size(-2), scores.size(-1)
        if query_len != key_len:
            raise ValueError("This teaching helper expects equal query/key lengths.")
        future = torch.triu(
            torch.ones(query_len, key_len, dtype=torch.bool, device=scores.device),
            diagonal=1,
        )
        scores = scores.masked_fill(future, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=32, num_heads=4, dropout=0.0):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, causal=True):
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x).view(
            batch, seq_len, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # SDPA boolean mask semantics: True means the position is allowed.
        mask = None
        if causal:
            mask = torch.ones(
                seq_len, seq_len, dtype=torch.bool, device=x.device
            ).tril()

        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch, seq_len, self.d_model
        )
        return self.out(attended)


class TransformerBlock(nn.Module):
    """A compact pre-LayerNorm decoder-style block."""
    def __init__(self, d_model=32, num_heads=4, ff_mult=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x), causal=True))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


def check_manual_attention(device):
    torch.manual_seed(20)
    q = torch.randn(2, 3, 7, 8, device=device)
    k = torch.randn(2, 3, 7, 8, device=device)
    v = torch.randn(2, 3, 7, 8, device=device)

    manual_out, manual_weights = manual_attention(q, k, v, causal=True)
    allowed = torch.ones(7, 7, dtype=torch.bool, device=device).tril()
    sdpa_out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=allowed, dropout_p=0.0
    )
    max_error = (manual_out - sdpa_out).abs().max().item()
    assert max_error < 2e-5, f"manual attention mismatch: {max_error}"
    assert torch.allclose(
        manual_weights.triu(1), torch.zeros_like(manual_weights.triu(1))
    )
    return max_error


def check_causal_no_leakage(device):
    torch.manual_seed(21)
    attention = MultiHeadSelfAttention(d_model=32, num_heads=4).to(device)
    attention.eval()
    x1 = torch.randn(1, 8, 32, device=device)
    x2 = x1.clone()
    x2[:, 5:, :] = torch.randn_like(x2[:, 5:, :]) * 10.0

    with torch.inference_mode():
        y1 = attention(x1, causal=True)
        y2 = attention(x2, causal=True)
        y_noncausal_1 = attention(x1, causal=False)
        y_noncausal_2 = attention(x2, causal=False)

    causal_error = (y1[:, :5] - y2[:, :5]).abs().max().item()
    assert causal_error < 1e-6, f"causal leakage detected: {causal_error}"

    noncausal_change = (
        y_noncausal_1[:, :5] - y_noncausal_2[:, :5]
    ).abs().max().item()
    assert noncausal_change > 1e-7
    return causal_error, noncausal_change


def check_transformer_block(device):
    torch.manual_seed(22)
    block = TransformerBlock(d_model=32, num_heads=4).to(device)
    x = torch.randn(4, 12, 32, device=device)
    y = block(x)
    assert y.shape == (4, 12, 32)
    assert torch.isfinite(y).all()
    parameter_count = sum(p.numel() for p in block.parameters())
    return tuple(y.shape), parameter_count


def worked_attention_example():
    scores = torch.tensor([1.0, 0.0]) / math.sqrt(2.0)
    weights = torch.softmax(scores, dim=0)
    values = torch.tensor([10.0, 20.0])
    result = (weights * values).sum()
    return weights.tolist(), result.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available.")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(2026)
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")

    weights, result = worked_attention_example()
    print(f"Worked example weights: {[round(x, 4) for x in weights]}")
    print(f"Worked example weighted value: {result:.4f}")

    manual_error = check_manual_attention(device)
    print(f"Manual attention vs SDPA max error: {manual_error:.3e}")

    causal_error, noncausal_change = check_causal_no_leakage(device)
    print(f"Causal future-token leakage max error: {causal_error:.3e}")
    print(f"Non-causal earlier-output change: {noncausal_change:.3e}")

    shape, params = check_transformer_block(device)
    print(f"Transformer block output shape: {shape}")
    print(f"Transformer block parameters: {params:,}")
    print("PASS: all Session 020 checks completed.")


if __name__ == "__main__":
    main()
