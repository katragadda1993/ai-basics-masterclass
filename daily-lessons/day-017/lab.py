"""Session 017: PyTorch training, autograd, batching and checkpointing.

Run: python lab.py
The lab uses a deterministic synthetic XOR dataset and requires no network access.
"""
from __future__ import annotations

import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 17017
BATCH_SIZE = 64
EPOCHS = 120
LEARNING_RATE = 0.02
HIDDEN_UNITS = 16


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism is useful for teaching and regression tests. Some accelerator
    # kernels may not have deterministic implementations; warn rather than hide it.
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_split(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n, 2)).astype(np.float32)
    # XOR-like quadrant classification: same signs => class 1.
    y = ((x[:, 0] * x[:, 1]) > 0.0).astype(np.float32).reshape(-1, 1)
    return torch.from_numpy(x), torch.from_numpy(y)


def make_datasets():
    # Independent deterministic draws; each split has balanced quadrants in expectation.
    x_train, y_train = make_split(1200, SEED + 1)
    x_val, y_val = make_split(300, SEED + 2)
    x_test, y_test = make_split(300, SEED + 3)
    return (TensorDataset(x_train, y_train), TensorDataset(x_val, y_val),
            TensorDataset(x_test, y_test))


class MLP(nn.Module):
    def __init__(self, hidden_units: int = HIDDEN_UNITS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, 1),  # logits, not sigmoid probabilities
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def make_loader(dataset, *, shuffle: bool, seed: int, device: torch.device):
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    return loader, generator


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total = 0
    all_logits, all_targets = [], []
    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=(device.type == "cuda"))
            yb = yb.to(device, non_blocking=(device.type == "cuda"))
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += float(loss.item()) * xb.size(0)
            total += xb.size(0)
            all_logits.append(logits.cpu())
            all_targets.append(yb.cpu())
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    preds = (torch.sigmoid(logits) >= 0.5).to(torch.int64)
    truth = targets.to(torch.int64)
    tp = int(((preds == 1) & (truth == 1)).sum())
    tn = int(((preds == 0) & (truth == 0)).sum())
    fp = int(((preds == 1) & (truth == 0)).sum())
    fn = int(((preds == 0) & (truth == 1)).sum())
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "loss": total_loss / max(1, total),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def save_checkpoint(path: Path, *, model, optimizer, epoch, best_val_loss,
                    loader_generator, config):
    # Store tensors and primitive containers so weights_only=True can load it.
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader_rng_state": loader_generator.get_state(),
        "config": config,
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, *, model, optimizer, loader_generator, device):
    # weights_only=True avoids general Python object deserialization.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if device.type == "cuda" and checkpoint["cuda_rng_state_all"]:
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state_all"]])
    loader_generator.set_state(checkpoint["loader_rng_state"].cpu())
    return checkpoint


def train_model(model, train_ds, val_ds, *, device, epochs, checkpoint_path, seed):
    model.to(device)
    train_loader, train_generator = make_loader(
        train_ds, shuffle=True, seed=seed, device=device)
    val_loader, _ = make_loader(val_ds, shuffle=False, seed=seed, device=device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    best_val = float("inf")
    best_epoch = -1
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=(device.type == "cuda"))
            yb = yb.to(device, non_blocking=(device.type == "cuda"))
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            # Clip only as a safety bound; the threshold is intentionally generous here.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        val_metrics = evaluate(model, val_loader, device)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            save_checkpoint(
                checkpoint_path, model=model, optimizer=optimizer, epoch=epoch,
                best_val_loss=best_val, loader_generator=train_generator,
                config={"hidden_units": HIDDEN_UNITS, "lr": LEARNING_RATE,
                        "batch_size": BATCH_SIZE, "seed": seed},
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return {"best_epoch": best_epoch, "best_val_loss": best_val,
            "train_seconds": elapsed, "train_examples_per_second":
            len(train_ds) * epochs / max(elapsed, 1e-12)}


def gradient_check():
    """Use torch.autograd.gradcheck on a tiny double-precision function."""
    x = torch.tensor([0.2, -0.4], dtype=torch.double, requires_grad=True)
    fn = lambda t: (torch.tanh(t) ** 2).sum()
    return bool(torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-5, rtol=1e-3))


def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_ds, val_ds, test_ds = make_datasets()
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    print(f"Samples: train={len(train_ds)}, validation={len(val_ds)}, test={len(test_ds)}")
    print(f"Gradient check: {'PASS' if gradient_check() else 'FAIL'}")

    with tempfile.TemporaryDirectory(prefix="ai-masterclass-017-") as tmp:
        tmpdir = Path(tmp)
        results = {}
        for name, model_cls, offset in [
            ("linear", LinearBaseline, 10),
            ("mlp", MLP, 20),
        ]:
            # Re-seed before each model so initialization is reproducible.
            seed_everything(SEED + offset)
            model = model_cls()
            ckpt = tmpdir / f"{name}-checkpoint.pt"
            train_info = train_model(
                model, train_ds, val_ds, device=device, epochs=EPOCHS,
                checkpoint_path=ckpt, seed=SEED + offset,
            )

            # Restore the best validation checkpoint before the one-time test evaluation.
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                          weight_decay=1e-4)
            _, train_generator = make_loader(train_ds, shuffle=True,
                                               seed=SEED + offset, device=device)
            checkpoint = load_checkpoint(
                ckpt, model=model, optimizer=optimizer,
                loader_generator=train_generator, device=device,
            )
            test_loader, _ = make_loader(test_ds, shuffle=False,
                                          seed=SEED + 100 + offset, device=device)
            test_metrics = evaluate(model, test_loader, device)
            results[name] = {**train_info, **test_metrics,
                             "checkpoint_epoch": int(checkpoint["epoch"]),
                             "checkpoint_bytes": ckpt.stat().st_size}

        if device.type == "cuda":
            results["gpu_peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        print("\nHeld-out test results (candidate selected on validation only):")
        for name in ("linear", "mlp"):
            r = results[name]
            print(f"{name:>6}: acc={r['accuracy']:.4f} f1={r['f1']:.4f} "
                  f"precision={r['precision']:.4f} recall={r['recall']:.4f} "
                  f"test_loss={r['loss']:.5f} best_epoch={r['checkpoint_epoch']}")
            print(f"       train_time={r['train_seconds']:.3f}s "
                  f"train_examples/s={r['train_examples_per_second']:.1f} "
                  f"checkpoint={r['checkpoint_bytes']} bytes")

        # Acceptance checks: nonlinear MLP must beat linear baseline substantially.
        assert gradient_check(), "Autograd numerical gradient check failed"
        assert results["mlp"]["accuracy"] >= 0.95, results["mlp"]
        assert results["mlp"]["accuracy"] > results["linear"]["accuracy"] + 0.20
        assert results["mlp"]["checkpoint_epoch"] >= 1
        assert all(np.isfinite(results[k]["loss"]) for k in ("linear", "mlp"))
        print("\nAll acceptance checks: PASS")
        print("Checkpoint round-trip: PASS (model + optimizer + epoch + RNG state)")


if __name__ == "__main__":
    main()
