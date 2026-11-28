#!/usr/bin/env python3
"""Session 018: a reproducible PyTorch CNN for synthetic shape images.

The dataset is intentionally synthetic. It teaches tensor layout, convolution,
pooling, parameter counting, training and held-out evaluation; it is not a
benchmark for real images, audio or silicon diagnostics.
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 18018
IMAGE_SIZE = 16
NUM_CLASSES = 3
BATCH_SIZE = 64
EPOCHS = 12


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_shape_dataset(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create balanced horizontal, vertical and diagonal 16x16 grayscale shapes."""
    rng = np.random.default_rng(seed)
    labels = np.arange(n, dtype=np.int64) % NUM_CLASSES
    rng.shuffle(labels)
    images = np.zeros((n, 1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    for index, label in enumerate(labels):
        image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        if label == 0:
            y0 = int(rng.integers(2, 13))
            thickness = int(rng.integers(2, 4))
            x0 = int(rng.integers(0, 4))
            x1 = int(rng.integers(12, 16))
            image[y0:y0 + thickness, x0:x1] = rng.uniform(0.75, 1.0)
        elif label == 1:
            x0 = int(rng.integers(2, 13))
            thickness = int(rng.integers(2, 4))
            y0 = int(rng.integers(0, 4))
            y1 = int(rng.integers(12, 16))
            image[y0:y1, x0:x0 + thickness] = rng.uniform(0.75, 1.0)
        else:
            offset = int(rng.integers(-3, 4))
            thickness = int(rng.integers(1, 3))
            intensity = rng.uniform(0.75, 1.0)
            for y in range(IMAGE_SIZE):
                x = y + offset
                for delta in range(thickness):
                    if 0 <= x + delta < IMAGE_SIZE:
                        image[y, x + delta] = intensity

        image += rng.normal(0.0, 0.10, size=image.shape).astype(np.float32)
        images[index, 0] = np.clip(image, 0.0, 1.0)

    return torch.from_numpy(images), torch.from_numpy(labels)


class SmallCNN(nn.Module):
    """NCHW input [batch, 1, 16, 16]; output is three class logits."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_predictions: list[int] = []
    all_targets: list[int] = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            total += labels.size(0)
            all_predictions.extend(logits.argmax(dim=1).cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    predictions = np.asarray(all_predictions)
    targets = np.asarray(all_targets)
    accuracy = float(np.mean(predictions == targets))
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for actual, predicted in zip(targets, predictions):
        matrix[int(actual), int(predicted)] += 1
    return total_loss / total, accuracy, matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Use auto to select CUDA when available, otherwise CPU.",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (
            args.device == "auto" and torch.cuda.is_available()
        ) else "cpu"
    )

    seed_everything()
    torch.set_num_threads(1)

    x_train, y_train = make_shape_dataset(1200, 101)
    x_validation, y_validation = make_shape_dataset(300, 202)
    x_test, y_test = make_shape_dataset(300, 303)

    train_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    validation_loader = DataLoader(
        TensorDataset(x_validation, y_validation),
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.003, weight_decay=1e-4
    )

    with torch.inference_mode():
        sample = torch.zeros(4, 1, IMAGE_SIZE, IMAGE_SIZE, device=device)
        feature_shape = tuple(model.features(sample).shape)
        output_shape = tuple(model(sample).shape)

    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    print(f"Train/validation/test: {len(x_train)}/{len(x_validation)}/{len(x_test)}")
    print(f"Feature tensor shape: {feature_shape}")
    print(f"Logit tensor shape: {output_shape}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    assert feature_shape == (4, 16, 4, 4)
    assert output_shape == (4, NUM_CLASSES)

    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=(device.type == "cuda"))
            labels = labels.to(device, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += float(loss.item()) * labels.size(0)
            total += labels.size(0)

        validation_loss, validation_accuracy, _ = evaluate(
            model, validation_loader, criterion, device
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

        print(
            f"epoch={epoch:02d} "
            f"train_loss={total_loss / total:.5f} "
            f"val_loss={validation_loss:.5f} "
            f"val_accuracy={validation_accuracy:.4f}"
        )

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")

    model.load_state_dict(best_state)
    test_loss, test_accuracy, confusion = evaluate(
        model, test_loader, criterion, device
    )
    elapsed = time.perf_counter() - start

    print(f"Best validation loss: {best_validation_loss:.6f}")
    print(f"Test loss: {test_loss:.6f}")
    print(f"Test accuracy: {test_accuracy:.4%}")
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion)
    print(f"Training wall time (environment-specific): {elapsed:.3f}s")

    assert np.isfinite(test_loss)
    assert np.isfinite(test_accuracy)
    assert confusion.sum() == len(x_test)
    assert np.all(confusion.sum(axis=1) > 0)
    assert test_accuracy >= 0.95, (
        f"Expected >=95% on this synthetic task, got {test_accuracy:.3%}"
    )
    print("ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
