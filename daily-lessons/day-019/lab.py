"""Session 019 lab: RNN, LSTM, BPTT and sequence memory.

Run:
    python lab.py
Optional:
    python lab.py --device cpu
    python lab.py --device cuda
"""
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 19) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def make_delayed_marker(
    count: int,
    sequence_length: int = 50,
    seed: int = 19,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Label is encoded at t=0; later steps contain distractor noise."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 2, (count,), generator=generator)
    features = torch.randn(
        count, sequence_length, 2, generator=generator
    ) * 0.5
    features[:, :, 0] = 0.0
    features[:, 0, 0] = labels.float() * 2.0 - 1.0
    return features.float(), labels.long()


class SequenceClassifier(nn.Module):
    def __init__(self, cell: str, input_size: int = 2, hidden_size: int = 32):
        super().__init__()
        if cell == "rnn":
            self.recurrent = nn.RNN(
                input_size, hidden_size, nonlinearity="tanh", batch_first=True
            )
        elif cell == "lstm":
            self.recurrent = nn.LSTM(
                input_size, hidden_size, batch_first=True
            )
            # Demonstration choice: initialize the LSTM forget-gate bias positively.
            # PyTorch's gate order is input, forget, cell candidate, output.
            with torch.no_grad():
                for name, parameter in self.recurrent.named_parameters():
                    if "bias_ih" in name:
                        parameter[hidden_size:2 * hidden_size].fill_(1.0)
                    elif "bias_hh" in name:
                        parameter[hidden_size:2 * hidden_size].zero_()
        else:
            raise ValueError(f"Unsupported recurrent cell: {cell}")
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        outputs, _state = self.recurrent(sequence)
        return self.classifier(outputs[:, -1, :])


@dataclass
class Metrics:
    validation_accuracy: float
    test_accuracy: float
    test_loss: float


def train_model(
    cell: str,
    train_data: tuple[torch.Tensor, torch.Tensor],
    validation_data: tuple[torch.Tensor, torch.Tensor],
    test_data: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    epochs: int = 20,
) -> tuple[SequenceClassifier, Metrics]:
    model = SequenceClassifier(cell).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()
    train_x, train_y = train_data
    loader_generator = torch.Generator().manual_seed(1901)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=128,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    for _epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    model.eval()
    val_x, val_y = validation_data
    test_x, test_y = test_data
    with torch.inference_mode():
        val_logits = model(val_x.to(device))
        test_logits = model(test_x.to(device))
        val_accuracy = (val_logits.argmax(1).cpu() == val_y).float().mean().item()
        test_accuracy = (test_logits.argmax(1).cpu() == test_y).float().mean().item()
        test_loss = criterion(test_logits, test_y.to(device)).item()

    return model, Metrics(val_accuracy, test_accuracy, test_loss)


def initial_state_gradient_norm(cell: str, sequence_length: int) -> float:
    """Illustrate long-path gradient attenuation at random initialization.

    Compares the vanilla RNN h0 path with the LSTM c0 memory path. These
    are different state variables, so values are illustrative, not a
    controlled benchmark of model quality.
    """
    torch.manual_seed(123)
    if cell == "rnn":
        recurrent = nn.RNN(2, 32, nonlinearity="tanh", batch_first=True)
        initial_h = torch.zeros(1, 1, 32, requires_grad=True)
        sequence = torch.randn(1, sequence_length, 2)
        outputs, _ = recurrent(sequence, initial_h)
        outputs[:, -1, :].sum().backward()
        return float(initial_h.grad.norm().item())

    recurrent = nn.LSTM(2, 32, batch_first=True)
    with torch.no_grad():
        for name, parameter in recurrent.named_parameters():
            if "bias_ih" in name:
                parameter[32:64].fill_(1.0)
            elif "bias_hh" in name:
                parameter[32:64].zero_()
    initial_h = torch.zeros(1, 1, 32, requires_grad=True)
    initial_c = torch.zeros(1, 1, 32, requires_grad=True)
    sequence = torch.randn(1, sequence_length, 2)
    outputs, _ = recurrent(sequence, (initial_h, initial_c))
    outputs[:, -1, :].sum().backward()
    return float(initial_c.grad.norm().item())


def verify_shapes() -> None:
    batch, steps, features, hidden = 8, 50, 2, 32
    x = torch.randn(batch, steps, features)

    rnn = nn.RNN(features, hidden, batch_first=True)
    rnn_out, rnn_h = rnn(x)
    assert rnn_out.shape == (batch, steps, hidden)
    assert rnn_h.shape == (1, batch, hidden)

    lstm = nn.LSTM(features, hidden, batch_first=True)
    lstm_out, (lstm_h, lstm_c) = lstm(x)
    assert lstm_out.shape == (batch, steps, hidden)
    assert lstm_h.shape == (1, batch, hidden)
    assert lstm_c.shape == (1, batch, hidden)

    gru = nn.GRU(features, hidden, batch_first=True)
    gru_out, gru_h = gru(x)
    assert gru_out.shape == (batch, steps, hidden)
    assert gru_h.shape == (1, batch, hidden)

    bidirectional = nn.LSTM(
        features, hidden, batch_first=True, bidirectional=True
    )
    bi_out, (bi_h, bi_c) = bidirectional(x)
    assert bi_out.shape == (batch, steps, hidden * 2)
    assert bi_h.shape == (2, batch, hidden)
    assert bi_c.shape == (2, batch, hidden)

    lengths = torch.tensor([50, 43, 37, 29, 21, 18, 12, 7])
    packed = nn.utils.rnn.pack_padded_sequence(
        x, lengths, batch_first=True, enforce_sorted=True
    )
    packed_out, _ = lstm(packed)
    padded_out, output_lengths = nn.utils.rnn.pad_packed_sequence(
        packed_out, batch_first=True
    )
    assert padded_out.shape == (batch, steps, hidden)
    assert torch.equal(output_lengths, lengths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (
            args.device == "auto" and torch.cuda.is_available()
        ) else "cpu"
    )

    set_seed(19)
    verify_shapes()

    train_data = make_delayed_marker(3000, 50, seed=190)
    validation_data = make_delayed_marker(600, 50, seed=191)
    test_data = make_delayed_marker(600, 50, seed=192)

    results: dict[str, Metrics] = {}
    for cell in ("rnn", "lstm"):
        torch.manual_seed(1900 + {"rnn": 1, "lstm": 2}[cell])
        _model, metrics = train_model(
            cell, train_data, validation_data, test_data, device
        )
        results[cell] = metrics

    print(f"Device: {device}")
    print("Delayed-marker task: sequence length=50, input features=2")
    print("Train/validation/test: 3000/600/600")
    print("\nHeld-out results")
    print("model  validation_acc  test_acc  test_loss")
    for cell, metric in results.items():
        print(
            f"{cell:5s}  {metric.validation_accuracy:14.4f}  "
            f"{metric.test_accuracy:8.4f}  {metric.test_loss:9.6f}"
        )

    print("\nInitial-state gradient-path illustration")
    print("steps  vanilla_RNN_dL_dh0  LSTM_dL_dc0")
    for steps in (10, 30, 60, 100):
        rnn_norm = initial_state_gradient_norm("rnn", steps)
        lstm_norm = initial_state_gradient_norm("lstm", steps)
        print(f"{steps:5d}  {rnn_norm:18.6e}  {lstm_norm:13.6e}")

    assert all(np.isfinite(m.test_loss) for m in results.values())
    assert all(0.0 <= m.test_accuracy <= 1.0 for m in results.values())
    assert max(m.test_accuracy for m in results.values()) >= 0.85
    assert all(
        np.isfinite(initial_state_gradient_norm("rnn", n))
        and np.isfinite(initial_state_gradient_norm("lstm", n))
        for n in (10, 30, 60)
    )
    print(
        "\nPASS: tensor shapes, packed sequences, finite gradients/losses, "
        "and held-out learning checks."
    )


if __name__ == "__main__":
    main()
