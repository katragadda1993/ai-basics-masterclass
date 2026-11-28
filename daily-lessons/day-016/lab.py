#!/usr/bin/env python3
"""Session 016: train a tiny MLP on a synthetic XOR classification problem.

Requires only NumPy. The example is educational; it is not a hardware diagnostic model.
"""
from __future__ import annotations

import numpy as np

SEED_DATA = 20261128
SEED_SPLIT = 12345
SEED_INIT = 7
SEED_BATCH = 99


def make_xor_data(samples_per_quadrant: int = 150):
    rng = np.random.default_rng(SEED_DATA)
    centers = [
        ((-1.0, -1.0), 0),
        ((1.0, 1.0), 0),
        ((-1.0, 1.0), 1),
        ((1.0, -1.0), 1),
    ]
    xs, ys = [], []
    for center, label in centers:
        xs.append(rng.normal(loc=center, scale=0.38, size=(samples_per_quadrant, 2)))
        ys.append(np.full(samples_per_quadrant, label, dtype=np.float64))
    X = np.vstack(xs)
    y = np.concatenate(ys).reshape(-1, 1)

    rng_split = np.random.default_rng(SEED_SPLIT)
    class_indices = [np.flatnonzero(y[:, 0] == label) for label in (0, 1)]
    partitions = []
    for indices in class_indices:
        indices = indices.copy()
        rng_split.shuffle(indices)
        partitions.append((indices[:180], indices[180:240], indices[240:]))

    train_idx = np.concatenate([partitions[0][0], partitions[1][0]])
    val_idx = np.concatenate([partitions[0][1], partitions[1][1]])
    test_idx = np.concatenate([partitions[0][2], partitions[1][2]])
    rng_split.shuffle(train_idx)
    rng_split.shuffle(val_idx)
    rng_split.shuffle(test_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Fit preprocessing on training data only.
    mean = X_train.mean(axis=0, keepdims=True)
    scale = X_train.std(axis=0, keepdims=True)
    scale[scale == 0] = 1.0
    X_train = (X_train - mean) / scale
    X_val = (X_val - mean) / scale
    X_test = (X_test - mean) / scale

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), mean, scale


def sigmoid(z):
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def initialize(seed: int = SEED_INIT, hidden_units: int = 8):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0.0, np.sqrt(1.0 / 2), size=(2, hidden_units)),
        "b1": np.zeros((1, hidden_units)),
        "W2": rng.normal(0.0, np.sqrt(1.0 / hidden_units), size=(hidden_units, 1)),
        "b2": np.zeros((1, 1)),
    }


def forward(X, params):
    z1 = X @ params["W1"] + params["b1"]
    a1 = np.tanh(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    probabilities = sigmoid(z2)
    return probabilities, (a1, probabilities)


def binary_cross_entropy(y, probabilities):
    eps = 1e-12
    p = np.clip(probabilities, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def backward(X, y, params):
    """Return gradients for a sigmoid/BCE output and tanh hidden layer."""
    n = X.shape[0]
    probabilities, (a1, _) = forward(X, params)

    # For sigmoid output + binary cross-entropy, dL/dz2 = (p - y) / n.
    dz2 = (probabilities - y) / n
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    da1 = dz2 @ params["W2"].T
    dz1 = da1 * (1.0 - a1 * a1)  # derivative of tanh
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def finite_difference_check(X, y, params, epsilon: float = 1e-5):
    analytic = backward(X, y, params)
    maximum_relative_error = 0.0
    for name in params:
        index = (0, 0)
        original = params[name][index]
        params[name][index] = original + epsilon
        loss_plus = binary_cross_entropy(y, forward(X, params)[0])
        params[name][index] = original - epsilon
        loss_minus = binary_cross_entropy(y, forward(X, params)[0])
        params[name][index] = original

        numerical = (loss_plus - loss_minus) / (2.0 * epsilon)
        expected = analytic[name][index]
        relative = abs(numerical - expected) / max(
            1e-8, abs(numerical) + abs(expected)
        )
        maximum_relative_error = max(maximum_relative_error, float(relative))
    return maximum_relative_error


def train_mlp(train, validation, learning_rate=0.08, epochs=1500,
              batch_size=32, patience=120):
    X_train, y_train = train
    X_val, y_val = validation
    params = initialize()
    rng = np.random.default_rng(SEED_BATCH)

    best_params = None
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    epochs_run = 0

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(X_train))
        for start in range(0, len(order), batch_size):
            batch_indices = order[start:start + batch_size]
            gradients = backward(
                X_train[batch_indices], y_train[batch_indices], params
            )
            for name in params:
                params[name] -= learning_rate * gradients[name]

        val_probabilities = forward(X_val, params)[0]
        val_loss = binary_cross_entropy(y_val, val_probabilities)
        epochs_run = epoch

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_params = {name: value.copy() for name, value in params.items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            break

    return best_params, best_epoch, best_val_loss, epochs_run


def train_linear_baseline(train, learning_rate=0.1, epochs=1000):
    X_train, y_train = train
    weights = np.zeros((X_train.shape[1], 1))
    bias = np.zeros((1, 1))
    for _ in range(epochs):
        probabilities = sigmoid(X_train @ weights + bias)
        dweights = X_train.T @ (probabilities - y_train) / len(X_train)
        dbias = np.mean(probabilities - y_train, axis=0, keepdims=True)
        weights -= learning_rate * dweights
        bias -= learning_rate * dbias
    return weights, bias


def classification_metrics(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)
    truth = y_true.astype(int)
    tp = int(np.sum((predictions == 1) & (truth == 1)))
    tn = int(np.sum((predictions == 0) & (truth == 0)))
    fp = int(np.sum((predictions == 1) & (truth == 0)))
    fn = int(np.sum((predictions == 0) & (truth == 1)))
    accuracy = (tp + tn) / len(truth)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def main():
    train, validation, test, mean, scale = make_xor_data()
    X_train, y_train = train
    X_val, _ = validation
    X_test, y_test = test

    params_for_check = initialize()
    gradient_error = finite_difference_check(
        X_train[:8], y_train[:8], params_for_check
    )

    mlp_params, best_epoch, best_val_loss, epochs_run = train_mlp(
        train, validation
    )
    test_probabilities = forward(X_test, mlp_params)[0]
    mlp_metrics = classification_metrics(y_test, test_probabilities)

    linear_weights, linear_bias = train_linear_baseline(train)
    linear_probabilities = sigmoid(X_test @ linear_weights + linear_bias)
    linear_metrics = classification_metrics(y_test, linear_probabilities)

    print("Session 016 — NumPy MLP XOR lab")
    print(f"Data sizes: train={len(X_train)}, validation={len(X_val)}, test={len(X_test)}")
    print(f"Train-only standardization mean={mean.ravel()}, scale={scale.ravel()}")
    print(f"Gradient-check maximum relative error: {gradient_error:.3e}")
    print(
        f"Best epoch: {best_epoch}; epochs executed: {epochs_run}; "
        f"best validation BCE: {best_val_loss:.5f}"
    )
    print("\nHeld-out test metrics")
    print("Model                 Accuracy  Precision  Recall   F1")
    print(
        "Linear logistic      "
        f"{linear_metrics['accuracy']:.3f}     {linear_metrics['precision']:.3f}      "
        f"{linear_metrics['recall']:.3f}   {linear_metrics['f1']:.3f}"
    )
    print(
        "MLP (2-8-1)          "
        f"{mlp_metrics['accuracy']:.3f}     {mlp_metrics['precision']:.3f}      "
        f"{mlp_metrics['recall']:.3f}   {mlp_metrics['f1']:.3f}"
    )
    print(
        f"MLP confusion counts: TP={mlp_metrics['tp']} TN={mlp_metrics['tn']} "
        f"FP={mlp_metrics['fp']} FN={mlp_metrics['fn']}"
    )

    assert X_train.shape == (360, 2)
    assert X_val.shape == (120, 2)
    assert X_test.shape == (120, 2)
    assert gradient_error < 1e-6, "Backpropagation gradient check failed"
    assert best_epoch > 0
    assert mlp_metrics["accuracy"] >= 0.95, "MLP did not learn nonlinear XOR"
    assert linear_metrics["accuracy"] <= 0.65, "Linear baseline unexpectedly solved XOR"
    assert np.isfinite(best_val_loss)
    assert all(np.all(np.isfinite(value)) for value in mlp_params.values())
    print("\nPASS: split, gradient, convergence, generalization and finite-value checks passed.")


if __name__ == "__main__":
    main()
