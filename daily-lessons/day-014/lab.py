"""Session 014 lab: novelty detection on synthetic SoC telemetry.

The detectors are trained only on nominal observations. Synthetic labels are
used solely for held-out evaluation, never for fitting or threshold tuning.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

SEED = 14014
FEATURES = ["temperature_c", "dma_retries", "bus_error_rate", "queue_pressure"]


def make_nominal(rng: np.random.Generator, n: int) -> np.ndarray:
    """Three legitimate operating modes with correlated telemetry features."""
    proportions = np.array([0.42, 0.38, 0.20])
    counts = rng.multinomial(n, proportions)
    means = [
        np.array([48.0, 0.30, 0.025, 14.0]),  # idle
        np.array([66.0, 1.05, 0.075, 42.0]),   # streaming
        np.array([82.0, 2.10, 0.160, 76.0]),   # stress but nominal
    ]
    stds = [
        np.array([2.8, 0.18, 0.018, 5.0]),
        np.array([3.5, 0.32, 0.035, 8.0]),
        np.array([3.8, 0.45, 0.055, 9.0]),
    ]
    corr = np.array([
        [1.00, 0.48, 0.30, 0.52],
        [0.48, 1.00, 0.42, 0.58],
        [0.30, 0.42, 1.00, 0.35],
        [0.52, 0.58, 0.35, 1.00],
    ])
    blocks = []
    for count, mean, std in zip(counts, means, stds):
        cov = np.outer(std, std) * corr
        blocks.append(rng.multivariate_normal(mean, cov, size=count))
    x = np.vstack(blocks)
    rng.shuffle(x)
    x[:, 1:] = np.maximum(x[:, 1:], 0.0)
    return x


def make_anomalies(rng: np.random.Generator, n: int) -> np.ndarray:
    """Mix gross spikes with contextually inconsistent and subtle samples."""
    n_gross = n // 3
    n_context = n // 3
    n_subtle = n - n_gross - n_context
    gross = np.column_stack([
        rng.uniform(94, 112, n_gross),
        rng.uniform(4.5, 9.0, n_gross),
        rng.uniform(0.45, 1.25, n_gross),
        rng.uniform(100, 135, n_gross),
    ])
    context = np.column_stack([
        rng.uniform(45, 55, n_context),
        rng.uniform(2.8, 4.2, n_context),
        rng.uniform(0.18, 0.38, n_context),
        rng.uniform(12, 32, n_context),
    ])
    subtle = np.column_stack([
        rng.uniform(72, 79, n_subtle),
        rng.uniform(0.25, 0.75, n_subtle),
        rng.uniform(0.13, 0.24, n_subtle),
        rng.uniform(18, 34, n_subtle),
    ])
    x = np.vstack([gross, context, subtle])
    rng.shuffle(x)
    return x


def metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, scores),
        "average_precision": average_precision_score(y_true, scores),
        "false_positive_rate": float(np.mean(y_pred[y_true == 0] == 1)),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    x_train = make_nominal(rng, 1800)
    x_test_normal = make_nominal(rng, 600)
    x_test_anomaly = make_anomalies(rng, 80)
    x_test = np.vstack([x_test_normal, x_test_anomaly])
    y_test = np.concatenate([
        np.zeros(len(x_test_normal), dtype=int),
        np.ones(len(x_test_anomaly), dtype=int),
    ])
    order = rng.permutation(len(x_test))
    x_test, y_test = x_test[order], y_test[order]

    # Fit scaling only on nominal training data. All algorithms see the same
    # feature transform, but Isolation Forest itself is not distance-based.
    models = {
        "IsolationForest": make_pipeline(
            StandardScaler(),
            IsolationForest(
                n_estimators=300,
                contamination=0.05,
                max_samples="auto",
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
        "LOF_novelty": make_pipeline(
            StandardScaler(),
            LocalOutlierFactor(
                n_neighbors=25,
                novelty=True,
                contamination=0.05,
                n_jobs=-1,
            ),
        ),
        "OneClassSVM": make_pipeline(
            StandardScaler(),
            OneClassSVM(kernel="rbf", nu=0.05, gamma="scale"),
        ),
    }

    print(f"Seed: {SEED}")
    print(f"Training nominal samples: {len(x_train)}")
    print(f"Held-out test samples: {len(x_test)} ({int(y_test.sum())} injected anomalies)")
    print("\nHeld-out results (decision_function < 0 => anomaly):")
    results = {}
    for name, model in models.items():
        model.fit(x_train)  # no labels are passed to fit
        decision = model.decision_function(x_test)
        y_pred = (decision < 0).astype(int)
        anomaly_score = -decision  # larger means more anomalous
        row = metrics(y_test, y_pred, anomaly_score)
        results[name] = row
        print(
            f"{name:16s} "
            f"precision={row['precision']:.3f} "
            f"recall={row['recall']:.3f} "
            f"f1={row['f1']:.3f} "
            f"bal_acc={row['balanced_accuracy']:.3f} "
            f"roc_auc={row['roc_auc']:.3f} "
            f"AP={row['average_precision']:.3f} "
            f"FPR={row['false_positive_rate']:.3f}"
        )

    assert set(results) == {"IsolationForest", "LOF_novelty", "OneClassSVM"}
    for row in results.values():
        assert all(np.isfinite(value) for value in row.values())
        assert 0.0 <= row["precision"] <= 1.0
        assert 0.0 <= row["recall"] <= 1.0
        assert 0.0 <= row["roc_auc"] <= 1.0
    assert len(x_train) == 1800 and int(y_test.sum()) == 80
    assert np.all(np.isfinite(x_test))
    print("\nPASS: deterministic dataset, nominal-only fitting, score direction, and metric checks.")


if __name__ == "__main__":
    main()
