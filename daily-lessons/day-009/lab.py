#!/usr/bin/env python3
"""Day 009: linear and logistic models on synthetic SoC telemetry.

Educational experiment only; not a validated silicon-fault predictor.
Run: python lab.py
"""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss,
    mean_absolute_error, mean_squared_error, precision_score, r2_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "temperature_c", "queue_depth", "payload_kb",
    "dma_retries", "bus_errors", "link_flaps",
]


def make_synthetic_data(seed: int = 2026, n_samples: int = 2400):
    """Return features, latency regression target and binary fault label."""
    rng = np.random.default_rng(seed)
    temperature = rng.normal(65.0, 8.0, n_samples)
    queue_depth = rng.poisson(3.0, n_samples)
    payload_kb = rng.uniform(32.0, 1024.0, n_samples)
    dma_retries = rng.poisson(0.9, n_samples)
    bus_errors = rng.poisson(0.45, n_samples)
    link_flaps = rng.poisson(0.25, n_samples)

    X = np.column_stack([
        temperature, queue_depth, payload_kb,
        dma_retries, bus_errors, link_flaps,
    ])

    latency_ms = (
        5.0 + 0.12 * (temperature - 65.0) + 0.65 * queue_depth
        + 0.020 * payload_kb + 1.8 * dma_retries
        + 0.9 * bus_errors + 1.2 * link_flaps
        + rng.normal(0.0, 3.5, n_samples)
    )

    logit = (
        -3.7 + 0.035 * (temperature - 65.0)
        + 0.52 * dma_retries + 0.88 * bus_errors
        + 0.75 * link_flaps + 0.10 * queue_depth
        + 0.0008 * payload_kb
    )
    fault_probability = 1.0 / (1.0 + np.exp(-logit))
    fault_label = (rng.random(n_samples) < fault_probability).astype(int)
    return X, latency_ms, fault_label


def regression_experiment(X: np.ndarray, y: np.ndarray) -> dict:
    ids = np.arange(len(y))
    train_ids, test_ids = train_test_split(
        ids, test_size=0.25, random_state=17
    )
    X_train, X_test = X[train_ids], X[test_ids]
    y_train, y_test = y[train_ids], y[test_ids]

    models = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge (alpha=5)": make_pipeline(StandardScaler(), Ridge(alpha=5.0)),
    }
    results, fitted = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        results[name] = {
            "mae": mean_absolute_error(y_test, prediction),
            "rmse": np.sqrt(mean_squared_error(y_test, prediction)),
            "r2": r2_score(y_test, prediction),
        }
        fitted[name] = model

    assert results["Linear regression"]["rmse"] < results["Mean baseline"]["rmse"]
    assert results["Linear regression"]["r2"] > 0.60
    return {"results": results, "models": fitted}


def classification_experiment(X: np.ndarray, y: np.ndarray) -> dict:
    ids = np.arange(len(y))
    train_ids, remainder_ids = train_test_split(
        ids, test_size=0.30, random_state=29, stratify=y
    )
    validation_ids, test_ids = train_test_split(
        remainder_ids, test_size=0.50, random_state=31,
        stratify=y[remainder_ids],
    )
    assert set(train_ids).isdisjoint(validation_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(validation_ids).isdisjoint(test_ids)

    X_train, y_train = X[train_ids], y[train_ids]
    X_validation, y_validation = X[validation_ids], y[validation_ids]
    X_test, y_test = X[test_ids], y[test_ids]

    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, random_state=0),
    ).fit(X_train, y_train)

    validation_probability = classifier.predict_proba(X_validation)[:, 1]
    candidates = [0.50, 0.40, 0.30, 0.25, 0.20, 0.15]
    validation_scores = []
    for threshold in candidates:
        pred = (validation_probability >= threshold).astype(int)
        validation_scores.append(
            (f1_score(y_validation, pred, zero_division=0), threshold)
        )
    # Threshold selection uses validation labels only.
    chosen_threshold = max(validation_scores)[1]

    probability = classifier.predict_proba(X_test)[:, 1]
    baseline_prediction = baseline.predict(X_test)
    default_prediction = (probability >= 0.50).astype(int)
    selected_prediction = (probability >= chosen_threshold).astype(int)

    def metrics(prediction: np.ndarray) -> dict:
        return {
            "accuracy": accuracy_score(y_test, prediction),
            "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
            "precision": precision_score(y_test, prediction, zero_division=0),
            "recall": recall_score(y_test, prediction, zero_division=0),
            "f1": f1_score(y_test, prediction, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, prediction, labels=[0, 1]),
        }

    probability_metrics = {
        "roc_auc": roc_auc_score(y_test, probability),
        "average_precision": average_precision_score(y_test, probability),
        "log_loss": log_loss(y_test, probability),
        "brier_score": brier_score_loss(y_test, probability),
    }
    baseline_metrics = metrics(baseline_prediction)
    default_metrics = metrics(default_prediction)
    selected_metrics = metrics(selected_prediction)

    assert probability_metrics["roc_auc"] > 0.60
    assert selected_metrics["balanced_accuracy"] > baseline_metrics["balanced_accuracy"]
    assert 0.0 < chosen_threshold <= 1.0
    assert np.all(np.isfinite(probability))

    return {
        "prevalence": float(y_test.mean()),
        "threshold": chosen_threshold,
        "baseline": baseline_metrics,
        "default": default_metrics,
        "selected": selected_metrics,
        "probability": probability_metrics,
    }


def main() -> None:
    X, latency_ms, fault_label = make_synthetic_data()
    print("Day 009 — Linear and Logistic Regression Lab")
    print(f"Samples: {len(X)} | Features: {len(FEATURE_NAMES)}")
    print(f"Fault-label prevalence: {fault_label.mean():.3f}")

    regression = regression_experiment(X, latency_ms)
    print("\nREGRESSION — held-out test")
    for name, values in regression["results"].items():
        print(
            f"{name:20s} MAE={values['mae']:.3f} ms  "
            f"RMSE={values['rmse']:.3f} ms  R2={values['r2']:.3f}"
        )

    coefficients = regression["models"]["Linear regression"][-1].coef_
    print("\nStandardized linear coefficients (per 1 SD feature change):")
    for name, coefficient in zip(FEATURE_NAMES, coefficients):
        print(f"  {name:16s}: {coefficient:+.3f} ms")

    classification = classification_experiment(X, fault_label)
    print("\nCLASSIFICATION — held-out test")
    print(f"Positive prevalence: {classification['prevalence']:.3f}")
    print(
        "Majority baseline balanced accuracy: "
        f"{classification['baseline']['balanced_accuracy']:.3f}"
    )
    default = classification["default"]
    print(
        "Logistic threshold=0.50: "
        f"accuracy={default['accuracy']:.3f}, "
        f"precision={default['precision']:.3f}, "
        f"recall={default['recall']:.3f}, F1={default['f1']:.3f}"
    )
    selected = classification["selected"]
    print(f"Validation-selected threshold: {classification['threshold']:.2f}")
    print(
        "Selected threshold test: "
        f"accuracy={selected['accuracy']:.3f}, "
        f"balanced_accuracy={selected['balanced_accuracy']:.3f}, "
        f"precision={selected['precision']:.3f}, "
        f"recall={selected['recall']:.3f}, F1={selected['f1']:.3f}"
    )
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(selected["confusion_matrix"])
    print("Probability metrics:")
    for name, value in classification["probability"].items():
        print(f"  {name:18s}: {value:.4f}")

    print("\nAll assertions passed.")
    print("Synthetic teaching results only; not real silicon validation.")


if __name__ == "__main__":
    main()
