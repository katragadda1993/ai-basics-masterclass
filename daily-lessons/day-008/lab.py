#!/usr/bin/env python3
"""Day 008: leakage-aware supervised ML baseline on synthetic SoC telemetry.

This is a teaching experiment, not a validated hardware-fault detector.
Run: python lab.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_COLUMNS = [
    "temperature_c",
    "dma_retries",
    "bus_errors",
    "link_flaps",
]
CATEGORICAL_COLUMNS = ["ip_family"]


def make_synthetic_telemetry(seed: int = 42, n_samples: int = 1200):
    """Generate deterministic teaching data with a moderately rare positive class."""
    rng = np.random.default_rng(seed)

    temperature = rng.normal(65.0, 9.0, n_samples)
    retries = rng.poisson(1.1, n_samples)
    bus_errors = rng.poisson(0.65, n_samples)
    link_flaps = rng.poisson(0.45, n_samples)
    ip_family = rng.choice(
        ["DMA_A", "DMA_B", "DMA_C"],
        size=n_samples,
        p=[0.50, 0.30, 0.20],
    )
    family_effect = np.select(
        [ip_family == "DMA_A", ip_family == "DMA_B", ip_family == "DMA_C"],
        [0.0, 0.25, 0.55],
        default=0.0,
    )
    noise = rng.normal(0.0, 1.1, n_samples)

    latent_score = (
        0.11 * (temperature - 65.0)
        + 0.75 * retries
        + 1.05 * bus_errors
        + 0.90 * link_flaps
        + family_effect
        + noise
        - 2.8
    )
    labels = (latent_score > 0.0).astype(int)

    features = pd.DataFrame(
        {
            "temperature_c": temperature,
            "dma_retries": retries,
            "bus_errors": bus_errors,
            "link_flaps": link_flaps,
            "ip_family": ip_family,
        }
    )

    # Add a small amount of reproducible missingness.
    missing_numeric = rng.choice(n_samples, size=18, replace=False)
    features.loc[missing_numeric, "temperature_c"] = np.nan
    missing_categorical = rng.choice(n_samples, size=8, replace=False)
    features.loc[missing_categorical, "ip_family"] = np.nan

    return features, labels


def make_pipeline() -> Pipeline:
    """Fit preprocessing only on the data passed to Pipeline.fit."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    )


def measure(model, features: pd.DataFrame, labels: np.ndarray) -> dict:
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
        "average_precision": average_precision_score(labels, probabilities),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ),
    }


def print_metrics(title: str, values: dict) -> None:
    print(f"\n{title}")
    for key in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ):
        print(f"  {key:20s}: {values[key]:.4f}")
    print("  confusion_matrix:")
    print(values["confusion_matrix"])


def main() -> None:
    features, labels = make_synthetic_telemetry()
    row_ids = np.arange(len(labels))

    # 70% train, 15% validation, 15% test. Stratify each split.
    train_ids, remainder_ids = train_test_split(
        row_ids,
        test_size=0.30,
        random_state=42,
        stratify=labels,
    )
    validation_ids, test_ids = train_test_split(
        remainder_ids,
        test_size=0.50,
        random_state=43,
        stratify=labels[remainder_ids],
    )

    # Prove that no row appears in more than one partition.
    assert set(train_ids).isdisjoint(validation_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(validation_ids).isdisjoint(test_ids)
    assert len(train_ids) + len(validation_ids) + len(test_ids) == len(labels)

    print("Day 008 — ML Foundations Lab")
    print(f"Rows: {len(labels)}")
    print(f"Positive-label prevalence: {labels.mean():.3f}")
    print(
        "Split sizes: "
        f"train={len(train_ids)}, "
        f"validation={len(validation_ids)}, "
        f"test={len(test_ids)}"
    )
    print(f"Missing values: {features.isna().sum().to_dict()}")

    X_train = features.iloc[train_ids]
    y_train = labels[train_ids]
    X_validation = features.iloc[validation_ids]
    y_validation = labels[validation_ids]

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_validation = measure(baseline, X_validation, y_validation)
    print_metrics("Majority baseline — validation", baseline_validation)

    candidate = make_pipeline()
    candidate.fit(X_train, y_train)
    candidate_validation = measure(candidate, X_validation, y_validation)
    print_metrics("Logistic pipeline — validation", candidate_validation)

    # Model selection is based on validation only.
    assert (
        candidate_validation["balanced_accuracy"]
        > baseline_validation["balanced_accuracy"] + 0.10
    )
    assert candidate_validation["f1"] > 0.60

    # Freeze the candidate, then refit on train + validation.
    development_ids = np.concatenate([train_ids, validation_ids])
    X_development = features.iloc[development_ids]
    y_development = labels[development_ids]

    final_model = make_pipeline()
    final_model.fit(X_development, y_development)

    # Verify the imputer learned only from the final development partition.
    learned_medians = (
        final_model.named_steps["preprocess"]
        .named_transformers_["numeric"]
        .named_steps["imputer"]
        .statistics_
    )
    expected_medians = X_development[NUMERIC_COLUMNS].median().to_numpy()
    np.testing.assert_allclose(learned_medians, expected_medians)

    # Evaluate the untouched test set once after model/configuration freeze.
    X_test = features.iloc[test_ids]
    y_test = labels[test_ids]
    final_test = measure(final_model, X_test, y_test)
    print_metrics("Frozen logistic pipeline — final test", final_test)

    assert final_test["balanced_accuracy"] > 0.70
    assert final_test["roc_auc"] > 0.80
    assert final_test["f1"] > 0.60

    # Smoke-test an unseen categorical level. It must not crash.
    unseen_family = pd.DataFrame(
        [
            {
                "temperature_c": 70.0,
                "dma_retries": 1,
                "bus_errors": 0,
                "link_flaps": 0,
                "ip_family": "DMA_NEW",
            }
        ]
    )
    unseen_probability = final_model.predict_proba(unseen_family)[0, 1]
    assert 0.0 <= unseen_probability <= 1.0

    print(
        "\nUnseen-category smoke test: "
        f"predicted positive-class score={unseen_probability:.4f}"
    )
    print(
        "All assertions passed. Synthetic results are for education only; "
        "they are not real hardware validation."
    )


if __name__ == "__main__":
    main()
