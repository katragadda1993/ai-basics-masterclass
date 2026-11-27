"""Day 011: KNN, Gaussian Naive Bayes and SVM comparison.

Reproducible synthetic SoC telemetry classification exercise.
All model selection is performed on validation data; the test set is
used once for the selected candidate and the frozen baseline.
"""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SEED = 20261128


def make_telemetry(n_samples: int = 3000):
    rng = np.random.default_rng(SEED)
    temperature = rng.uniform(35, 105, n_samples)
    dma_retries = rng.poisson(0.7, n_samples)
    bus_errors = rng.poisson(0.45, n_samples)
    queue_depth = rng.integers(0, 129, n_samples)
    link_flaps = rng.poisson(0.25, n_samples)
    noise_sensor = rng.normal(0, 1, n_samples)

    # A deliberately nonlinear synthetic fault-generating process.
    latent = (
        0.045 * (temperature - 70)
        + 0.75 * dma_retries
        + 0.85 * bus_errors
        + 0.018 * queue_depth
        + 1.35 * link_flaps
        + 0.035 * (temperature - 70) * link_flaps
        - 3.8
        + rng.normal(0, 1.1, n_samples)
    )
    X = np.column_stack(
        [temperature, dma_retries, bus_errors, queue_depth, link_flaps, noise_sensor]
    )
    y = (latent > 0).astype(int)
    return X, y


def metrics(y_true, y_pred, score):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, score),
    }


def main():
    X, y = make_telemetry()
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=17, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=18, stratify=y_temp
    )

    print("Dataset sizes:", len(y_train), "train,", len(y_val), "validation,", len(y_test), "test")
    print(f"Positive class prevalence: {y.mean():.3%}")

    # Majority baseline is fixed before model comparison.
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    baseline_val_pred = baseline.predict(X_val)
    baseline_val_bal_acc = balanced_accuracy_score(y_val, baseline_val_pred)

    candidates = []

    # KNN: scaling is essential because telemetry fields use different units.
    for k in (5, 11, 21, 35):
        for weights in ("uniform", "distance"):
            model = make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(n_neighbors=k, weights=weights),
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            candidates.append(
                ("KNN", f"k={k}, weights={weights}", balanced_accuracy_score(y_val, pred), model)
            )

    # GaussianNB is a fast probabilistic baseline with a conditional
    # independence and per-feature Gaussian likelihood assumption.
    for smoothing in (1e-9, 1e-6, 1e-3):
        model = GaussianNB(var_smoothing=smoothing).fit(X_train, y_train)
        pred = model.predict(X_val)
        candidates.append(
            ("GaussianNB", f"var_smoothing={smoothing:g}",
             balanced_accuracy_score(y_val, pred), model)
        )

    # SVC: scale inputs; compare linear and RBF kernels.
    for kernel in ("linear", "rbf"):
        for C in (0.5, 1.0, 3.0, 10.0):
            gammas = ("scale",) if kernel == "linear" else ("scale", 0.03, 0.1)
            for gamma in gammas:
                model = make_pipeline(
                    StandardScaler(),
                    SVC(
                        kernel=kernel,
                        C=C,
                        gamma=gamma,
                        class_weight="balanced",
                    ),
                )
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                candidates.append(
                    ("SVC", f"kernel={kernel}, C={C}, gamma={gamma}",
                     balanced_accuracy_score(y_val, pred), model)
                )

    # Select the best hyperparameter setting within each model family
    # using validation balanced accuracy only.
    best_by_family = {}
    for family, config, score, model in candidates:
        if family not in best_by_family or score > best_by_family[family][1]:
            best_by_family[family] = (config, score, model)

    print("\nValidation comparison (best setting per family):")
    print(f"Majority baseline balanced accuracy: {baseline_val_bal_acc:.4f}")
    for family, (config, score, _) in best_by_family.items():
        print(f"{family:10s} {score:.4f} | {config}")

    selected_family = max(
        best_by_family, key=lambda family: best_by_family[family][1]
    )
    selected_config, selected_val_score, selected_model = best_by_family[selected_family]

    # The test set is evaluated only after model family and hyperparameters
    # have been selected and frozen on the validation set.
    test_pred = selected_model.predict(X_test)
    test_score = selected_model.decision_function(X_test) if selected_family == "SVC" else (
        selected_model.predict_proba(X_test)[:, 1]
    )
    selected_test_metrics = metrics(y_test, test_pred, test_score)

    baseline_test_pred = baseline.predict(X_test)
    baseline_test_metrics = metrics(
        y_test, baseline_test_pred, baseline_test_pred.astype(float)
    )

    print(f"\nSelected model: {selected_family} ({selected_config})")
    print(f"Validation balanced accuracy: {selected_val_score:.4f}")
    print("Frozen selected model — held-out test metrics:")
    for key, value in selected_test_metrics.items():
        print(f"  {key:18s}: {value:.4f}")
    print("Majority baseline — held-out test metrics:")
    for key, value in baseline_test_metrics.items():
        print(f"  {key:18s}: {value:.4f}")

    # Acceptance checks: shapes, finite scores, meaningful baseline,
    # validation-selected model beats a trivial baseline on this fixed data.
    assert X.shape == (3000, 6)
    assert len(np.unique(y)) == 2
    assert all(np.isfinite(list(selected_test_metrics.values())))
    assert selected_val_score > baseline_val_bal_acc
    assert selected_test_metrics["roc_auc"] > 0.80
    assert selected_test_metrics["balanced_accuracy"] > 0.70
    print("\nPASS: all Day 011 lab assertions passed.")


if __name__ == "__main__":
    main()
