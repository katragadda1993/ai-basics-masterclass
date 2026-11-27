"""Session 010 lab: decision trees, random forests and feature importance.

Run:
    python -m pip install -r requirements.txt
    python lab.py

The dataset is synthetic and for education only.
"""
from __future__ import annotations

import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

SEED = 20261128
FEATURES = [
    "temperature",
    "dma_retries",
    "bus_errors",
    "queue_depth",
    "link_flaps",
    "noise_sensor",
]


def make_dataset(n_samples: int = 3000):
    """Generate noisy nonlinear telemetry labels with a fixed seed."""
    rng = np.random.default_rng(SEED)
    temperature = rng.normal(68, 12, n_samples).clip(30, 110)
    dma_retries = rng.poisson(1.1, n_samples)
    bus_errors = rng.poisson(0.8, n_samples)
    queue_depth = rng.integers(0, 65, n_samples)
    link_flaps = rng.poisson(0.25, n_samples)
    noise_sensor = rng.normal(0, 1, n_samples)

    # Nonlinear teaching rule: interacting telemetry signals matter.
    fault = (
        ((bus_errors >= 2) & (temperature > 72))
        | ((dma_retries >= 3) & (link_flaps >= 1))
        | ((queue_depth >= 52) & (temperature > 78))
    )
    # Flip 6% of labels to represent measurement/label noise.
    flip = rng.random(n_samples) < 0.06
    y = np.logical_xor(fault, flip).astype(int)

    X = np.column_stack(
        [
            temperature,
            dma_retries,
            bus_errors,
            queue_depth,
            link_flaps,
            noise_sensor,
        ]
    )
    return X, y


def evaluate(model, X, y):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
    }


def print_metrics(name, result):
    values = " | ".join(f"{key}={value:.3f}" for key, value in result.items())
    print(f"{name:22s} {values}")


def main():
    X, y = make_dataset()
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.25,
        random_state=43,
        stratify=y_trainval,
    )

    print(f"scikit-learn: {sklearn.__version__}")
    print(
        f"Samples: total={len(y)}, train={len(y_train)}, "
        f"validation={len(y_val)}, test={len(y_test)}"
    )
    print(f"Positive class prevalence: {y.mean():.3f}")
    assert len(y_train) + len(y_val) + len(y_test) == len(y)
    assert len(set(map(tuple, X_train)).intersection(map(tuple, X_test))) == 0

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    print("\nTEST METRICS (threshold=0.5)")
    print_metrics("Majority baseline", evaluate(baseline, X_test, y_test))

    trees = {}
    for depth in (2, 4, 6, None):
        tree = DecisionTreeClassifier(
            criterion="gini",
            max_depth=depth,
            random_state=7,
        )
        tree.fit(X_train, y_train)
        trees[depth] = tree
        label = "unrestricted" if depth is None else f"tree depth={depth}"
        print_metrics(f"{label} train", evaluate(tree, X_train, y_train))
        print_metrics(f"{label} test", evaluate(tree, X_test, y_test))

    forest = RandomForestClassifier(
        n_estimators=200,
        criterion="gini",
        max_features="sqrt",
        min_samples_leaf=3,
        class_weight="balanced",
        bootstrap=True,
        n_jobs=1,
        random_state=7,
    )
    forest.fit(X_train, y_train)
    print_metrics("Random forest train", evaluate(forest, X_train, y_train))
    print_metrics("Random forest val", evaluate(forest, X_val, y_val))
    forest_test = evaluate(forest, X_test, y_test)
    print_metrics("Random forest test", forest_test)

    print("\nDEPTH-6 TREE (first three levels)")
    print(export_text(trees[6], feature_names=FEATURES, max_depth=3))

    print("\nRANDOM FOREST IMPURITY IMPORTANCE (MDI)")
    for name, value in sorted(
        zip(FEATURES, forest.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{name:18s} {value:.4f}")

    print("\nVALIDATION PERMUTATION IMPORTANCE (ROC-AUC decrease)")
    permutation = permutation_importance(
        forest,
        X_val,
        y_val,
        n_repeats=10,
        random_state=11,
        scoring="roc_auc",
        n_jobs=1,
    )
    for index in np.argsort(permutation.importances_mean)[::-1]:
        print(
            f"{FEATURES[index]:18s} "
            f"{permutation.importances_mean[index]:.4f} "
            f"+/- {permutation.importances_std[index]:.4f}"
        )

    # Regression tests for the intended teaching observations.
    deep_tree = trees[None]
    deep_train_accuracy = accuracy_score(y_train, deep_tree.predict(X_train))
    deep_test_accuracy = accuracy_score(y_test, deep_tree.predict(X_test))
    assert deep_train_accuracy == 1.0
    assert deep_test_accuracy < deep_train_accuracy
    assert forest_test["roc_auc"] > 0.75
    assert forest_test["f1"] > 0.65

    # The noise feature should have very little validation permutation impact.
    noise_index = FEATURES.index("noise_sensor")
    temperature_index = FEATURES.index("temperature")
    assert (
        permutation.importances_mean[noise_index]
        < permutation.importances_mean[temperature_index]
    )

    print("\nPASS: split integrity, overfit demonstration, forest metrics,")
    print("      and validation permutation-importance checks succeeded.")
    print("Reminder: synthetic teaching data is not silicon sign-off evidence.")


if __name__ == "__main__":
    main()
