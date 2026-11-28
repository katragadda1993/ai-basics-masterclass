#!/usr/bin/env python3
"""Session 013: PCA, eigenvectors, SVD and reconstruction-residual lab.

Synthetic telemetry is generated from three latent factors and expanded into
12 correlated, differently scaled measurements. PCA sees only feature matrices.
Injected test anomalies are used only for post-hoc teaching evaluation.
"""
from __future__ import annotations

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SEED = 13
N_FEATURES = 12
FEATURE_NAMES = (
    "temperature", "dma_retry_rate", "dma_queue_wait", "bus_error_rate",
    "axi_backpressure", "pcie_replay_rate", "audio_underrun_rate",
    "memory_stall_rate", "interrupt_rate", "power_rail_noise",
    "cache_miss_rate", "thermal_gradient",
)
LOADINGS = np.array([
    [1.0, 0.1, 0.0], [0.9, 0.2, 0.1], [0.8, 0.0, 0.2],
    [0.1, 1.0, 0.1], [0.2, 0.9, 0.0], [0.0, 0.8, 0.2],
    [0.1, 0.1, 1.0], [0.0, 0.2, 0.9], [0.2, 0.0, 0.8],
    [0.6, 0.5, 0.1], [0.1, 0.6, 0.5], [0.4, 0.1, 0.7],
], dtype=float)
SCALES = np.array([1, 10, 0.1, 5, 0.2, 20, 2, 0.5, 8, 0.1, 4, 15], dtype=float)
OFFSETS = np.array([40, 100, 0.01, 2, 0.5, 20, 3, 0.1, 50, 0.02, 8, 30], dtype=float)


def make_batch(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate correlated telemetry from three latent operating factors."""
    latent = rng.normal(size=(n, 3))
    noise = rng.normal(scale=0.18, size=(n, N_FEATURES))
    return (latent @ LOADINGS.T + noise) * SCALES + OFFSETS


def reconstruction_residual(model: PCA, X: np.ndarray) -> np.ndarray:
    """Per-row squared reconstruction error in PCA input space."""
    reconstructed = model.inverse_transform(model.transform(X))
    return np.sum((X - reconstructed) ** 2, axis=1)


def main() -> None:
    rng = np.random.default_rng(SEED)
    X_train = make_batch(800, rng)
    X_calibration = make_batch(300, rng)
    X_test_nominal = make_batch(300, rng)
    X_test_anomaly = make_batch(30, rng)

    # Inject departures from the learned low-dimensional subspace.
    signs = rng.choice([-1.0, 1.0], size=(len(X_test_anomaly), 4))
    X_test_anomaly[:, [2, 5, 8, 11]] += signs * np.array([0.8, 100, 30, 150])
    X_test = np.vstack([X_test_nominal, X_test_anomaly])
    y_test = np.r_[
        np.zeros(len(X_test_nominal), dtype=int),
        np.ones(len(X_test_anomaly), dtype=int),
    ]

    assert X_train.shape[1] == N_FEATURES
    assert np.isfinite(X_train).all() and np.isfinite(X_test).all()

    # Fit preprocessing on training/reference data only; reuse it unchanged.
    scaler = StandardScaler().fit(X_train)
    Z_train = scaler.transform(X_train)
    Z_calibration = scaler.transform(X_calibration)
    Z_test = scaler.transform(X_test)

    # Exact SVD-based PCA. Select the smallest k reaching 95% cumulative variance.
    full_pca = PCA(n_components=None, svd_solver="full").fit(Z_train)
    cumulative = np.cumsum(full_pca.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, 0.95) + 1)
    pca = PCA(n_components=n_components, svd_solver="full").fit(Z_train)

    # Independently solve the covariance eigenproblem.
    covariance = np.cov(Z_train, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    max_eigenvalue_difference = float(
        np.max(np.abs(eigenvalues - full_pca.explained_variance_))
    )

    # Independently inspect singular values of the centered training matrix.
    centered = Z_train - Z_train.mean(axis=0)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    svd_variances = singular_values**2 / (len(Z_train) - 1)
    max_svd_variance_difference = float(
        np.max(np.abs(svd_variances - full_pca.explained_variance_))
    )

    # Orthonormality of retained principal axes.
    component_gram = pca.components_ @ pca.components_.T
    orthogonality_error = float(
        np.max(np.abs(component_gram - np.eye(n_components)))
    )

    # A 99th-percentile calibration threshold is a teaching policy, not a
    # universal operating threshold. Calibration data are nominal in this lab.
    calibration_scores = reconstruction_residual(pca, Z_calibration)
    threshold = float(np.quantile(calibration_scores, 0.99))
    test_scores = reconstruction_residual(pca, Z_test)
    predicted_anomaly = test_scores > threshold

    tp = int(np.sum(predicted_anomaly & (y_test == 1)))
    fp = int(np.sum(predicted_anomaly & (y_test == 0)))
    fn = int(np.sum((~predicted_anomaly) & (y_test == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    raw_pca = PCA(n_components=1, svd_solver="full").fit(X_train)
    standardized_pca = PCA(n_components=1, svd_solver="full").fit(Z_train)

    print(f"NumPy: {np.__version__} | scikit-learn: {sklearn.__version__}")
    print(f"Training samples: {len(X_train)} | Features: {N_FEATURES}")
    print(f"Feature compression: {N_FEATURES} -> {n_components} "
          f"({N_FEATURES / n_components:.1f}x fewer coordinates)")
    print("\nExplained variance")
    for index, ratio in enumerate(full_pca.explained_variance_ratio_[:6], start=1):
        print(f"PC{index}: {ratio:.4%} | cumulative: {cumulative[index - 1]:.4%}")
    print(f"Selected components for >=95% variance: {n_components}")
    print(f"Retained variance: {cumulative[n_components - 1]:.4%}")
    print(f"Top-PC variance ratio without scaling: "
          f"{raw_pca.explained_variance_ratio_[0]:.4%}")
    print(f"Top-PC variance ratio after scaling: "
          f"{standardized_pca.explained_variance_ratio_[0]:.4%}")

    print("\nIndependent linear-algebra checks")
    print(f"Max covariance-eigenvalue vs PCA variance difference: "
          f"{max_eigenvalue_difference:.3e}")
    print(f"Max SVD-derived variance vs PCA variance difference: "
          f"{max_svd_variance_difference:.3e}")
    print(f"Retained component orthogonality error: {orthogonality_error:.3e}")

    print("\nPCA reconstruction-residual triage")
    print(f"99th-percentile calibration threshold: {threshold:.6f}")
    print(f"True injected anomalies detected: {tp}/{len(X_test_anomaly)}")
    print(f"Nominal observations flagged: {fp}/{len(X_test_nominal)}")
    print(f"Precision: {precision:.2%} | Recall: {recall:.2%}")
    print(f"Mean residual, nominal test: {test_scores[:len(X_test_nominal)].mean():.6f}")
    print(f"Mean residual, injected anomalies: {test_scores[len(X_test_nominal):].mean():.6f}")

    # Numerical and behavioral acceptance checks.
    assert n_components == 3
    assert cumulative[n_components - 1] >= 0.95
    assert cumulative[n_components - 2] < 0.95
    assert max_eigenvalue_difference < 1e-10
    assert max_svd_variance_difference < 1e-10
    assert orthogonality_error < 1e-10
    assert tp == 30
    assert recall >= 0.95
    assert precision >= 0.85
    assert fp <= 10
    assert np.isfinite(calibration_scores).all()
    assert np.isfinite(test_scores).all()

    print("\nPASS: all Day 013 lab assertions passed.")


if __name__ == "__main__":
    main()
