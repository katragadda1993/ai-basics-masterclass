#!/usr/bin/env python3
"""Session 012: reproducible clustering and anomaly-triage lab.

The data are synthetic and intentionally unlabeled to the clustering algorithms.
Synthetic generator labels are retained only for a post-hoc teaching metric (ARI).
"""
from __future__ import annotations

import numpy as np
import sklearn
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


SEED = 42
N_INLIERS = 900
N_OUTLIERS = 30
FEATURE_NAMES = (
    "temperature",
    "dma_retry_rate",
    "bus_error_rate",
    "queue_pressure",
)


def build_dataset():
    """Return features, latent modes (post-hoc only), anomaly truth."""
    X_inliers, latent_modes = make_blobs(
        n_samples=N_INLIERS,
        centers=[
            [-2.0, -2.0, -1.0, -1.0],
            [2.0, 2.0, 1.0, 1.0],
            [-2.0, 2.0, 1.0, -2.0],
        ],
        cluster_std=[0.55, 0.70, 0.60],
        random_state=SEED,
    )
    rng = np.random.default_rng(SEED)
    X_outliers = rng.uniform(-7.0, 7.0, size=(N_OUTLIERS, 4))
    X = np.vstack([X_inliers, X_outliers])
    anomaly_truth = np.r_[
        np.zeros(N_INLIERS, dtype=int),
        np.ones(N_OUTLIERS, dtype=int),
    ]
    return X, latent_modes, anomaly_truth


def safe_silhouette(X, labels):
    """For DBSCAN, compute silhouette on non-noise points only."""
    labels = np.asarray(labels)
    mask = labels != -1
    X_eval, y_eval = X[mask], labels[mask]
    if len(set(y_eval.tolist())) < 2 or len(set(y_eval.tolist())) >= len(y_eval):
        return float("nan")
    return float(silhouette_score(X_eval, y_eval))


def main():
    X_raw, latent_modes, anomaly_truth = build_dataset()
    assert X_raw.shape == (N_INLIERS + N_OUTLIERS, 4)
    assert np.isfinite(X_raw).all()

    # Exploratory reference batch. In production, fit and version the scaler
    # on a designated reference window, then reuse it on future observations.
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    print(f"scikit-learn: {sklearn.__version__}")
    print(f"Samples: {len(X)} ({N_INLIERS} generated inliers, {N_OUTLIERS} injected anomalies)")
    print(f"Features: {', '.join(FEATURE_NAMES)}")
    print("\nK-Means candidate sweep")
    print("k | inertia       | silhouette | smallest cluster")
    kmeans_results = {}
    for k in range(2, 7):
        model = KMeans(n_clusters=k, n_init=10, random_state=SEED)
        labels = model.fit_predict(X)
        counts = np.bincount(labels)
        sil = float(silhouette_score(X, labels))
        kmeans_results[k] = (model, labels, sil, int(counts.min()))
        print(f"{k} | {model.inertia_:12.3f} | {sil:10.4f} | {counts.min():16d}")

    # Teaching policy: every cluster must contain >=3% of this reference batch.
    # This is not a universal production threshold.
    min_cluster_size = int(np.ceil(0.03 * len(X)))
    eligible = [
        (k, result)
        for k, result in kmeans_results.items()
        if result[3] >= min_cluster_size
    ]
    chosen_k, (kmeans_model, kmeans_labels, kmeans_sil, _) = max(
        eligible, key=lambda item: item[1][2]
    )
    assert chosen_k == 3
    ari_kmeans = adjusted_rand_score(latent_modes, kmeans_labels[:N_INLIERS])

    print("\nSelected K-Means candidate")
    print(f"Minimum cluster size policy: {min_cluster_size}")
    print(f"Selected k: {chosen_k}")
    print(f"Silhouette: {kmeans_sil:.4f}")
    print(f"ARI on generated inliers (post-hoc only): {ari_kmeans:.4f}")

    agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
    agg_labels = agg.fit_predict(X)
    agg_sil = float(silhouette_score(X, agg_labels))
    ari_agg = adjusted_rand_score(latent_modes, agg_labels[:N_INLIERS])
    print("\nAgglomerative (Ward, 3 clusters)")
    print(f"Silhouette: {agg_sil:.4f}")
    print(f"ARI on generated inliers (post-hoc only): {ari_agg:.4f}")

    db_labels = DBSCAN(eps=0.60, min_samples=8).fit_predict(X)
    cluster_ids = set(db_labels.tolist()) - {-1}
    noise_mask = db_labels == -1
    noise_fraction = float(noise_mask.mean())
    tp = int(np.sum(noise_mask & (anomaly_truth == 1)))
    fp = int(np.sum(noise_mask & (anomaly_truth == 0)))
    fn = int(np.sum((~noise_mask) & (anomaly_truth == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    db_sil = safe_silhouette(X, db_labels)

    print("\nDBSCAN (eps=0.60, min_samples=8)")
    print(f"Clusters (excluding noise): {len(cluster_ids)}")
    print(f"Noise samples: {int(noise_mask.sum())}")
    print(f"Noise fraction: {noise_fraction:.4f}")
    print(f"Silhouette on non-noise points only: {db_sil:.4f}")
    print(f"Injected-anomaly precision (teaching truth): {precision:.4f}")
    print(f"Injected-anomaly recall (teaching truth): {recall:.4f}")

    print("\nDBSCAN eps sensitivity (min_samples=8)")
    for eps in (0.50, 0.55, 0.60, 0.65):
        labels = DBSCAN(eps=eps, min_samples=8).fit_predict(X)
        predicted_noise = labels == -1
        n_clusters = len(set(labels.tolist()) - {-1})
        detected = int(np.sum(predicted_noise & (anomaly_truth == 1)))
        false_alarms = int(np.sum(predicted_noise & (anomaly_truth == 0)))
        anomaly_precision = detected / max(detected + false_alarms, 1)
        anomaly_recall = detected / N_OUTLIERS
        print(
            f"eps={eps:.2f}: clusters={n_clusters}, "
            f"noise={predicted_noise.mean():.4f}, "
            f"precision={anomaly_precision:.4f}, recall={anomaly_recall:.4f}"
        )

    # Acceptance checks use structure and tolerances, not brittle exact
    # floating-point snapshots across every future sklearn/BLAS build.
    assert len(set(kmeans_labels)) == 3
    assert ari_kmeans > 0.99
    assert len(set(agg_labels)) == 3
    assert ari_agg > 0.99
    assert len(cluster_ids) == 3
    assert int(noise_mask.sum()) == 40
    assert abs(noise_fraction - (40 / 930)) < 1e-12
    assert abs(precision - 0.75) < 1e-12
    assert abs(recall - 1.0) < 1e-12
    assert 0.0 <= db_sil <= 1.0

    print("\nPASS: all Day 012 lab assertions passed.")


if __name__ == "__main__":
    main()
