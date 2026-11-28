# Session 013 — Dimensionality Reduction: PCA, Eigenvectors, SVD and Feature Compression

**Phase:** 3 — Machine Learning  
**Duration:** 60 minutes  
**Prerequisites:** Sessions 003–008 and 012: vectors/matrices, gradients, probability, data preparation, clustering and evaluation.

## Learning objectives

By the end of this session, you will be able to:

1. Explain why high-dimensional telemetry creates storage, compute, visualization and distance problems.
2. Interpret eigenvalues and eigenvectors of a covariance matrix.
3. Derive PCA from covariance eigendecomposition and from singular value decomposition (SVD).
4. Select principal components using explained variance while understanding what that criterion misses.
5. Reconstruct observations and calculate reconstruction residuals.
6. Explain centering, feature scaling and whitening.
7. Build a leakage-safe PCA pipeline and compare its linear-algebra implementations.
8. Design a governed, observable agent-assisted dimensionality-reduction workflow.

## One-hour plan

| Time | Activity |
|---|---|
| 0–5 min | Why dimensionality reduction follows clustering |
| 5–15 min | Eigenvalues, eigenvectors and covariance |
| 15–27 min | PCA derivation and principal components |
| 27–35 min | SVD, reconstruction and component selection |
| 35–42 min | Scaling, whitening and limitations |
| 42–53 min | Python experiment |
| 53–58 min | Production architecture A19 |
| 58–60 min | Review questions and next step |

---

## 1. Why reduce dimensions?

A telemetry table may contain hundreds or thousands of measurements: temperature sensors, DMA counters, bus utilization, cache events, interrupt rates, queue depths and audio underrun counters.

Many measurements are correlated. For example, queue wait, backpressure and retry rate may rise together because they respond to a shared operating condition.

High dimensionality creates practical problems:

- More storage and memory traffic.
- More compute for distance calculations and model fitting.
- Harder visualization and human interpretation.
- Redundant or noisy measurements can obscure useful structure.
- Distance-based methods can become less discriminative as dimensions grow.

**Dimensionality reduction** maps the original feature vector \(x\in\mathbb{R}^{d}\) into a smaller vector \(z\in\mathbb{R}^{k}\), where \(k<d\).

~~~mermaid
flowchart LR
  A[Many correlated telemetry features] --> B[Validate units and schema]
  B --> C[Fit scaler on reference data]
  C --> D[PCA projection]
  D --> E[Compact component vector]
  E --> F[Visualization / clustering / downstream model]
  D --> G[Reconstruction residual]
  G --> H[Anomaly triage evidence]
~~~

PCA is one of the most widely used linear dimensionality-reduction methods. It finds orthogonal directions that successively explain the largest amount of variance in centered data.

## 2. Eigenvalues and eigenvectors

An eigenvector of a square matrix A is a nonzero vector v whose direction is unchanged by multiplication by A:

\[
Av=\lambda v
\]

- v: eigenvector (direction).
- λ: eigenvalue (scaling along that direction).

For a real covariance matrix C, the matrix is symmetric and positive semidefinite. Its eigenvectors can be chosen orthonormal, and its eigenvalues are non-negative apart from tiny floating-point error.

### 2.1 A small worked example

Consider this covariance matrix:

\[
C=\begin{bmatrix}4&2\\2&3\end{bmatrix}
\]

Its eigenvalues are approximately:

\[
\lambda_1=5.562,\qquad \lambda_2=1.438
\]

A unit eigenvector for the largest eigenvalue is approximately:

\[
v_1=\begin{bmatrix}0.788\\0.615\end{bmatrix}
\]

(The sign of an eigenvector is arbitrary: the negative of v1 is equally valid.)

The first direction has more variance than the second. Projecting data onto v1 retains the direction with the most variance. If we keep only this direction, we compress two coordinates into one and lose information associated with the second direction.

### 2.2 Covariance matrix

For a data matrix X with n observations and d features, first center each feature:

\[
X_c=X-\mathbf{1}\mu^T
\]

where μ is the vector of feature means.

The sample covariance matrix is:

\[
C=\frac{1}{n-1}X_c^T X_c
\]

The diagonal contains feature variances; off-diagonal entries contain covariances.

PCA solves:

\[
Cv_i=\lambda_i v_i
\]

Sort eigenvalues from largest to smallest. The associated eigenvectors are the principal directions.

## 3. Principal Component Analysis (PCA)

PCA projects centered data onto the leading eigenvectors of the covariance matrix.

Let:

\[
V_k=[v_1,v_2,\ldots,v_k]
\]

Then the reduced representation (component scores) is:

\[
Z=X_cV_k
\]

The principal components are orthogonal. PC1 captures the maximum possible variance among unit-length linear directions. PC2 captures the maximum remaining variance subject to being orthogonal to PC1, and so on.

### 3.1 Explained variance

The variance explained by component i is its eigenvalue λ_i.

The explained-variance ratio is:

\[
r_i=\frac{\lambda_i}{\sum_{j=1}^{d}\lambda_j}
\]

Cumulative explained variance through component k:

\[
R_k=\sum_{i=1}^{k}r_i
\]

A common teaching heuristic is to choose the smallest k that reaches 90%, 95% or 99% cumulative explained variance. These are not universal acceptance thresholds. Choose the threshold based on downstream utility and the cost of discarded information.

### 3.2 What the components mean

A principal component is generally a weighted combination of many original features:

\[
PC_1=w_1x_1+w_2x_2+\cdots+w_dx_d
\]

The weights are called **loadings** (terminology varies by convention). Large absolute loadings indicate which original features contribute strongly to that component.

PCA is not feature selection. It creates new coordinates; it does not simply choose a few original columns. Component signs are arbitrary, and when eigenvalues are close, the individual directions can rotate substantially while their combined subspace remains similar.

## 4. SVD — the numerical route to PCA

Singular Value Decomposition factors a centered data matrix:

\[
X_c=U\Sigma V^T
\]

- U: left singular vectors.
- Σ: diagonal matrix of singular values.
- V: right singular vectors.

The columns of V are eigenvectors of \(X_c^TX_c\). The covariance eigenvalues are:

\[
\lambda_i=\frac{\sigma_i^2}{n-1}
\]

Therefore, PCA can be computed directly using SVD without explicitly forming the covariance matrix.

**Why this matters:** forming \(X_c^TX_c\) squares the condition number. Direct SVD is generally the numerically safer default, especially when feature scales or singular values vary greatly. For a small number of features and many rows, covariance eigendecomposition can be efficient, but numerical conditioning must be considered.

~~~mermaid
flowchart TD
  A[Centered data matrix Xc] --> B{PCA solver}
  B --> C[Covariance: Xc transpose times Xc]
  C --> D[Symmetric eigensolver]
  B --> E[Direct SVD of Xc]
  D --> F[Eigenvectors and eigenvalues]
  E --> G[Right singular vectors and singular values]
  F --> H[Principal directions and explained variance]
  G --> H
  H --> I[Project onto first k components]
~~~

### 4.1 Eigen decomposition versus SVD

| Method | Main operation | Useful when | Caution |
|---|---|---|---|
| Covariance + eigensolver | Build a d×d covariance matrix, solve symmetric eigenproblem | Many samples, relatively few features | Squared condition number; covariance storage |
| Direct SVD | Factor the centered n×d data matrix | General dense PCA; numerical robustness | Full SVD can be expensive for very large matrices |
| Randomized/truncated SVD | Approximate leading singular subspace | Large matrices where k is much smaller than min(n,d) | Approximation error and random-seed/oversampling choices |
| Incremental PCA | Update components from batches | Data that does not fit in memory | Batch/order effects and approximation details |

For symmetric covariance matrices, use a symmetric/Hermitian eigensolver such as numpy.linalg.eigh, not a general eigensolver unless there is a specific reason.

## 5. Reconstruction and information loss

Given k components, reconstruct an approximation:

\[
\hat X=\mu+ZV_k^T
\]

The per-observation reconstruction residual is:

\[
e_i=\|x_i-\hat x_i\|_2^2
\]

The average reconstruction error increases or stays the same as fewer components are retained. It is the information discarded outside the retained subspace.

### PCA residuals and anomaly triage

If normal telemetry follows a low-dimensional subspace, an observation far from that subspace may have a large reconstruction residual. This is sometimes called a Q-statistic or squared prediction error.

But residual scoring has blind spots:

- An anomaly that lies within the retained subspace can have a small residual.
- A legitimate rare mode can have a large residual.
- A distribution shift can raise residuals for an entire batch.
- Residuals depend on feature selection, scaling and the reference population.

A complementary score, Hotelling's T², measures unusual distance **within** the retained component space. Use both only with a calibrated and validated policy; neither is a fault diagnosis by itself.

## 6. Centering, scaling and whitening

### Centering

PCA assumes the data has been centered. scikit-learn PCA centers features but does **not** scale each feature to unit variance automatically.

### Standardization

When features use different units or have very different scales, standardize before PCA:

\[
z_j=\frac{x_j-\mu_j}{s_j}
\]

Fit the scaler only on training/reference data, then reuse the same fitted scaler for validation, test and inference data. If the feature units are already comparable and variance magnitude is meaningful, standardization may not be appropriate. Make the choice explicit.

### Whitening

Whitening scales projected components to unit variance. This removes relative variance magnitude between retained components. It can help downstream algorithms that expect isotropic features, but it can amplify low-variance noise and removes information about relative component variance.

Do not enable whitening automatically. Compare downstream performance with and without it.

## 7. Hands-on lab — compress synthetic SoC telemetry

The lab generates 12 correlated telemetry measurements from three latent operating factors. Features intentionally have different physical scales and offsets.

The data is split into:
- 800 training/reference observations.
- 300 nominal calibration observations.
- 300 nominal test observations.
- 30 injected test anomalies.

The scaler and PCA are fitted on the training data only. Calibration data sets a 99th-percentile residual threshold; the test labels are used only after scoring to report teaching precision and recall.

### Reproduced results

**Compression and explained variance**

| Component | Explained variance | Cumulative variance |
|---|---:|---:|
| PC1 | 54.2850% | 54.2850% |
| PC2 | 21.9311% | 76.2160% |
| PC3 | 20.7380% | 96.9540% |
| PC4 | 0.4345% | 97.3885% |
| PC5 | 0.3911% | 97.7796% |
| PC6 | 0.3820% | 98.1616% |

The 95% heuristic selects 3 components:

- Original representation: 12 features.
- Reduced representation: 3 components.
- Coordinate count reduction: 4×.
- Training variance retained: 96.9540%.

**Independent linear-algebra checks**

| Check | Result |
|---|---:|
| Max covariance-eigenvalue vs PCA variance difference | 1.155 × 10⁻¹⁴ |
| Max SVD-derived variance vs PCA variance difference | 0 |
| Retained component orthogonality error | 1.332 × 10⁻¹⁵ |

These checks confirm that the covariance eigenproblem and direct SVD agree numerically for this well-conditioned teaching dataset.

**PCA reconstruction-residual triage**

| Measure | Result |
|---|---:|
| 99th-percentile nominal calibration threshold | 0.869580 |
| Injected anomalies detected | 30 / 30 |
| Nominal test observations flagged | 3 / 300 |
| Precision against injected synthetic truth | 90.91% |
| Recall against injected synthetic truth | 100.00% |

The injected anomalies were designed to depart from the learned low-dimensional subspace. These numbers are a controlled teaching result, not evidence that PCA will achieve this performance on real silicon data.

### Run locally

~~~bash
cd daily-lessons/day-013
python -m pip install -r requirements.txt
python lab.py
~~~

Expected final line:

~~~text
PASS: all Day 013 lab assertions passed.
~~~

### Experiment tasks

1. Change the retained-variance target from 95% to 90% and 99%. Compare component count and reconstruction residual.
2. Run PCA without standardization. Observe how features with large numeric scales influence the first component.
3. Change the residual threshold from the 99th to the 95th calibration percentile. Record the false-positive/recall trade-off.
4. Explain why an anomaly lying inside the retained principal subspace may not be detected by reconstruction residual.
5. Keep the training/reference window and scaler fixed while projecting a later batch. Then intentionally refit the scaler and explain why the coordinate system is no longer directly comparable.

## 8. Production agent architecture — A19

### PCA Representation and Compression Governance Gate

The agent can translate an engineering request into a typed experiment proposal and explain results. Deterministic services own data access, preprocessing, PCA fitting, scoring, artifact registration and authorization.

~~~mermaid
flowchart TD
  U[Engineer request] --> I[Intake agent: typed reduction contract]
  I --> V[Schema, tenant, units and data validator]
  V --> S[Immutable reference split and versioned scaler]
  S --> P[Deterministic PCA fit service]
  P --> E[Variance, reconstruction and stability evaluator]
  E --> D[Downstream clustering / diagnostic utility checks]
  D --> X[Independent lineage and leakage verifier]
  X --> H[Human engineering approval]
  H --> R[Versioned PCA artifact registry]
  R --> Q[Controlled projection and residual-scoring service]
~~~

### Component boundaries

| Component | Owns | Must not own |
|---|---|---|
| Intake agent | Translate request into typed PCA proposal | Direct dataset access or authorization |
| Data validator | Tenant, schema, feature order, units and provenance | Model interpretation |
| Preprocessing service | Versioned scaler fit/transform | Silent refitting on production batches |
| PCA fit service | Solver, component fit and immutable model artifact | Deployment approval |
| Evaluator | Explained variance, reconstruction, stability and downstream utility | Declaring a fault from residual alone |
| Verifier | Recompute/check artifact lineage and leakage | Altering candidate results |
| Human reviewer | Approve component policy and operational use | Unlogged manual changes |
| Registry | Store scaler, PCA model, metadata and approvals | Mutable unversioned artifacts |

### Reusable orchestration patterns and trade-offs

- **Deterministic workflow:** default for a known PCA training and validation protocol. It provides reproducibility and a clear data boundary.
- **Single agent + typed tools:** useful for requests such as “compress the last month of DMA and PCIe telemetry to 95% explained variance.” The agent proposes a configuration; services enforce feature, tenant, solver and resource policies.
- **Router–specialist:** route small dense matrices to exact SVD, large low-rank matrices to randomized SVD, and batch-streamed datasets to IncrementalPCA. A deterministic policy checks dimensions and memory estimates before dispatch.
- **Planner–executor–verifier:** useful when comparing component counts, scaling policies and downstream clustering. Persist a bounded plan; independently verify each candidate's scaler and PCA lineage.
- **Supervisor–worker:** run independent solver/configuration candidates in parallel only when they use the same immutable training snapshot. Reserve CPU/GPU and memory budgets atomically.
- **Event-driven:** use a versioned reference-window-ready event for scheduled retraining and a separate projection-batch-ready event for inference. Consumers must be idempotent.
- **Human-in-the-loop:** require review before changing production feature schemas, component count, residual thresholds or downstream alert/remediation behavior.

Start with a deterministic pipeline and a single agent only if natural-language intake or explanation adds measurable value. Multiple agents do not improve the numerical quality of PCA.

### Typed tool contract example

~~~json
{
  "tool": "fit_pca_representation",
  "input": {
    "dataset_version": "telemetry-reference-window-sha256",
    "feature_schema_version": "soc-telemetry-v3",
    "feature_order": [
      "temperature",
      "dma_retry_rate",
      "bus_error_rate",
      "queue_pressure"
    ],
    "preprocessing": {
      "scaler": "standard",
      "fit_scope": "training_reference_only"
    },
    "solver": "full",
    "component_policy": {
      "type": "minimum_cumulative_variance",
      "target": 0.95,
      "max_components": 64
    },
    "resource_budget": {
      "max_rows": 1000000,
      "max_features": 512,
      "deadline_seconds": 900
    }
  }
}
~~~

The authenticated service derives tenant scope; never trust a tenant ID supplied by model-generated arguments. Validate feature names/order, units, solver enum, component bounds, data digest, missing-value policy and resource budget at the tool boundary. Do not expose arbitrary shell or unrestricted file tools to the agent.

### State, memory and artifact lineage

Persist the following as one immutable representation version:

- dataset/window digest and tenant-safe provenance
- feature schema version, exact feature order and units
- scaler means/scales and preprocessing code version
- PCA mean, components, singular values and explained variance
- selected component count, solver, dtype and random seed
- training/reference window and calibration policy
- Python, NumPy, scikit-learn and BLAS/LAPACK versions
- fit metrics, reconstruction thresholds and evaluation populations
- code commit, artifact digest, reviewer and approval record

Keep conversation memory separate from numerical model state. A PCA artifact without its exact scaler and feature order is not a reproducible representation.

### Security and data boundaries

- Enforce tenant isolation and dataset permissions in data services.
- Treat log strings, register dumps and uploaded documents as untrusted data, not agent instructions.
- Keep raw telemetry out of LLM prompts and traces unless explicitly authorized and redacted.
- Use an allow-list for features, solvers and component policies.
- Prevent the agent from changing the training/reference window or reading protected evaluation labels.
- Never let an LLM-generated feature order, scaler or artifact path bypass validation.
- Require approval before changing production alert thresholds or suppressing signals based on a compressed representation.

### Evaluation, reliability and observability

Evaluate more than explained variance:

- cumulative explained variance and selected component count
- reconstruction error on held-out reference data
- downstream clustering/classification quality and fault-signal retention
- stability of the retained subspace across time windows
- residual-score distribution and false-alert budget
- orthogonality, finite values and numerical conditioning
- CPU/GPU time, peak memory, artifact size and projection latency

Operational controls:

- Fit preprocessing on the designated training/reference window only.
- Persist immutable scaler and PCA artifacts; do not refit silently during inference.
- Use idempotency keys based on dataset digest, feature schema and fit configuration.
- Bound rows, features, candidate count, concurrency, memory and deadlines.
- Retry transient infrastructure failures without changing solver parameters.
- Checkpoint long-running jobs and support cancellation.
- Trace request ID, experiment ID, data digest, feature schema, scaler/PCA version, solver, dimensions, component count, explained variance, residual statistics, fit time, projection latency and reviewer decision.
- Monitor input drift, component-space drift and reconstruction residual drift.
- Measure cost per successfully reviewed representation update, not just LLM token cost.

### Failure modes to test

1. Scaler is refitted on every incoming batch, so coordinates drift.
2. Feature order changes but the old PCA artifact is reused.
3. PCA is fitted before the train/test split, leaking evaluation data.
4. A low-variance but safety-critical fault feature is discarded.
5. A fault signature lies within the retained subspace and has a small residual.
6. A legitimate new operating mode produces large residuals and false alarms.
7. Near-equal eigenvalues cause component axes to rotate between retraining runs.
8. A retry creates duplicate model artifacts or deployment events.
9. A worker exceeds GPU memory or deadline limits.
10. A model-generated request attempts cross-tenant access or unrestricted feature selection.

### Acceptance criteria

A reusable implementation should demonstrate:

- deterministic, versioned preprocessing and exact feature order
- covariance-eigen and SVD agreement on a well-conditioned test case
- orthonormal retained components and finite outputs
- explicit explained-variance and reconstruction metrics
- held-out evaluation with no preprocessing leakage
- residual-threshold policy calibrated on reference data
- artifact lineage including scaler, PCA and library versions
- tenant-isolation and prompt-injection boundary tests
- bounded execution, idempotent retries and traceable events
- human approval before production representation or alert-policy changes

## 9. Key terminology

| Term | Definition |
|---|---|
| Eigenvalue | Scalar describing how a matrix scales an eigenvector |
| Eigenvector | Nonzero direction preserved by a square matrix transformation |
| Covariance matrix | Matrix of variances and pairwise covariances |
| Principal component | Orthogonal direction capturing ordered variance in centered data |
| Loading | Weight connecting an original feature to a component |
| Explained variance ratio | Fraction of total variance captured by a component |
| SVD | Factorization \(X=U\Sigma V^T\) |
| Singular value | Non-negative diagonal value in Σ |
| Projection | Mapping original observations into component coordinates |
| Reconstruction residual | Difference between an observation and its low-rank reconstruction |
| Whitening | Rescaling retained components to unit variance |
| Incremental PCA | Batch-wise PCA update for data too large to fit at once |
| Subspace drift | Change in the learned principal directions over time |

## 10. Five review questions

1. Why are eigenvectors of a covariance matrix useful for finding principal components?
2. How are covariance eigenvalues related to singular values of the centered data matrix?
3. Why does scikit-learn PCA require explicit scaling when feature units differ?
4. Why can high explained variance still discard a safety-critical low-variance fault signal?
5. What exact artifacts must be versioned so that a production PCA projection can be reproduced?

## 11. References

1. [scikit-learn — PCA user guide](https://scikit-learn.org/stable/modules/decomposition.html#principal-component-analysis-pca): PCA, explained variance, whitening and solver choices.
2. [scikit-learn — PCA API](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html): solver behavior, components and variance attributes.
3. [NumPy — Singular Value Decomposition](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html): SVD factors and the relationship between singular values and eigenvalues.
4. [SciPy — Symmetric/Hermitian eigenproblems](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html): stable eigenvalue/eigenvector computation for covariance matrices.
5. Jolliffe and Cadima (2016), [Principal component analysis: a review and recent developments](https://doi.org/10.1098/rsta.2015.0202).
6. Halko, Martinsson and Tropp (2011), [Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions](https://doi.org/10.1137/090771806).

---

## Next session — Session 014

**Anomaly Detection: Isolation Forest, Local Outlier Factor and One-Class SVM**

We will compare tree-isolation, local-density and boundary-based anomaly detectors; discuss contamination assumptions and threshold calibration; and build a synthetic SoC telemetry triage lab. PCA residuals will be one baseline, not the only detector.
