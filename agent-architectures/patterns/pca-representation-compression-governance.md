# A19 — PCA Representation and Compression Governance Gate

## Purpose

Reduce high-dimensional telemetry into a versioned, reproducible representation without allowing an agent to silently change feature semantics, leak evaluation data or discard safety-critical signals.

The pattern governs scaler fitting, PCA/SVD selection, component-count evaluation, reconstruction-residual analysis and controlled projection. It extends the experiment-governance family from clustering and supervised learning into linear representation learning.

## Reference workflow

~~~mermaid
flowchart TD
  U[Engineer request] --> I[Typed reduction contract]
  I --> V[Schema, tenant, units and data validation]
  V --> S[Immutable reference split and versioned scaler]
  S --> P[Deterministic PCA fit service]
  P --> E[Variance, reconstruction and stability evaluator]
  E --> D[Downstream clustering / diagnostic utility checks]
  D --> X[Independent lineage and leakage verifier]
  X --> H[Human engineering approval]
  H --> R[Versioned PCA artifact registry]
  R --> Q[Controlled projection and residual-scoring service]
~~~

## Typed request contract

~~~json
{
  "experiment_id": "generated-by-service",
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
  },
  "tenant_scope": "derived-from-authenticated-identity"
}
~~~

Tenant scope is derived by the service from authenticated identity; it is never trusted from model-generated text.

## Component responsibilities

1. **Intake agent:** translate an engineer request into a typed PCA proposal and explain missing choices.
2. **Data validator:** verify tenant boundary, schema, feature order, units, provenance, missing values and sample limits.
3. **Preprocessing service:** fit the scaler on the declared training/reference window and persist its parameters; reuse it for all later transforms.
4. **PCA fit service:** run an approved exact, randomized or incremental solver and produce an immutable artifact.
5. **Evaluator:** calculate explained variance, reconstruction error, subspace stability, downstream utility and resource use.
6. **Verifier:** independently check feature order, scaler/PCA lineage, evaluation population and absence of preprocessing leakage.
7. **Human reviewer:** approve component policy, downstream use and any production alert-policy change.
8. **Registry/projection service:** store the paired scaler/PCA artifact and apply it consistently at inference.

## Orchestration patterns and trade-offs

| Pattern | Use | Guardrail |
|---|---|---|
| Deterministic workflow | Fixed PCA training and validation protocol | Immutable reference split, scaler and artifact lineage |
| Single agent + typed tools | Natural-language request translation and report explanation | Agent proposes; services enforce feature, tenant and resource policy |
| Router + specialists | Choose exact SVD, randomized SVD or IncrementalPCA | Deterministic size/memory policy validates the route |
| Planner–executor–verifier | Compare component counts, scalers and downstream use | Persist a bounded plan and independently verify every artifact |
| Supervisor–worker | Parallel solver/configuration comparisons | Same immutable data snapshot; atomic CPU/GPU and memory budgets |
| Event-driven | Scheduled reference-window fitting and projection batches | Idempotent event keys, checkpoints and bounded retries |
| Human-in-the-loop | Change feature schema, component policy, residual threshold or operational use | Reviewer identity, rationale and artifact digest are mandatory |

Start with a deterministic workflow. Add an agent for natural-language intake or interpretation only when it provides measurable value. Multiple agents do not improve the numerical quality of PCA.

## Solver selection policy

- **Full SVD:** robust default for moderate dense matrices and reference-quality comparisons.
- **Covariance eigensolver:** potentially efficient when sample count is much larger than feature count, but forming XᵀX squares the condition number.
- **Randomized SVD:** useful when only a small number of leading components is needed from a large matrix; record seed, oversampling and iteration parameters.
- **IncrementalPCA:** useful when the dataset cannot fit in memory; record batch size and batch order.
- **TruncatedSVD:** consider for sparse matrices when centering would destroy sparsity; it is not identical to centered PCA.

Route only after checking matrix shape, dtype, sparsity, memory estimate, target rank and acceptable approximation error.

## Evaluation contract

Do not select a PCA representation from explained variance alone. Record:

- cumulative explained variance and retained component count
- reconstruction error on held-out reference data
- downstream clustering/classification utility
- preservation of known fault signatures and rare but safety-critical patterns
- subspace stability across time windows and retraining runs
- residual score distribution, threshold policy and false-alert budget
- component orthogonality, finite values and numerical conditioning
- fit time, projection latency, peak memory, GPU utilization and artifact size

A high-variance direction is not necessarily predictive or safety-critical. A low-variance fault signature can be discarded. A reconstruction residual detects departures from the retained subspace, not every possible anomaly. Consider within-subspace score distance (e.g. Hotelling's T-squared) as a complementary signal, with its own validation.

## State and artifact provenance

Persist as one immutable representation version:

- experiment/request ID and immutable dataset/window digest
- tenant-safe provenance
- feature schema version, exact feature order and units
- scaler means/scales and preprocessing code version
- PCA mean, components, singular values and explained variance
- selected component count, solver, dtype, seed and solver parameters
- training/reference window and calibration policy
- Python, NumPy, scikit-learn and BLAS/LAPACK versions
- evaluation metrics and exact populations
- code commit, artifact digest, reviewer identity and approval record

A PCA artifact without its exact scaler and feature order is not a reproducible representation. Keep conversational memory separate from numerical model state.

## Security and data boundaries

- Enforce tenant isolation and dataset permissions in data services.
- Treat logs, register dumps and uploaded documents as untrusted data, not agent instructions.
- Keep raw telemetry out of prompts and traces unless explicitly authorized and redacted.
- Use allow-lists for features, solvers and component policies.
- Prevent the agent from changing the reference window or reading protected evaluation labels.
- Validate all generated arguments at the tool boundary; do not trust model-generated tenant IDs, paths or feature order.
- Require approval before changing production schemas, alert thresholds or remediation behavior.

## Reliability and operations

- Fit preprocessing on the designated reference/training window only.
- Persist immutable scaler and PCA artifacts; never refit silently during inference.
- Use idempotency keys based on dataset digest, feature schema and fit configuration.
- Bound rows, features, candidate count, concurrency, CPU/GPU memory and deadlines.
- Retry transient infrastructure errors without silently changing solver parameters.
- Checkpoint long-running jobs and support cancellation.
- Record trace ID, experiment ID, data digest, feature schema, scaler/PCA version, solver, dimensions, component count, explained variance, residual statistics, fit time, projection latency and reviewer decision.
- Monitor input drift, component-space drift and reconstruction-residual drift.
- Measure cost per successfully reviewed representation update, not only LLM token cost.

## Failure modes to test

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

## Acceptance criteria

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

## Recommended baseline

Use a deterministic pipeline with full SVD for a moderate reference dataset. Add an agent for request translation and report explanation. Route to randomized or incremental methods only when data shape and resource measurements justify it. Never use PCA compression as the sole reason to suppress an alert or declare a hardware fault.
