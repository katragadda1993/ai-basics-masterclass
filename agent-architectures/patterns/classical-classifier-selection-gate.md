# A17 — Classical Classifier Selection Gate

## Purpose

Run a reproducible comparison of distance-based, probabilistic and margin-based classifiers while preserving dataset boundaries, preprocessing provenance and approval controls.

This pattern is an extension of the ML Experiment Governance Agent, Objective-Aware Model Selection Gate and Tree Ensemble Experiment Gate.

## Reference workflow

~~~mermaid
flowchart TD
  U[Request] --> I[Typed problem contract]
  I --> D[Dataset and schema validation]
  D --> S[Immutable split manifest]
  S --> R[Bounded candidate runner]
  R --> K[KNN + fitted scaler]
  R --> N[Naive Bayes]
  R --> V[Linear / RBF SVM + fitted scaler]
  K --> E[Validation evaluator]
  N --> E
  V --> E
  E --> G[Deterministic metric and resource gate]
  G --> F[Frozen candidate]
  F --> T[Isolated test gate]
  T --> H[Human approval]
  H --> P[Versioned model registry]
~~~

## Component responsibilities

1. **Intake/router:** parse the task into a typed prediction contract; propose candidate families without granting permissions.
2. **Dataset validator:** verify feature schema, label semantics, provenance and split membership.
3. **Split service:** create and persist train/validation/test membership before candidate execution.
4. **Preprocessing pipeline:** fit transformations only on training data or training folds; package them with the estimator.
5. **Candidate runners:** run approved KNN, GaussianNB, linear SVM and kernel SVM configurations under compute limits.
6. **Validation evaluator:** compute task-appropriate metrics, calibration where probabilities are used, and latency/memory measurements.
7. **Policy gate:** apply predeclared acceptance thresholds and select a candidate using validation data only.
8. **Test gate:** expose the final test partition only to the frozen candidate evaluation service.
9. **Human reviewer:** approve model promotion and record reviewer, rationale and artifact digest.
10. **Registry:** store the full model artifact and lineage.

## Candidate configuration contract

~~~json
{
  "candidate_id": "svc-rbf-001",
  "algorithm": "rbf_svm",
  "feature_schema_version": "telemetry-v3",
  "preprocessing_version": "standard-scaler-v1",
  "hyperparameters": {"C": 0.5, "gamma": "scale"},
  "training_data_version": "dataset-sha256",
  "split_manifest_id": "split-2026-11",
  "random_seed": 17,
  "code_version": "git-commit"
}
~~~

Validate enum values and numeric ranges before dispatch. The example is illustrative; production IDs must be generated and recorded by the experiment service.

## Architecture selection and trade-offs

| Pattern | Appropriate use | Risk / control |
|---|---|---|
| Deterministic workflow | Fixed model comparison protocol | Least orchestration overhead; version every step |
| Single agent + typed tools | Natural-language experiment setup | Agent proposes; tools validate and enforce |
| Router + specialists | Distinct data/training/review domains | Wrong routing; add fallback and route evaluation |
| Planner–executor–verifier | Open-ended experiment planning | Plan drift; verify each artifact against explicit criteria |
| Supervisor–worker | Independent model candidates can run concurrently | Resource contention; atomic budgets and deterministic aggregation |
| Event-driven | Long-running training jobs | Duplicate delivery; idempotency keys and checkpoints |
| Human-in-the-loop | Promotion or high-impact deployment | Approval bottleneck; make decisions auditable and resumable |

Start with a deterministic workflow. Add an LLM only for tasks where language understanding or flexible planning adds measurable value. Add multiple agents only for genuine isolation, ownership or parallelism needs.

## State and memory

Persist:
- experiment and candidate IDs
- dataset and split-manifest digests
- feature and preprocessing schema versions
- candidate configurations and code/model versions
- status of each state transition
- validation and final test report references
- reviewer decision and artifact digest

Keep transient reasoning separate from durable experiment state. A chat transcript is not a reliable experiment database.

## Security and data boundaries

- Candidate workers must not receive final test labels.
- Enforce dataset access at the data/tool service, not in a system prompt.
- Treat dataset text, logs and retrieved content as untrusted input.
- Do not expose secrets or proprietary data in model context or traces.
- Validate tool schemas and allow-list algorithms and hyperparameters.
- Require human approval before deployment or external writes.
- Preserve audit events with sensitive values redacted.

## Reliability and operations

- Bound candidate count, CPU, RAM, wall-clock time and cost.
- Use job IDs, idempotency keys, checkpoints, cancellation and dead-letter handling.
- Retry only transient failures; do not silently change a candidate's configuration on retry.
- Record per-candidate training time, prediction latency, peak memory and artifact size.
- Track quality, calibration, false-positive/false-negative cost and subgroup performance.
- Re-run a fixed regression suite whenever data, preprocessing, library, model or prompt versions change.

## Evaluation contract

Before running experiments, specify:
- primary validation metric and tie-breaking rules
- minimum recall/precision or balanced-accuracy constraints
- calibration requirements if probabilities are consumed
- maximum inference latency and memory
- robustness tests for missing, shifted and out-of-range features
- acceptable subgroup performance gaps
- exact conditions for test-set release and model promotion

Do not select the model based on held-out test results. The test set is a final audit of a candidate already frozen using validation data.

## Failure modes to test

1. Scaler is accidentally fitted on validation/test data.
2. Candidate requests a feature not present in the versioned schema.
3. SVM receives unscaled features.
4. Worker retries a completed job and duplicates an artifact.
5. LLM attempts to read test labels or change split membership.
6. Candidate exceeds memory, time or inference-latency budget.
7. Probability-dependent policy consumes uncalibrated SVM margins.
8. A model is promoted without a reviewer decision and artifact digest.

## Acceptance criteria

A reusable implementation should demonstrate:
- deterministic split and candidate configuration manifests
- schema validation and leakage-safe pipelines
- validation-only hyperparameter selection
- isolated frozen-candidate test evaluation
- reproducible metrics and artifact lineage
- permission-denial tests for protected test labels
- idempotent retries and resource-limit enforcement
- traces, audit records and per-success cost/latency reporting
