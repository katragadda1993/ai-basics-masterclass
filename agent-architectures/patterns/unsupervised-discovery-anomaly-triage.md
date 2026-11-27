# A18 — Unsupervised Discovery and Anomaly Triage Gate

## Purpose

Discover recurring structure and unusual observations in unlabeled telemetry without converting a cluster ID or density outlier into an unsupported fault diagnosis.

The pattern compares K-Means, agglomerative hierarchical clustering and DBSCAN. It extends the experiment-governance family (A14–A17) from supervised candidate selection to unsupervised discovery.

## Reference workflow

~~~mermaid
flowchart TD
  U[Engineer request] --> I[Typed discovery contract]
  I --> V[Schema, tenant and quality validation]
  V --> S[Versioned reference window and scaler]
  S --> R[Deterministic experiment workflow]
  R --> K[K-Means worker]
  R --> H[Agglomerative worker]
  R --> D[DBSCAN worker]
  K --> E[Metric, size and stability evaluator]
  H --> E
  D --> E
  E --> P[Cluster profiles and noise candidates]
  P --> X[Independent verifier]
  X --> A[Human engineering review]
  A --> M[Versioned discovery report]
  A --> Q[Follow-up test / triage queue]
~~~

## Typed request contract

~~~json
{
  "experiment_id": "generated-by-service",
  "dataset_version": "telemetry-window-sha256",
  "feature_schema_version": "soc-telemetry-v2",
  "scaler_version": "standard-scaler-v3",
  "algorithms": ["kmeans", "agglomerative", "dbscan"],
  "resource_budget": {
    "max_samples": 100000,
    "max_workers": 3,
    "deadline_seconds": 900
  },
  "tenant_scope": "resolved-from-authenticated-identity"
}
~~~

The tenant scope is derived from authenticated identity by the service; it is not trusted from model-generated text.

## Component responsibilities

1. **Intake agent:** convert the engineer's request into a typed discovery contract; ask for missing business constraints through the caller when interactive.
2. **Data validator:** verify schema, provenance, tenant boundary, missing values, units and sample limits.
3. **Feature service:** apply a versioned feature allow-list and fitted scaler; persist transformation metadata.
4. **Candidate runners:** execute bounded K-Means, agglomerative and DBSCAN configurations.
5. **Evaluator:** calculate appropriate internal metrics, cluster sizes, noise fraction, stability and resource use.
6. **Verifier:** check that the report's metrics correspond to the correct artifact, population and configuration; flag unsupported causal language.
7. **Human reviewer:** assign semantic interpretations, decide whether a follow-up experiment is warranted and approve any operational action.
8. **Registry:** persist experiment configuration, artifacts, reviewer decision and report digest.

## Orchestration patterns and trade-offs

| Pattern | Use | Guardrail |
|---|---|---|
| Deterministic workflow | Fixed discovery protocol | Version each stage and preserve the same reference snapshot |
| Single agent + typed tools | Natural-language experiment setup | Agent proposes; tool services enforce permissions and bounds |
| Router + specialists | Select a suitable algorithm family | Deterministic policy checks data size, geometry assumptions and supported metrics |
| Planner–executor–verifier | Open-ended discovery plan | Persist bounded candidate plan; independently verify outputs |
| Supervisor–worker | Parallel candidates on the same snapshot | Atomic CPU/memory/time budgets and deterministic aggregation |
| Event-driven | Batch-ready telemetry windows | Idempotency key, checkpoint, retry limit and dead-letter queue |
| Human-in-the-loop | Interpret clusters and authorize remediation | Reviewer identity, rationale and artifact digest are mandatory |

Start with a deterministic workflow. Introduce agents only where natural-language intake, planning or interpretation provides measurable value. Multi-agent execution is not automatically better.

## Candidate configuration example

~~~json
{
  "candidate_id": "dbscan-eps060-min8",
  "algorithm": "dbscan",
  "parameters": {
    "eps": 0.60,
    "min_samples": 8,
    "metric": "euclidean"
  },
  "feature_schema_version": "soc-telemetry-v2",
  "scaler_artifact": "sha256:...",
  "dataset_digest": "sha256:...",
  "code_version": "git-commit",
  "library_versions": {
    "scikit_learn": "pinned-version"
  }
}
~~~

Validate algorithm enums, parameter ranges, feature count, tenant scope, dataset digest and resource budgets before dispatch.

## Evaluation contract

Do not choose a clustering solution by a single score.

Record:
- K-Means inertia and silhouette over a declared K range
- minimum/maximum cluster size and size imbalance
- silhouette with the exact evaluation population recorded
- DBSCAN number of clusters, noise count/fraction and eps sensitivity
- stability across seeds, time windows or bootstrap samples
- external metrics (e.g. ARI) only when trusted labels exist and are kept out of fitting/selection
- downstream utility such as engineer triage time or follow-up-test yield
- CPU, peak memory, wall-clock latency and cost per completed experiment

For DBSCAN, if silhouette excludes noise points, label the population explicitly. Do not compare that score directly with a score computed over all observations for another algorithm.

## State and provenance

Persist:
- experiment ID and request ID
- immutable dataset/window digest and tenant scope
- feature schema, units, selected features and scaler artifact/version
- algorithm, parameters, random seed and code/library versions
- cluster labels, centroid/hierarchy artifacts where applicable
- cluster profiles and noise observation IDs
- metrics and exact evaluation populations
- stability results, warnings and resource measurements
- reviewer identity, semantic interpretation, rationale and approval
- report digest and follow-up ticket/test IDs

Cluster labels are arbitrary and may be permuted between runs. Never assume that cluster 0 in one experiment means the same thing as cluster 0 in another. Use permutation-invariant comparisons or explicit cluster alignment.

## Security and data boundaries

- Enforce tenant isolation and dataset access at the data service.
- Treat logs, telemetry strings and retrieved documents as untrusted content; do not execute embedded instructions.
- Keep raw sensitive telemetry out of prompts and traces unless explicitly authorized and redacted.
- Use allow-listed features, algorithms, metrics and bounded parameter ranges.
- Do not let the model mint authorization scopes or access a different tenant's data.
- Do not expose secrets in tool results, prompts, artifacts or observability systems.
- Require human approval for semantic labeling that triggers action, ticket creation with impact, threshold changes or remediation.

## Reliability and operations

- Bound sample count, feature count, number of candidates, worker concurrency, CPU, RAM and wall-clock time.
- Use idempotency keys based on experiment ID, dataset digest and candidate configuration.
- Persist checkpoints and cancellation state for long-running jobs.
- Retry transient infrastructure errors only; do not silently alter candidate parameters on retry.
- Make event consumers idempotent and route repeatedly failing jobs to a dead-letter queue.
- Record trace ID, experiment ID, tenant-safe identifier, data digest, scaler version, algorithm/configuration, duration, peak memory, sample count, cluster count, noise fraction and reviewer action.
- Monitor feature-distribution drift, centroid movement, cluster-size changes and noise-rate shifts.
- Measure cost per completed and human-reviewed discovery, not only model-call cost.

## Failure modes to test

1. A scaler is silently refitted on every live batch, making cluster coordinates incomparable.
2. A request tries to read a dataset belonging to another tenant.
3. An agent passes an unsupported algorithm or unbounded eps/K value.
4. DBSCAN labels are described as confirmed hardware faults.
5. A tiny K-Means cluster is accepted only because it marginally improves silhouette.
6. The report compares DBSCAN silhouette on non-noise points with K-Means silhouette on all points without disclosing the population.
7. A retry creates duplicate reports or tickets.
8. Cluster IDs are compared across runs without label alignment.
9. A worker exceeds memory or deadline budget.
10. A report contains unsupported causal claims or an unreviewed remediation instruction.

## Acceptance criteria

A reusable implementation should demonstrate:
- deterministic data and feature provenance
- versioned preprocessing fitted on a declared reference window
- bounded candidate execution and typed tool validation
- multiple metrics plus cluster-size and stability checks
- explicit DBSCAN noise semantics and population-aware reporting
- idempotent event handling and bounded retries
- tenant-isolation and prompt-injection tests
- independently verified report lineage
- human approval before operational actions
- per-experiment cost, latency and resource measurements

## Recommended baseline

Use a deterministic workflow with K-Means or DBSCAN as appropriate. Add a single agent for request translation and report explanation. Add specialists or supervisor/worker parallelism only when evaluation demonstrates a meaningful improvement in turnaround time or quality without weakening reproducibility or security.
