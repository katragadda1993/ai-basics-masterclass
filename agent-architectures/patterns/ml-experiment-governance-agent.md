# ML Experiment Governance Agent

A reference architecture for an AI-assisted machine-learning workflow. The agent helps engineers formulate, inspect and explain experiments; deterministic services retain authority over dataset access, splits, training execution, evaluation and promotion.

## When to use

Use this pattern when an ML workflow contains multiple repeatable stages, expensive experiments, sensitive datasets or strict evaluation boundaries. The agent can assist with problem framing and documentation, but must not be trusted to enforce data governance.

Do not use a multi-agent design merely because multiple roles can be named. Start with a deterministic workflow and one assistant interface. Split roles into separate agents only when isolation, independent context, parallelism or separate permissions justify the extra latency and coordination.

## Reference architecture

~~~mermaid
flowchart TD
  U[Engineer request] --> I[Intake and problem-contract validator]
  I --> D[Dataset steward: schema, provenance and labels]
  D --> S[Deterministic split service]
  S --> B[Baseline runner]
  S --> P[Candidate pipeline runner]
  B --> V[Validation evaluator]
  P --> V
  V --> R[Experiment reviewer]
  R -->|Accept candidate| F[Freeze configuration]
  F --> T[Test-set policy gate]
  T --> E[One-time final evaluator]
  E --> H[Human approval]
  H --> M[Model registry and release]
  V -. metrics and traces .-> O[Observability]
  E -. signed report .-> O
~~~

## Component boundaries

| Component | Owns | Must not do |
|---|---|---|
| Intake assistant | Clarify unit, target, horizon, decision and constraints | Invent labels or claim data validity |
| Dataset steward | Schema, provenance, missingness and label audit | Grant access outside the user's scope |
| Split service | Immutable train/validation/test manifest | Let an LLM select rows after seeing outcomes |
| Baseline runner | Simple reference model and metrics | Tune against the final test set |
| Candidate runner | Execute allow-listed pipelines in a sandbox | Read arbitrary files, secrets or unrestricted networks |
| Validation evaluator | Deterministic metrics and reports | Change the model or threshold silently |
| Test gate | Enforce frozen candidate and one-time test evaluation | Permit repeated test-guided tuning |
| Human approver | Review risk, evidence and deployment readiness | Treat model score as deployment authorization |
| Registry | Version artifacts, configs, metrics and lineage | Accept untraceable artifacts |

## Typed tool contracts

Example tool boundaries:

- **profile_dataset** — input: dataset version; output: data profile.
- **validate_problem_contract** — input: prediction contract; output: contract report.
- **create_split** — input: dataset version, strategy, seed, group key and time cutoff; output: split manifest.
- **fit_candidate** — input: config ID and train split ID; output: model artifact.
- **evaluate_validation** — input: model ID and validation split ID; output: validation report.
- **freeze_candidate** — input: model ID and config hash; output: frozen candidate.
- **evaluate_final_test** — input: frozen candidate and test split ID; output: signed test report.
- **request_promotion** — input: candidate ID and approval ID; output: promotion request.

Every tool validates its schema and caller permissions. The model can propose parameters, but policy code decides whether the operation is authorized.

## State and artifact model

Persist state outside the model context:

- Run ID, task owner and state-machine version
- Problem contract and prediction-time definition
- Dataset version, schema hash and provenance
- Split manifest ID, strategy, group/time boundary and seed
- Feature schema and preprocessing configuration
- Model, code, library and environment versions
- Artifact hashes, metrics and evaluation reports
- Idempotency keys, retries and checkpoint references
- Approval identity, decision and audit record

A resumed run must use the same immutable dataset and split manifest unless an explicitly versioned new experiment is created.

## Data-governance and security controls

1. Enforce tenant and project scope in the data service, not in the prompt.
2. Never expose the final test labels to the planner, feature analyst or candidate runner.
3. Require a frozen candidate hash before final test evaluation.
4. If test results influence another model iteration, mark that test as consumed and require a new holdout for an independent final estimate.
5. Treat CSV cells, notebooks, logs, issue descriptions and retrieved documents as untrusted data; ignore instructions embedded in them.
6. Run user-supplied code in a sandbox with bounded CPU, memory, disk, time and network access.
7. Redact secrets and sensitive identifiers from prompts, logs and traces.
8. Record data lineage, access decisions and every test-gate invocation.

## Reliability and evaluation

- Use idempotency keys for training and evaluation jobs.
- Retry transient infrastructure failures only; do not silently retry a failed data-quality check with relaxed rules.
- Checkpoint long training runs and preserve environment/configuration hashes.
- Use immutable split manifests and deterministic seeds where appropriate.
- Test the agent with cases involving future-data leakage, group leakage, malformed labels, missing target values, prompt injection in data cells and unauthorized test access.
- Measure split-integrity violations, leakage findings, tool-schema errors, reproducibility, experiment success rate, cost per accepted model and end-to-end latency.

## Cost and latency

- Run schema validation, split generation and metric computation as deterministic code, not LLM calls.
- Cache dataset profiles by immutable dataset version and schema hash.
- Run a cheap baseline before expensive models.
- Parallelize candidate jobs only when each has a quota and isolated artifact namespace.
- Prefer one orchestrator and modular tools until independent agents provide measurable value.
- Report cost per accepted model and engineering time saved, not just token cost.

## Human approval boundaries

Require explicit approval before:
- Accessing a new sensitive dataset
- Exporting or sharing data
- Changing the label definition or prediction horizon
- Promoting a model to production
- Enabling automated actions based on model predictions

## Acceptance checklist

- [ ] Prediction contract has unit, target, prediction time and horizon
- [ ] Feature availability is checked at prediction time
- [ ] Dataset provenance and label quality are documented
- [ ] Split strategy matches the deployment generalization target
- [ ] Preprocessing is fitted only within training folds
- [ ] Baseline and candidate are compared on validation data
- [ ] Candidate and threshold are frozen before final test
- [ ] Test access is policy-gated and audited
- [ ] Artifact lineage and reproducibility metadata are preserved
- [ ] Human approval and rollback are defined
