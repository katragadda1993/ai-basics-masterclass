# A15 — Objective-Aware Model Selection Gate

## Purpose

Route a validated supervised-learning problem to a regression or classification workflow, enforce objective-specific evaluation, and protect the final test set and model-promotion boundary.

This is a reference architecture, not an autonomous model-release authority.

## Architecture

~~~mermaid
flowchart TD
  U[Engineer request] --> I[Intake agent: clarify target and decision]
  I --> C[Typed problem contract validator]
  C --> R{Deterministic task router}
  R -->|Numeric target| RG[Regression workflow]
  R -->|Categorical target| CL[Classification workflow]
  RG --> RB[Baseline and candidate runner]
  CL --> CB[Baseline and candidate runner]
  RB --> EV[Objective-specific evaluator]
  CB --> EV
  EV --> VG[Validation policy gate]
  VG -->|Accept candidate| F[Freeze model, features and threshold]
  F --> TG[Untouched test-set access gate]
  TG --> TE[Final evaluator]
  TE --> H[Human approval]
  H --> MR[Model registry / release]
  EV -. traces .-> O[Metrics, audit and observability]
  TE -. signed report .-> O
~~~

## Component boundaries

1. **Intake agent:** helps translate a request into a prediction contract.
2. **Contract validator:** deterministic validation of target type, unit, horizon, feature availability and metric requirements.
3. **Task router:** routes numeric targets to regression and categorical targets to classification.
4. **Candidate runner:** executes allow-listed pipelines in an isolated environment.
5. **Evaluator:** computes task-specific metrics from versioned predictions and labels.
6. **Validation gate:** compares candidates against predeclared acceptance criteria.
7. **Test gate:** permits final evaluation only for a frozen candidate and immutable split manifest.
8. **Human approver:** authorizes promotion and records rationale.

## Typed tool contracts

| Tool | Input | Output |
|---|---|---|
| validate_problem_contract | Target type, unit, horizon, features | Contract report |
| run_regression_candidate | Dataset/split IDs, model config | Model artifact |
| run_classifier_candidate | Dataset/split IDs, model config | Model artifact |
| evaluate_candidate | Artifact ID, validation split ID, metric contract | Validation report |
| select_threshold | Validation probabilities, labels, operating objective | Threshold proposal |
| freeze_candidate | Artifact, feature schema, threshold, config hash | Immutable candidate ID |
| evaluate_final_test | Frozen candidate ID, test split ID | Final test report |
| request_promotion | Candidate ID, approval record | Promotion request |

Every tool validates its schema and caller permission. The model may propose a configuration; deterministic policy decides whether the operation is authorized.

## State and memory

Persist the experiment outside the model context:
- Task ID and state-machine status
- Prediction contract and metric contract
- Dataset and immutable split-manifest IDs
- Feature schema and preprocessing version
- Candidate model/configuration hashes
- Validation report and frozen threshold
- Final test report and approval reference

Do not use conversational memory as the source of truth for experiment state.

## Reliability and security

- The LLM cannot alter the test split, inspect test labels during development or promote a model.
- Use least-privilege dataset scopes and sandboxed training workers.
- Treat CSV cells, notebooks, logs and retrieved documents as untrusted content.
- Make long-running training event-driven with job IDs, idempotency keys, durable status and bounded retries.
- Propagate cancellation and deadlines; checkpoint expensive jobs.
- Verify artifact hashes and metric calculations independently.
- Require human approval before consequential model release.

## Architecture trade-offs

- **Deterministic workflow:** best for known training/evaluation steps and reproducibility.
- **Single agent + tools:** enough for requirement clarification and experiment assistance.
- **Router + specialists:** useful when regression, classification, time-series and ranking require different domain expertise; avoid unnecessary agents.
- **Planner–executor–verifier:** useful for exploratory analysis; the verifier checks outputs against explicit criteria.
- **Supervisor–worker:** parallelize independent candidates only when they share the same immutable data split and resource budget.
- **Event-driven:** suitable for long-running training and asynchronous evaluation.
- **Human-in-the-loop:** required for model promotion and exceptions to policy.

## Observability and evaluation

Trace task ID, dataset/model versions, split ID, tool latency, CPU/memory use, cost, retries and outcome. Redact secrets and sensitive row-level data.

Measure:
- Contract-validation accuracy
- Reproducibility of splits and metrics
- Candidate improvement over baseline
- Regression MAE/RMSE/R² or classification PR-AUC/recall/precision/calibration as appropriate
- Leakage and policy-violation test results
- Cost per accepted model and time to trustworthy result

A high metric is not sufficient for release. Require representative data, segment analysis, calibration when needed, documented limitations and an approval record.

## Acceptance tests

- A numeric target cannot silently enter a classifier workflow.
- A categorical target cannot silently enter a regression workflow.
- The same split manifest is used for all candidate comparisons.
- Threshold selection can read validation labels but not test labels.
- The test gate rejects unfrozen or modified candidates.
- Tool calls without the required permission are rejected by code.
- Repeated training-job events do not create duplicate side effects.
- A promotion request without a human approval record is rejected.

## Further reading

- [scikit-learn Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [scikit-learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [scikit-learn Common Pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
