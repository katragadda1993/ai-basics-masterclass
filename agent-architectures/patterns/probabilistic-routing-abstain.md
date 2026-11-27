# Pattern: Probabilistic Routing with an Abstain Path

Use this pattern when a request or event must be routed to one of several specialist workflows and a validated model can estimate class probabilities. The model proposes a route; deterministic policy and authorization decide what may happen.

## Architecture

\`\`\`mermaid
flowchart TD
  E[Event] --> V[Validate schema, identity and provenance]
  V --> F[Feature extraction]
  F --> M[Versioned classifier]
  M --> C[Calibration and distribution checks]
  C --> R[Routing policy: thresholds + abstain]
  R -->|Clear, low-risk case| S[Read-only specialist]
  R -->|Ambiguous or shifted data| H[Human triage]
  R -->|High-impact action| H
  P[Independent authorization policy] --> S
  P --> H
  S --> Q[Evidence verifier]
  H --> Q
  Q --> A[Audit record]
  A --> O[Evaluation and observability]
\`\`\`

## Separation of concerns

- **Classifier:** estimates class probabilities and exposes model/version metadata. It does not call tools.
- **Calibration layer:** checks whether probability estimates align with observed frequencies on representative held-out data.
- **Distribution monitor:** detects missing features, out-of-range values and potential drift. An out-of-distribution score is not a calibrated probability unless separately validated.
- **Router policy:** applies thresholds, tie margins and abstention. Thresholds are selected using validation data and operational costs.
- **Authorization:** independently verifies user identity, case scope, tool permissions and action class. Model confidence never grants permission.
- **Specialist:** receives a typed, minimal task and least-privilege tools.
- **Verifier:** checks evidence, required outputs and acceptance criteria.
- **Audit/telemetry:** records event ID, feature schema, model version, calibration version, route, reason, evidence references and approval.

## Typed route proposal

\`\`\`json
{
  "event_id": "unique-id",
  "case_id": "case-123",
  "feature_schema_version": "v1",
  "model_version": "router-1.0",
  "calibration_version": "cal-1.0",
  "candidate": "dma",
  "estimated_probability": 0.82,
  "evidence_refs": ["trace-123", "log-456"],
  "requested_mode": "read_only"
}
\`\`\`

Validate ranges, required fields, allowed candidate names and event identity. Treat the proposal as untrusted input even when it came from your own model service.

## Decision policy

Use an explicit abstain region rather than forcing every event into a class:
- Route only when the input passes schema and distribution checks, the selected probability exceeds a threshold validated for that class, and the margin over the runner-up is adequate.
- Abstain when confidence is low, classes are close, evidence is incomplete or distribution shift is suspected.
- Escalate high-impact or irreversible work for human review, regardless of confidence.
- Always run authorization independently before any tool invocation.

Thresholds must be tuned to the relative cost of false positives and false negatives, class prevalence, available human-review capacity and deployment conditions. Do not copy illustrative thresholds into production.

## State, memory and event reliability

Persist case state outside the model context:
- stable event and case IDs
- event timestamp and source
- feature and model versions
- evidence references and their hashes
- proposed route, policy decision and reason
- human approval identity and timestamp
- execution status and result

Use an idempotency key derived from the event identity and operation. Handle duplicate, delayed and out-of-order events. Use bounded retries for transient failures, durable checkpoints and a dead-letter path. Never let a retry duplicate a consequential side effect.

## Evaluation

Track:
- per-class precision, recall and confusion matrix
- false-negative rate for safety-critical classes
- calibration curve, Brier score and log loss
- abstention/coverage rate and human override rate
- performance by workload, device, firmware and environment
- latency, model/tool cost, queue delay and failure rate
- drift in features, prevalence and label quality

Keep a time-aware, leakage-free test set. Evaluate after model, prompt, feature, firmware or data-pipeline changes. A classifier can retain ranking performance while its calibration degrades under prevalence shift.

## Operational trade-offs

- A deterministic first-stage filter is cheap and explainable; it can miss novel cases.
- A compact classifier is fast; it needs representative training data and calibration.
- An LLM can summarize unstructured evidence; it is slower, costlier and may hallucinate.
- Human review increases latency but is appropriate for ambiguity and high-impact decisions.
- A multi-agent design is not automatically better than a single workflow. Add specialists only when clear isolation, parallelism or independent verification justifies the coordination overhead.

## Security checklist

- Enforce authorization in code, not in prompts.
- Treat logs, telemetry, retrieved documents and model outputs as untrusted.
- Minimize and redact sensitive features in prompts and traces.
- Keep model services read-only where possible.
- Require explicit approval for irreversible operations.
- Record decisions and evidence without retaining unnecessary raw data.
