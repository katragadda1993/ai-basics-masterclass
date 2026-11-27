# Entropy-Aware Specialist Routing with Abstention

## Purpose

Use a classifier's predictive distribution to help route a task to a specialist, while retaining a safe abstention path. This pattern extends the probability-aware routing example from Day 005.

Entropy is a measure of predictive distribution spread. It is not a probability of correctness, a full out-of-distribution detector, or an authorization signal.

## Architecture

~~~mermaid
flowchart TD
  E[Event] --> V[Validate identity, schema and provenance]
  V --> C[Versioned classifier]
  C --> P[Calibrated probability vector]
  P --> U[Entropy, margin and OOD signals]
  U --> R[Deterministic routing policy]
  R -->|Clear low-risk route| S[Least-privilege specialist]
  R -->|Ambiguous or OOD| H[Human triage]
  R -->|Consequential action| A[Approval gate]
  S --> Q[Independent verifier]
  Q --> L[Audit log and evaluation]
  H --> L
  A --> L
~~~

## Decision signals

For class probabilities p:

- Predictive entropy: H(p) = -sum_i p_i log(p_i)
- Top-two margin: largest probability minus second-largest probability
- Calibration: agreement between predicted probabilities and empirical frequencies
- OOD signal: a separately evaluated indicator that input differs from development data

Thresholds must be selected using representative validation data and explicit operational costs. Do not copy arbitrary thresholds from a tutorial.

## Important distinctions

- Predictive entropy captures spread in the output distribution.
- Epistemic uncertainty concerns uncertainty in the model and may be estimated with ensembles or Bayesian approaches.
- Aleatoric uncertainty comes from inherent data ambiguity/noise.
- OOD detection asks whether the input differs from the training/development distribution.

A model may be confidently wrong on an OOD input. Low entropy alone must never force a route.

## Typed routing decision

~~~json
{
  "event_id": "stable-event-id",
  "case_id": "diagnostic-case-id",
  "model_version": "router-model-version",
  "feature_schema_version": "v1",
  "candidate_route": "dma",
  "probabilities": {
    "dma": 0.72,
    "pcie": 0.18,
    "audio": 0.10
  },
  "predictive_entropy": 0.0,
  "top_two_margin": 0.54,
  "ood_status": "not_detected",
  "decision": "route_read_only",
  "evidence_refs": ["trace-123"]
}
~~~

The example values are illustrative. Recompute entropy and margin from the supplied probability vector in the implementation; do not trust model-supplied derived values. Validate event identity, schema and model version.

## Policy sequence

1. Validate source identity, event schema, tenant and evidence provenance.
2. Obtain probabilities from a versioned classifier.
3. Recompute entropy and top-two margin in trusted code.
4. Evaluate calibration and OOD indicators.
5. Apply a deterministic policy tuned on validation data.
6. Route only low-risk read-only tasks automatically.
7. Abstain for ambiguous, OOD or unsupported inputs.
8. Require independent authorization and human approval for consequential actions.
9. Persist the decision, evidence references, model version and reviewer identity.

## Reliability and security

- Use stable event IDs and idempotency keys to avoid duplicate execution.
- Persist case state and evidence references outside the model prompt.
- Bound model calls, retries, tokens, time and cost.
- Treat logs, telemetry, retrieved documents and tool outputs as untrusted data.
- Enforce permissions in an authorization service, never from model confidence.
- Keep secrets out of prompts and traces; redact sensitive fields.
- Use a verifier independent from the routing model for consequential findings.

## Evaluation

Measure:
- Route accuracy and per-class recall
- False route rate and cost-weighted error
- Abstention and human-escalation rate
- Calibration (reliability diagrams, Brier score, log loss)
- OOD detection performance
- End-to-end task success
- p50/p95 latency, token use and cost per successful task

Evaluate on temporally and operationally representative holdout data. Include class imbalance, correlated telemetry, distribution shift, malformed events and adversarial inputs.

## When to use it

Use this pattern when a classifier can narrow a large specialist set but ambiguous cases should be escalated. Prefer deterministic routing if rules are sufficient. Compare against a simple single-agent or fixed-workflow baseline before adding probabilistic routing.
