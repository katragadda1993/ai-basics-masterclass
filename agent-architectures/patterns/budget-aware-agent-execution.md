# Pattern: Budget-Aware Agent Execution Controller

**Related architecture module:** A13 — Reliability and cost/latency  
**Companion lesson:** [Day 007 — Optimization Algorithms](../../daily-lessons/day-007/README.md)

This pattern controls resources consumed by an agentic workflow. It is inspired by controlled resource allocation, but it is not a neural-network optimizer: it does not update model weights or use gradients.

## When to use

Use when an agent task has explicit token, tool-call, wall-clock, monetary or parallelism limits. It is useful for multi-step research, code-generation, diagnostic and supervisor/worker workflows.

## Architecture

~~~mermaid
flowchart TD
  R[Request] --> I[Identity, goal and risk validation]
  I --> B[Budget envelope service]
  B --> P[Planner proposes bounded steps]
  P --> G[Policy + budget reservation gate]
  G -->|Allowed| X[Executor: approved model/tool]
  G -->|Denied / exhausted| H[Stop, fallback or human review]
  X --> V[Independent verifier]
  V -->|Accept| D[Deliver artifact]
  V -->|Repair within budget| P
  V -->|Exhausted / unsafe| H
  X -. usage .-> O[Telemetry and cost ledger]
  V -. usage .-> O
  O --> S[Persistent task state]
~~~

## Component contracts

### 1. Budget envelope

Create a server-owned immutable task envelope:

~~~json
{
  "task_id": "stable-id",
  "max_wall_time_ms": 120000,
  "max_model_tokens": 12000,
  "max_tool_calls": 20,
  "max_parallel_workers": 3,
  "max_cost_microunits": 500000,
  "deadline": "timestamp",
  "policy_version": "budget-policy-v1"
}
~~~

The example values are illustrative. Monetary units and currency must be defined by the application. Do not accept a model-generated budget as authoritative.

### 2. Planner

The planner proposes steps, dependencies, estimated resource needs and expected outputs. It cannot change the task envelope or authorize its own tool calls.

### 3. Reservation and policy gate

Before starting work:
- Atomically reserve tokens, tool-call slots, parallel-worker slots and expected spend.
- Check user identity, data scope, tool permissions and side-effect risk.
- Reject steps that exceed remaining budget.
- Require approval for irreversible or high-impact actions.

Reservations prevent concurrent workers from each assuming the same remaining budget is available.

### 4. Executor

The executor invokes typed, allow-listed tools or approved model endpoints. It records actual usage, timeout, result status and idempotency key. Retries are bounded and reserved against the same task budget.

### 5. Verifier

The verifier checks acceptance criteria and evidence independently of the planner. A repair loop is allowed only while budget and deadline remain and policy still permits the action.

### 6. Persistent state and ledger

Persist task state, budget reservations, actual usage, model/tool versions, artifact references and approval events. Use a transactional ledger or equivalent atomic accounting for concurrent execution. Do not rely on LLM context as the source of truth.

## Execution policy

1. Validate the task and classify its risk.
2. Allocate a hard resource envelope.
3. Prefer a deterministic workflow when steps are known.
4. Select the least expensive approved model/tool that meets quality requirements.
5. Escalate to a stronger model or specialist only when policy and remaining budget allow it.
6. Verify the result before delivery.
7. Stop, return a partial result or request human review when budget is exhausted.

## Reliability and security

- Use stable task IDs and idempotency keys for side-effecting tools.
- Use atomic budget reservations for parallel workers.
- Bound retries and use backoff only for transient failures.
- Support cancellation, checkpoints and deadline propagation.
- Enforce permissions in code, not prompt instructions.
- Treat retrieved content and tool output as untrusted.
- Redact secrets and sensitive data from traces.
- Never allow an agent to raise its own budget or grant itself additional permissions.
- Require human approval for irreversible operations.

## Evaluation

Track:
- Success rate against a fixed task set
- Cost per successful task
- p50/p95 end-to-end latency
- Token and tool-call usage distributions
- Retry and timeout rate
- Escalation and abstention rate
- Quality loss when using cheaper models
- Budget overruns (target: zero)
- Duplicate side effects (target: zero)

Compare deterministic, single-agent and multi-agent baselines on the same tasks. Do not assume adding agents improves quality or efficiency.

## Trade-offs

| Decision | Benefit | Cost / risk |
|---|---|---|
| Hard budgets | Prevent runaway spend and latency | May stop useful work early |
| Cheap-first escalation | Lower average cost | Extra routing hop; needs quality gates |
| Parallel workers | Lower elapsed time for independent tasks | Budget contention and coordination |
| Strict verifier | Better output reliability | Additional latency and model/tool cost |
| Human approval | Controls high-impact actions | Human response delay |

## Acceptance checklist

- [ ] Budget limits are server-owned and cannot be raised by the model.
- [ ] Reservations are atomic under concurrency.
- [ ] Every tool is typed, allow-listed and permission-checked.
- [ ] Retries, workers and repair loops consume the same bounded budget.
- [ ] Deadline and cancellation propagate to child tasks.
- [ ] High-impact actions have an approval gate.
- [ ] Traces include model/tool version, cost, latency and decision reason.
- [ ] Evaluation reports cost per successful task and quality, not just token count.
