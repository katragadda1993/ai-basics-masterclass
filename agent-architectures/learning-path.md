# AI Agent Architecture Learning Path

This track is introduced after the foundations and LLM application concepts. Each module should include an architecture diagram, a runnable reference implementation, failure-mode tests, a security review and cost/latency measurements.

| Sequence | Pattern / engineering topic | Main design question | Expected artifact |
|---|---|---|---|
| A1 | Deterministic workflow + model steps | Which steps are fixed and which need model judgment? | State-machine workflow with typed model adapter |
| A2 | Single agent + typed tools | How does one agent safely call a small tool set? | Tool schemas, allow-list, bounded loop |
| A3 | Router + specialists | How should a task be routed to a domain expert? | Router with fallback and routing evaluation |
| A4 | Planner–executor–verifier | How do we plan, act and independently check? | Durable plan, executor and verifier |
| A5 | Supervisor + workers | Which subtasks can safely run in parallel? | Supervisor, worker contracts and aggregation |
| A6 | Event-driven agents | How do events start work without duplicate side effects? | Queue consumer, idempotency and dead-letter flow |
| A7 | Human-in-the-loop | Which actions require approval and how is approval audited? | Approval state, reviewer UI/API and audit trail |
| A8 | Embedding retrieval and specialist routing | How can vector similarity rank evidence or suggest a specialist without becoming an authorization decision? | Versioned embedding contract, tenant-scoped retrieval and evaluated router with fallback |\n| A9 | State, memory and context | What belongs in working state, episodic memory or retrieval? | Versioned state schema and retrieval policy |
| A10 | Security and tool boundaries | How are identity, permissions and untrusted content handled? | Threat model, tool policy and injection tests |
| A11 | Evaluation and observability | How do we know the agent is reliable and improving? | Golden set, traces, metrics and regression gate |
| A12 | Reliability and cost/latency | How do we bound failures and optimize the whole workflow? | Budgets, retries, checkpointing and performance report |

## Architecture selection rules

- Prefer a deterministic workflow when the sequence is known.
- Use a single agent before introducing multiple agents.
- Add specialists only when their responsibilities, context or permissions genuinely differ.
- Use planner–executor–verifier when work is multi-step and output correctness can be checked.
- Use parallel workers only for independent subtasks with explicit merge rules.
- Use event-driven execution for asynchronous work; make side effects idempotent.
- Put human approval before irreversible or high-impact operations.

## Required engineering contract for every pattern

Every implementation must document:
- Inputs, outputs and typed tool contracts
- Component ownership and state transitions
- Model and prompt versions
- Permission scopes and side-effect classification
- Timeouts, retries, budgets, cancellation and recovery
- Memory and data retention boundaries
- Threat model and prompt-injection tests
- Evaluation cases and measurable acceptance criteria
- Trace fields, metrics, redaction and audit events
- End-to-end latency and cost per successful task

Do not assume a multi-agent design is more capable. Compare it against a simpler baseline on the same evaluation set.
