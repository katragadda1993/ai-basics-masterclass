# Agent Specification: [Name]

## Purpose
What single responsibility does this agent own?

## Non-goals
What must it never do?

## Inputs
- Required fields:
- Optional fields:
- Trusted/untrusted sources:

## Outputs
- Output schema:
- Evidence/provenance required:
- Artifact format:

## Operating procedure
1. Validate the task and inputs.
2. Retrieve only authorized context.
3. Produce a structured plan or analysis.
4. Request tool authorization where needed.
5. Execute bounded actions.
6. Verify against acceptance criteria.
7. Return result, evidence, limitations and next steps.

## Tools
| Tool | Read/write | Permission scope | Timeout | Idempotent? |
|---|---|---|---|---|

## Policies and safety
- Allowed actions:
- Disallowed actions:
- Human approval triggers:
- Data handling and redaction:

## Reliability
- Maximum steps:
- Maximum tool calls:
- Token/cost budget:
- Retry policy:
- Timeout/cancellation:
- Recovery behavior:

## Evaluation
- Golden test cases:
- Success criteria:
- Safety tests:
- Regression tests:
- Quality/latency/cost thresholds:

## Observability
- Trace fields:
- Metrics:
- Audit events:
- Sensitive fields to redact:

## Ownership and versioning
- Owner:
- Version:
- Model/prompt version:
- Tool/API versions:
- Last evaluation date:
