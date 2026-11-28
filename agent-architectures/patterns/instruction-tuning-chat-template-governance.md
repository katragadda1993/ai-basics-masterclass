# A29 — Instruction-Tuning and Chat-Template Governance Gate

**Status:** reusable reference architecture  
**Related lesson:** [Session 023 — Instruction Tuning, Supervised Fine-Tuning and Chat Templates](../../daily-lessons/day-023/README.md)  
**Scope:** curated conversation data, model-specific serialization, assistant-target masking, SFT/LoRA training, regression evaluation and model promotion.

## Problem

A chat model is not just a base checkpoint plus a list of prompt/answer strings. It is a coupled contract among the base model, tokenizer, special-token IDs, chat template, message roles, end-of-message/turn semantics, assistant-generation mask, truncation/padding policy, training objective and evaluation suite.

A mismatch can silently train the wrong positions, make generation start in the wrong role, or cause the model to emit malformed tool calls. An agent may assist with data review and experiment planning, but deterministic services must own serialization, masks, training, test access and release.

## Architecture

~~~mermaid
flowchart TD
    A[SFT request] --> B[Read-only intake agent]
    B --> C[Typed SFT run contract]
    C --> D[Dataset policy and split validator]
    D --> E[Versioned tokenizer and chat template]
    E --> F[Tokenization and assistant-mask verifier]
    F --> G[Deterministic SFT trainer]
    G --> H[Checkpoint and adapter artifacts]
    H --> I[Independent evaluator]
    I --> J[Instruction and regression gates]
    J --> K{Acceptance policy}
    K -->|Reject| L[Bounded correction or stop]
    K -->|Accept| M[Frozen candidate]
    M --> N[Protected test evaluation]
    N --> O[Human approval]
    O --> P[Model registry]
~~~

## Component contracts

| Component | Owns | Must not own |
|---|---|---|
| Dataset service | Source policy, quality review, deduplication and split manifests | Model promotion |
| Chat-template service | Role serialization, control-token IDs and generation prefix | Runtime authorization |
| Tokenizer | Normalization, vocabulary, special IDs and tokenization | Dataset access policy |
| Mask builder | Assistant/tool-call target spans, shift alignment and padding mask | Editing source labels |
| Trainer | Full/LoRA updates, optimizer and checkpointing | Protected test labels |
| Evaluator | Instruction, schema, tool-call, safety and regression metrics | Training data mutation |
| Agent assistant | Explain reports and propose bounded experiments | Arbitrary code execution or release authority |

## Seven reusable agent patterns

| Pattern | Application | Trade-off |
|---|---|---|
| Deterministic workflow | Validate → template → tokenize → mask → train → evaluate | Reproducible; less flexible for exploration |
| Single agent + typed tools | Inspect manifests, masks and evaluation reports | Low overhead; limited specialization |
| Router-specialist | Route data, template, training, evaluation and safety questions | Focused expertise; needs abstain/fallback |
| Planner–executor–verifier | Plan bounded SFT experiments and independently verify masks/results | Strong control; adds latency and state |
| Supervisor-worker | Parallel seeds, LoRA ranks or learning-rate trials | Faster exploration; higher compute and merge complexity |
| Event-driven | Queue runs, checkpoints and evaluation events | Scalable; needs durable state and idempotency |
| Human-in-the-loop | Approve dataset exceptions and model promotion | Safer; adds review latency |

Prefer the simplest pattern that meets the requirement. Parallel workers require immutable manifests, isolated checkpoints and explicit candidate-selection rules.

## Typed tool contract

Illustrative Python schema:

~~~python
from dataclasses import dataclass

@dataclass(frozen=True)
class SFTRequest:
    base_model_revision: str
    tokenizer_revision: str
    chat_template_sha256: str
    dataset_manifest_sha256: str
    split_manifest_sha256: str
    max_sequence_length: int
    learning_rate: float
    epochs: int
    adapter_mode: str  # "full" or "lora"
    lora_rank: int | None
    seed: int
~~~

Validate model/tokenizer/template compatibility, numeric bounds, caller identity, resource quotas and dataset permissions in deterministic services. Model-generated plans are untrusted data, not executable authority.

## State and memory boundaries

Persist:
- Dataset/source/license and split manifests.
- Message schema and chat-template hash.
- Tokenizer artifact hash, normalization and special-token IDs.
- Assistant-mask generation rules and mask-test results.
- Base model revision and adapter configuration.
- Optimizer, scheduler, scaler, RNG, sampler and step state.
- Checkpoint/adapter hashes, evaluation-suite revision and approval record.

Keep three state classes separate:
1. **Agent conversation state:** goals, tool results and approval workflow.
2. **Training state:** model/optimizer/scheduler/scaler/RNG/sampler checkpoint.
3. **Serving conversation state:** conversation history and request-scoped KV cache.

A template or tokenizer change is a compatibility change requiring re-tokenization and evaluation. Scope serving state by tenant, authorized session and model revision.

## Security and governance

- Treat dataset content as untrusted text, not agent instructions.
- Apply privacy, licensing, retention and consent checks before training.
- Keep protected test labels inaccessible to planners and workers.
- Sandbox preprocessing and training; deny arbitrary shell/network access by default.
- Do not train on secrets, credentials or unauthorized internal logs.
- Validate tool-call schemas and enforce permissions outside the model.
- Hash/sign artifacts and load only trusted checkpoints.
- Require human approval for production promotion.
- Keep hardware writes, resets, firmware flashing and other high-impact actions outside LLM authority.

## Evaluation gates

Minimum gates:
- Role/template round-trip and special-token tests.
- Assistant-mask coverage and shifted-target alignment.
- Nonzero supervised-token count for every batch.
- Padding, truncation and packed-sequence boundary correctness.
- Held-out assistant-token NLL plus task-specific instruction metrics.
- Structured-output schema and tool-call contract tests.
- Safety, abstention and prompt-injection regression tests.
- General-capability regression suite before and after tuning.
- Checkpoint/adapter restore test.
- Protected final test after candidate freeze.

Loss reduction is not sufficient evidence of instruction following. Evaluate exact match only for deterministic tasks; use rubrics and expert review for open-ended technical answers.

## Reliability

- Immutable run IDs and dataset/template/model manifests.
- Idempotent job submission and artifact writes.
- Atomic checkpoints and restore tests.
- Bounded retries for transient infrastructure failures only.
- Deadlines, cancellation and retry budgets.
- Durable event offsets and deduplication keys.
- Frozen validation candidate before protected test evaluation.
- Auditable human approval and registry promotion.

A tokenizer mismatch, zero-supervised-token batch or invalid assistant mask is a deterministic configuration/data error, not a transient retry.

## Observability, cost and latency

Track:
- Examples and assistant target tokens per second.
- Prompt/context tokens versus supervised response tokens.
- Masked-token ratio and truncation rate.
- Train/validation loss and task-specific success.
- Learning rate, gradient norm and skipped optimizer steps.
- Trainable parameter count, peak memory and GPU utilization.
- Checkpoint duration/size and resume count.
- p50/p95 training-step and inference latency.
- Adapter size and cost per successful task.

LoRA often reduces trainable-state memory and adapter storage, but benchmark actual throughput. SFT does not necessarily make inference faster: the base model still performs its forward pass.

## Acceptance criteria

A release candidate is eligible for human review only if:
1. Dataset, split, tokenizer, template and base-model manifests are immutable and compatible.
2. Role, special-token, assistant-mask and shifted-target tests pass.
3. Every training batch has a valid supervised-token count.
4. Held-out assistant loss and task-specific metrics meet predeclared thresholds.
5. General-capability and safety regression tests pass.
6. Checkpoint/adapter restoration reproduces outputs within declared tolerance.
7. Artifact hashes, training configuration and evaluation provenance are complete.
8. Resource, cost and latency budgets are met.

**Release rule:** the assistant may explain runs and propose bounded experiments. Deterministic services own data access, serialization, masks, training and evaluation. Independent verification and human approval own production promotion.
