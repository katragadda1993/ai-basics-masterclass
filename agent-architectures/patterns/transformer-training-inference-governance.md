# A26 — Transformer Training and Inference Governance Gate

**Status:** Reference architecture  
**Track:** Reusable agent architectures  
**Companion lesson:** [Session 020 — Attention Mechanism and Transformers](../../daily-lessons/day-020/README.md)

## Problem

Transformer experiments and inference pipelines combine tokenization, positional information, attention masks, model weights, context limits and (for autoregressive generation) KV-cache state. A subtle contract mismatch can cause leakage, incorrect generation, cross-tenant data exposure or excessive cost.

An LLM assistant can help plan experiments and explain traces, but deterministic services must own tokenization contracts, mask construction, training, evaluation, cache isolation and release policy.

## Design goals

- Make tokenizer, vocabulary, special-token and tensor contracts explicit.
- Verify causal and padding masks with executable tests.
- Separate encoder, decoder-only and encoder-decoder model paths.
- Keep user/conversation memory separate from numerical KV-cache state.
- Bound context length, cache memory, concurrency, output tokens and retries.
- Preserve dataset and checkpoint lineage.
- Require independent evaluation and human approval for production release.

## Reference workflow

~~~mermaid
flowchart TD
    A[Engineer objective] --> B[Read-only intake agent]
    B --> C[Typed token and sequence contract]
    C --> D[Dataset, tokenizer and split validator]
    D --> E[Deterministic representation pipeline]
    E --> F{Task router}
    F --> G[Encoder experiment worker]
    F --> H[Decoder-only worker]
    F --> I[Encoder-decoder worker]
    G --> J[Deterministic training runtime]
    H --> J
    I --> J
    J --> K[Mask, gradient and artifact verifier]
    K --> L[Validation and robustness evaluator]
    L --> M{Acceptance policy}
    M -->|Reject| N[Bounded retry or report]
    M -->|Accept| O[Frozen candidate]
    O --> P[Protected test gate]
    P --> Q[Human approval]
    Q --> R[Model registry]
    R --> S[Inference service]
    S --> T[Context and KV-cache manager]
~~~

## Component boundaries

| Component | Owns | Must not own |
|---|---|---|
| Intake agent | Clarify task, summarize requirements, propose experiment plan | Data access authorization or release approval |
| Tokenizer service | Tokenizer version, vocabulary hash, special tokens, encode/decode | Model release decision |
| Dataset validator | Schema, split manifest, provenance, leakage checks | Editing protected test labels |
| Mask builder | Causal, padding and cross-attention masks with explicit semantics | Policy decisions based on model output |
| Training runtime | Tensor execution, gradients, optimizer, checkpointing | Permission escalation |
| Evaluator | Validation metrics, robustness and acceptance evidence | Modifying the frozen candidate |
| Inference runtime | Model execution, request limits and KV-cache lifecycle | Cross-tenant cache reuse |
| Human reviewer | Approve consequential release or policy change | Bypassing recorded evidence |

## Typed contracts

A production implementation should define schemas equivalent to:

~~~json
{
  "experiment_id": "exp-unique-id",
  "task": "decoder_only_language_model",
  "tokenizer": {
    "name": "tokenizer-name",
    "revision": "immutable-revision",
    "vocabulary_hash": "sha256:...",
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0
  },
  "sequence": {
    "max_length": 2048,
    "truncation": "right",
    "padding": "right",
    "position_method": "declared-method"
  },
  "attention": {
    "causal": true,
    "mask_semantics": "true_means_allowed",
    "padding_mask_required": true
  },
  "model": {
    "d_model": 512,
    "num_heads": 8,
    "num_layers": 12
  },
  "data": {
    "dataset_revision": "immutable-revision",
    "split_manifest_hash": "sha256:..."
  },
  "budget": {
    "max_context_tokens": 2048,
    "max_output_tokens": 256,
    "deadline_seconds": 3600
  }
}
~~~

The values are illustrative, not a recommended configuration. Validate dimensions (for example, d_model divisible by num_heads), allowed task types, tokenizer compatibility and resource limits before scheduling work.

## Agent pattern mapping

| Pattern | Use in this architecture | Trade-off |
|---|---|---|
| Deterministic workflow | Validate → tokenize → mask → train → evaluate → release | Predictable and auditable; less flexible for open-ended exploration |
| Single agent + typed tools | Read-only metric, tokenizer and trace inspection | Low coordination overhead; limited parallelism |
| Router-specialist | Select encoder, decoder-only, encoder-decoder, audio or telemetry path | Better component ownership; routing errors need fallback |
| Planner–executor–verifier | Plan bounded experiments, execute and independently verify masks/metrics | Better multi-step control; more latency and state |
| Supervisor–worker | Parallel seeds/configurations/dataset cohorts | Higher throughput; requires resource limits and merge policy |
| Event-driven | Queue training, checkpoint, evaluation and inference events | Durable asynchronous work; duplicate delivery must be idempotent |
| Human-in-the-loop | Approve release, context policy or consequential diagnostic use | Safer governance; adds review latency |

Do not add multiple agents when a deterministic script or one tool-using agent is sufficient.

## State and memory boundaries

Keep these separate:

1. **Experiment state:** immutable configuration, job status, retries, checkpoint references and metric history.
2. **Tokenizer/model state:** exact revisions, vocabulary hash, positional method, weights and architecture configuration.
3. **Conversation memory:** user-visible conversation context and retrieval metadata.
4. **KV cache:** numerical key/value tensors for one inference request or explicitly scoped generation session.

KV-cache identity should include at least tenant/security principal, session or request ID, model revision, adapter revision, tokenizer/position contract and cache format. Expire or invalidate it on session end, model swap, authorization change or incompatible configuration. Never share cache across tenants.

## Security and safety controls

- Treat retrieved documents, logs, code and user prompts as untrusted data.
- Keep tool authorization outside the model; a model-generated tool call is a request, not permission.
- Do not expose arbitrary shell, filesystem or network tools to the training agent.
- Protect dataset splits and test labels.
- Validate mask semantics and test for future-token leakage.
- Prevent cross-tenant context, cache and trace access.
- Redact secrets and sensitive prompt content from logs.
- Require explicit approval for production promotion and consequential actions.
- Load only trusted model/checkpoint artifacts and verify hashes.

## Evaluation contract

Minimum checks:

- Tokenizer encode/decode and special-token round trips.
- Tensor shape, dtype, device and context-limit validation.
- Causal future-perturbation test.
- Padding invariance test: adding padded positions must not change valid-token outputs when masks are correct.
- Cross-attention source/target shape checks.
- Gradient finite-value and gradient-check tests on small cases.
- Validation-only model selection and protected final test evaluation.
- Long-context, truncation and adversarial-input tests.
- Task metrics appropriate to the use case; perplexity alone is not a complete quality or safety metric.

## Reliability and observability

Record:
- request/experiment ID, model/tokenizer/config revisions
- sequence length, padding ratio and truncation count
- mask type and mask-contract version
- training step, loss, gradient norm and checkpoint ID
- prompt/context token counts and output token counts
- time-to-first-token, inter-token latency, tokens/sec and p50/p95 latency
- KV-cache occupancy, GPU memory and queue age
- retry/cancellation reason and policy decision

Use idempotency keys for job submissions and event consumers. Checkpoint long-running jobs. Bound retries, context, concurrency, output tokens, deadlines and GPU allocation.

## Cost and latency controls

- Use batching when throughput matters and queue latency permits.
- Use KV caching for autoregressive generation; account for memory growth with context and batch size.
- Evaluate efficient SDPA/FlashAttention implementations on actual hardware.
- Consider GQA/MQA or sliding-window attention when model quality and architecture permit.
- Route simple deterministic tasks to tools or smaller models.
- Do not call an LLM for every token or every telemetry event when deterministic processing suffices.

## Acceptance and release

1. Typed contracts validate.
2. Tokenizer/model/data revisions are immutable.
3. Mask tests pass.
4. Training and evaluation complete within declared budgets.
5. Independent verifier reproduces key metrics and artifact hashes.
6. Protected test evaluation is performed once on the frozen candidate.
7. Human reviewer approves production promotion.
8. Rollback artifact and cache invalidation plan are recorded.

## Example implementation checklist

- [ ] Tokenizer and vocabulary revision pinned
- [ ] BOS/EOS/PAD IDs validated
- [ ] Sequence length and truncation policy declared
- [ ] Causal and padding mask semantics explicit
- [ ] Future-token leakage test passes
- [ ] Padding invariance test passes
- [ ] Model and positional method versioned together
- [ ] Dataset split manifest immutable
- [ ] KV cache scoped and invalidated correctly
- [ ] Resource, token, latency and retry budgets enforced
- [ ] Evaluation and artifact lineage independently verified
- [ ] Human approval and rollback plan recorded
