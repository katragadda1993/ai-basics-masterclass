# Session 023 — Instruction Tuning, Supervised Fine-Tuning and Chat Templates

**Phase 5 · Transformers and Generative AI · 60-minute lesson**

> **Continuity:** Session 020 derived Transformer attention; Session 021 covered tokenization and embeddings; Session 022 trained a tiny causal LM and generated text. This session adapts a pretrained causal model to follow conversational instructions. It focuses on message serialization, assistant-only loss masking, SFT data quality, evaluation and parameter-efficient adaptation. It is not a repeat of pretraining or decoding.

## Learning objectives

By the end of this lesson, you will be able to:

1. Distinguish pretraining, supervised fine-tuning (SFT), preference optimization and reinforcement learning from human feedback (RLHF).
2. Represent a conversation as role-tagged messages and serialize it with a model-specific chat template.
3. Explain message-end, turn-end and end-of-sequence tokens without assuming they are interchangeable.
4. Build assistant-only loss masks and align them correctly after tokenization, truncation and padding.
5. Construct leakage-resistant instruction datasets and validate their quality.
6. Fine-tune a small causal language model with a supervised demonstration objective.
7. Explain catastrophic forgetting and design regression evaluations.
8. Explain LoRA and estimate its trainable-parameter savings.
9. Evaluate instruction following, structured-output adherence and safety beyond loss alone.
10. Design a production-minded, agent-assisted SFT pipeline with typed contracts and human release approval.

## 60-minute schedule

| Time | Topic |
|---|---|
| 0–5 min | From base LM to instruction-following model |
| 5–15 min | Conversation schemas and chat templates |
| 15–25 min | Assistant-only masking and SFT loss |
| 25–34 min | Dataset quality, forgetting and LoRA |
| 34–40 min | Evaluation and regression gates |
| 40–53 min | PyTorch SFT lab |
| 53–60 min | Production architecture and review |

## 1. Pretraining versus instruction tuning

A pretrained causal language model learns to continue token sequences. It estimates the next-token distribution:

\[
P(x_t\mid x_{<t})
\]

Pretraining can teach language structure, facts represented in the corpus, code patterns and many statistical regularities. It does not automatically teach the model to respond helpfully to a user, obey a requested output schema or use a consistent conversational protocol.

**Supervised fine-tuning (SFT)** continues training from a base checkpoint using curated examples of desired behavior. Each example contains an input context and a demonstration of the response we want the model to learn.

A simplified progression is:

~~~text
Large general corpus
       |
       v
Causal LM pretraining
       |
       v
Base language model
       |
       v
Curated instruction / response demonstrations
       |
       v
Supervised fine-tuning (SFT)
       |
       v
Instruction-tuned checkpoint
       |
       +----> Evaluation and deployment
       |
       +----> Optional preference optimization / RLHF
~~~

SFT is not the same as RLHF. In the InstructGPT research pipeline, supervised demonstrations were followed by preference comparisons, reward-model training and reinforcement-learning optimization. This lesson covers the supervised demonstration stage; preference optimization is a later topic.

**Key distinction:** pretraining teaches broad continuation behavior; SFT shifts the model toward the response patterns represented in the demonstrations. It does not guarantee truth, safety or reliable tool execution.

## 2. Conversation data and roles

A conversational dataset should preserve message structure instead of flattening everything into ambiguous plain text.

Example source record:

~~~json
{
  "messages": [
    {"role": "system", "content": "You are a concise silicon diagnostic assistant."},
    {"role": "user", "content": "Classify SoundWire port 2: retries=5, latency=high."},
    {"role": "assistant", "content": "STATUS: INVESTIGATE"}
  ]
}
~~~

Common roles:

| Role | Meaning |
|---|---|
| system | High-level behavior and policy context |
| developer | Application-level instructions |
| user | User input |
| assistant | Model-authored response or tool call |
| tool name | Result returned by a named external tool |

Roles are part of the serialized training context. They are not security boundaries by themselves. Runtime authorization must be enforced outside the model.

### Chat templates

A chat template maps structured messages to the exact token sequence the model expects. A simplified ChatML-like example is:

~~~text
<|im_start|>system
You are a concise silicon diagnostic assistant.<|im_end|>
<|im_start|>user
Classify SoundWire port 2: retries=5, latency=high.<|im_end|>
<|im_start|>assistant
STATUS: INVESTIGATE<|im_end|>
~~~

The strings shown are illustrative. Real model families use different control-token conventions. Some distinguish:
- end of message,
- end of assistant turn,
- end of sequence,
- hand-off to a tool,
- refusal or untrusted-region markers.

Do not assume that one EOS token means all of these things. Pin the model's actual tokenizer and template revision.

### Training versus inference formatting

During training, serialize the complete demonstration, normally without an extra open-ended generation prompt after the answer.

During inference, the template may append an assistant prefix so generation begins in the correct role:

~~~text
<|im_start|>user
Explain the AXI SLVERR response.<|im_end|>
<|im_start|>assistant
~~~

This is often called a generation prompt. Whether it is needed depends on the model's template.

Avoid double-inserting BOS/EOS or other special tokens. If a template already inserts control tokens, tokenizing the rendered text with another automatic special-token pass can duplicate them. Prefer the tokenizer's native chat-template API and test the exact rendered token IDs.

## 3. Assistant-only loss masking

The model needs the entire conversation as context, but a common SFT objective computes loss only on assistant-authored target tokens.

Why?

- System and user messages are conditioning context, not the desired answer.
- Training on user text can waste capacity reproducing prompts.
- Assistant-only loss makes the supervised target explicit.
- It is especially important for multi-turn and tool-using conversations.

Suppose the serialized conversation is:

~~~text
[BOS] [SYSTEM] system text [END]
[USER] user question [END]
[ASSISTANT] desired answer [TURN_END]
~~~

The input sequence is shifted by one token. The target mask is:

| Target region | Loss? |
|---|---|
| System header and content | No |
| User header and content | No |
| Assistant header | Usually no |
| Assistant response body | Yes |
| Assistant turn-end token | Usually yes, if it is the model's learned stopping target |
| Padding | No |

The precise mask is template-dependent. Some training templates mark assistant generations explicitly. Hugging Face TRL supports assistant-only loss when the chat template exposes the assistant-generation span.

Let token targets be \(y_t\) and let \(m_t\in\{0,1\}\) mark supervised positions:

\[
L_{SFT}=-\frac{1}{\sum_t m_t}
\sum_t m_t\log P(y_t\mid x_{<t})
\]

Implementation idea:

~~~python
labels = input_ids.clone()
labels[~assistant_mask] = -100
loss = cross_entropy(logits, labels, ignore_index=-100)
~~~

In a causal model, remember that the mask must be shifted consistently with the input/target shift. A one-position mask error can silently train on the wrong tokens.

**Padding:** set padded labels to the ignore index. If using packed sequences, ensure attention cannot leak across independent conversations unless that cross-example context is explicitly intended.

### Worked example

Conversation tokens after tokenization:

~~~text
[USER] Explain SLVERR [END] [ASSISTANT] The slave reported an error. [TURN_END]
~~~

After shifting:
- the model sees the user message and assistant prefix,
- the target sequence contains the assistant response,
- only assistant response and the intended turn-end target contribute to loss.

The model still attends to the user prompt. Masking removes prompt positions from the loss; it does not remove them from the forward-pass context.

## 4. Tool-call conversations need explicit treatment

A tool-using conversation can look like:

~~~text
[user] Read the PCIe link status.
[assistant to=read_pcie_status] {"device_id":"dev-17"} [handoff]
[tool] {"link_speed":"Gen4","link_width":4}
[assistant] The link is operating at Gen4 x4.
~~~

The assistant's tool-call message is model-authored behavior and may be a supervised target. The tool result is external context and is usually not treated as an assistant target. The final assistant answer is supervised.

Do not flatten tool calls into ordinary prose. Preserve:
- tool name,
- typed arguments,
- hand-off marker,
- tool result role,
- correlation/request ID where applicable.

Tool schemas and runtime permissions remain deterministic. SFT can teach the model a tool-call format, but cannot grant the model authority to execute a tool.

## 5. Dataset quality and split design

SFT quality depends heavily on the demonstrations.

A useful dataset review includes:

| Dimension | Check |
|---|---|
| Correctness | Is the answer technically accurate? |
| Instruction match | Does it answer the actual request? |
| Consistency | Do similar inputs receive compatible treatment? |
| Diversity | Are task types, wording and operating conditions covered? |
| Safety | Does the example avoid unsafe or unauthorized actions? |
| Formatting | Does it obey the required schema and stop convention? |
| Privacy/licensing | Are sources permitted and sensitive data handled? |
| Duplication | Are near-identical examples overrepresented? |

Split by conversation, source, repository, device, customer or time before expanding or packing examples. Keep near-duplicates and paraphrases in the same split where appropriate. Do not let a generated variant of a training example appear in validation or test.

For hardware diagnostics, include both normal and abnormal cases, ambiguous cases, missing evidence, contradictory logs and cases where the correct answer is to abstain or request more evidence.

## 6. Catastrophic forgetting

Fine-tuning changes model parameters to improve performance on the SFT distribution. If the dataset is narrow, the model may become less capable on behaviors it previously handled.

This is called catastrophic forgetting or capability regression.

Possible symptoms:
- better formatting but worse general question answering,
- stronger domain vocabulary but weaker general language behavior,
- overuse of a narrow response template,
- reduced ability to handle tasks absent from the fine-tuning corpus.

Mitigations to evaluate:
- use a conservative learning rate and limited epochs,
- curate broader and more diverse demonstrations,
- retain a small representative replay/pretraining mixture where appropriate,
- use parameter-efficient adapters,
- use early stopping based on a balanced validation suite,
- evaluate general capabilities before and after tuning.

No mitigation guarantees zero forgetting. Always measure it.

## 7. LoRA — parameter-efficient fine-tuning

Full fine-tuning updates all model weights. For a large model, optimizer states and gradients can require substantial memory.

LoRA freezes the original weight matrix and learns a low-rank update:

\[
W'=W+\Delta W
\]

\[
\Delta W=BA\cdot\frac{\alpha}{r}
\]

For a matrix with dimensions \(d_{out}\times d_{in}\):
- full update parameters: \(d_{out}d_{in}\),
- LoRA parameters: \(r(d_{in}+d_{out})\).

Example with a 64×64 linear layer and rank 4:

| Method | Trainable parameters |
|---|---:|
| Full weight matrix | 4,096 |
| LoRA A and B | 512 |

The bias is excluded from this comparison. LoRA reduces trainable parameters, optimizer state and adapter storage. It does not remove the base model's forward-pass cost or guarantee that the adapted model preserves all previous capabilities.

A common initialization sets one low-rank factor to zero so the initial adapter contribution is zero.

LoRA is an adaptation method, not a replacement for good data, evaluation or release governance.

## 8. Evaluation: loss is necessary but not sufficient

A lower SFT loss shows that the model assigns higher likelihood to demonstrated target tokens. It does not prove that the model follows unseen instructions or produces safe, useful answers.

Build a held-out evaluation suite with several layers:

| Evaluation | Example metric |
|---|---|
| Assistant-token prediction | Held-out assistant-only NLL |
| Instruction following | Human rubric or task success |
| Structured output | JSON/schema validity |
| Tool calling | Correct tool, arguments and hand-off format |
| Domain accuracy | Expert-reviewed diagnostic correctness |
| Abstention | Correctly requests missing evidence |
| Safety | Unauthorized-action and prompt-injection tests |
| Regression | General capability suite before/after SFT |

For deterministic classification or schema tasks, exact match and schema validation can be useful. For open-ended explanations, use a rubric and expert review; string matching is too brittle.

Keep the final test set protected until the candidate and acceptance thresholds are frozen.

## 9. Hands-on lab: a tiny two-stage SFT pipeline

The runnable lab uses a small character-level causal Transformer to make chat formatting and loss masking visible. It is an educational model, not a production tokenizer or a replacement for a pretrained foundation model.

### Lab stages

1. Generate synthetic silicon-diagnostic instruction/response examples.
2. Split complete prompt/response examples into train, validation and test sets.
3. Build a fixed toy vocabulary with atomic control tokens.
4. Serialize system, user and assistant messages.
5. Pretrain the tiny model on all serialized tokens as a toy base-LM stage.
6. Fine-tune the same model with loss only on assistant response and end-marker tokens.
7. Evaluate assistant-only and full-sequence validation loss.
8. Generate a response for a seen training prompt.
9. Save and restore a checkpoint.
10. Run a LoRA parameter-count and zero-initialized-update sanity check.

### Reference CPU run

Environment: PyTorch 2.10.0 CPU, seed 23023.

| Measurement | Result |
|---|---:|
| Train / validation / test examples | 118 / 20 / 20 |
| Character vocabulary plus control tokens | 72 |
| Tiny model parameters | 21,216 |
| Assistant target tokens in train set | 4,151 |
| Validation assistant loss: random → toy pretrain | 22.3709 → 1.1773 |
| Validation assistant loss after SFT | 0.5857 |
| Held-out test assistant loss after SFT | 0.5170 |
| Validation full-sequence loss: after pretrain → after SFT | 0.5139 → 2.2523 |
| Checkpoint restore max-logit error | 0.0 |
| LoRA rank-4 trainable params vs full 64×64 linear weight+bias | 512 vs 4,160 |

The rise in full-sequence loss while assistant-only loss falls is an intentional illustration of objective shift: SFT no longer optimizes user/system tokens. It is a warning to monitor regressions, not proof that a production model has catastrophically forgotten general capabilities.

A greedy response for a seen training prompt was:

~~~text
Prompt: Classify SoundWire port 2: retries=5, latency=low.
Response: STATUS: INVESTIGATE
~~~

This is a memorized synthetic training example, not a held-out generalization result. The toy corpus is small and repetitive. Its scores are not comparable to real LLM benchmarks.

### Run locally

~~~bash
cd daily-lessons/day-023
python -m pip install -r requirements.txt
python lab.py
python lab.py --device cpu
python lab.py --device cuda
~~~

The CPU path was executed. The CUDA branch is available but was not hardware-tested in the reference run.

### Exercises

1. Print the rendered chat string and token IDs for one example.
2. Print the assistant mask and verify that user/system target positions are ignored.
3. Deliberately shift the mask by one position and observe the failure.
4. Compare assistant-only loss with full-sequence loss.
5. Add a multi-turn example and mask each assistant response correctly.
6. Add a tool-call message, a tool result and a final assistant response.
7. Add a held-out JSON-schema evaluator and a refusal/abstention test.
8. Change SFT learning rate and number of steps; track both task metrics and regression metrics.
9. Compare full fine-tuning with a LoRA adapter on a larger pretrained model using PEFT.
10. Test template mismatch by changing a control token; verify that the compatibility test fails.

## 10. A29 — Instruction-Tuning and Chat-Template Governance Gate

### Architecture

~~~mermaid
flowchart TD
    A[Instruction-tuning request] --> B[Read-only intake agent]
    B --> C[Typed SFT run contract]
    C --> D[Dataset policy and split validator]
    D --> E[Versioned chat template and tokenizer]
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

### Component boundaries

| Component | Owns | Must not own |
|---|---|---|
| Dataset service | Source policy, quality review, dedup and split manifests | Model promotion |
| Chat-template service | Role serialization, control-token IDs and generation prefix | Runtime authorization |
| Tokenizer | Normalization, vocabulary, special IDs and tokenization | Dataset access policy |
| Mask builder | Assistant/tool-call target spans, shift alignment and padding mask | Editing source labels |
| Trainer | Full/LoRA updates, optimizer and checkpointing | Protected test labels |
| Evaluator | Instruction, schema, tool-call, safety and regression metrics | Training data mutation |
| Agent assistant | Explain reports and propose bounded experiments | Arbitrary code execution or release authority |

### Seven reusable agent patterns

| Pattern | Application | Trade-off |
|---|---|---|
| Deterministic workflow | Validate → template → tokenize → mask → train → evaluate | Reproducible; less flexible for exploration |
| Single agent + typed tools | Inspect manifests, masks and evaluation reports | Low overhead; limited specialization |
| Router-specialist | Route data, template, training, evaluation and safety questions | Focused expertise; requires abstain/fallback |
| Planner–executor–verifier | Plan bounded SFT experiments and independently verify masks/results | Strong control; adds latency and state |
| Supervisor-worker | Parallel seeds, LoRA ranks or learning-rate trials | Faster exploration; higher compute and merge complexity |
| Event-driven | Queue runs, checkpoints and evaluation events | Scalable; needs durable state and idempotency |
| Human-in-the-loop | Approve dataset exceptions and model promotion | Safer; adds review latency |

Use the simplest architecture that satisfies the requirement. Parallel workers need immutable manifests, isolated checkpoints and explicit candidate-selection rules.

### Typed tool contract

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

Validate model/tokenizer/template compatibility, numeric bounds, caller identity, resource quotas and dataset permissions in deterministic services. The model may suggest values but cannot grant itself access to the test set or execute arbitrary training code.

### State and memory

Persist:
- Dataset/source/license and split manifests.
- Message schema and chat-template hash.
- Tokenizer artifact hash, normalization and special-token IDs.
- Assistant-mask generation rules and mask-test results.
- Base model revision and adapter configuration.
- Optimizer, scheduler, scaler, RNG, sampler and step state.
- Checkpoint/adapter hashes, evaluation suite revision and approval record.

Keep agent conversation state, training checkpoint state and serving conversation/KV-cache state separate. A template or tokenizer change is a compatibility change that requires re-tokenization and evaluation.

### Security

- Treat dataset content as untrusted text, not agent instructions.
- Apply privacy, licensing, retention and consent checks before training.
- Keep protected test labels inaccessible to planners and workers.
- Sandbox preprocessing and training; deny arbitrary shell/network access by default.
- Do not train on secrets, credentials or unauthorized internal logs.
- Validate tool-call schemas and enforce permissions outside the model.
- Sign or hash artifacts and load only trusted checkpoints.
- Require human approval for production promotion.

### Evaluation and reliability

Required gates:
- Role/template round-trip and special-token tests.
- Assistant-mask coverage and shifted-target alignment.
- Nonzero supervised-token count for every training batch.
- Padding and truncation correctness.
- Held-out assistant NLL plus instruction task metrics.
- Structured-output and tool-call schema tests.
- Safety, abstention and prompt-injection regression tests.
- General-capability regression suite.
- Checkpoint/adapter restore test.
- Protected final test after candidate freeze.

Use immutable run IDs, idempotent job submission, atomic checkpoints, bounded retries, cancellation, deadlines and durable event offsets. A failed mask or tokenizer mismatch is a deterministic data/configuration error, not a transient retry.

### Observability, cost and latency

Track:
- Examples and assistant target tokens per second.
- Prompt/context tokens versus supervised response tokens.
- Masked-token ratio and truncation rate.
- Train/validation loss and task-specific success.
- Learning rate, gradient norm and skipped optimizer steps.
- Trainable parameter count, peak memory and GPU utilization.
- Checkpoint duration/size and resume count.
- p50/p95 training-step latency and inference latency.
- Adapter size, cost per successful task and cost per million supervised tokens.

LoRA often reduces trainable-state memory and adapter storage, but benchmark actual throughput. SFT does not necessarily make inference faster; the base model still performs its forward pass.

**Release rule:** the assistant can explain results and propose experiments. Deterministic services own dataset boundaries, chat serialization, loss masks, training and evaluation. Independent verification and human approval own production promotion.

## Key terminology

- **SFT:** supervised fine-tuning on curated demonstrations.
- **Chat template:** model-specific serialization of role-tagged messages.
- **Assistant-only loss:** cross-entropy is computed only on designated assistant target tokens.
- **Generation prompt:** template prefix that signals the assistant response should begin.
- **EOS / EOT:** end-of-sequence or end-of-turn markers; exact semantics are model-specific.
- **Teacher forcing:** conditioning training on the known preceding tokens.
- **Catastrophic forgetting:** degradation on previously learned capabilities after adaptation.
- **LoRA:** low-rank trainable updates applied while base weights remain frozen.
- **Adapter:** compact set of learned parameters used to adapt a base model.
- **Preference optimization:** learning from comparative judgments, beyond SFT demonstrations.

## Five review questions

1. Why is a chat template part of the model/tokenizer compatibility contract rather than just display formatting?
2. Why does assistant-only loss mask user/system targets while retaining those tokens in the model's context?
3. What can go wrong if the assistant mask is shifted by one token?
4. How does LoRA reduce trainable parameters, and what does it not eliminate?
5. Why must a fine-tuned model be evaluated on both instruction tasks and general-capability regression tests?

## Recommended references

1. [Hugging Face Transformers — Chat Templates](https://huggingface.co/docs/transformers/main/chat_templating)
2. [Hugging Face TRL — SFT Trainer](https://huggingface.co/docs/trl/en/sft_trainer)
3. [Ouyang et al. — Training Language Models to Follow Instructions with Human Feedback (2022)](https://arxiv.org/abs/2203.02155)
4. [Hu et al. — LoRA: Low-Rank Adaptation of Large Language Models (2021)](https://arxiv.org/abs/2106.09685)
5. [Hugging Face PEFT — LoRA guide](https://huggingface.co/docs/peft/main/conceptual_guides/lora)
6. [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

---

## Next lesson — Session 024

**Preference Optimization and Alignment: DPO, Reward Models, Pairwise Comparisons and Evaluation**

- Why SFT alone does not capture preferences.
- Chosen and rejected response pairs.
- Reward-model intuition.
- Bradley–Terry preference modeling.
- DPO objective and reference policy.
- KL regularization intuition.
- Preference-data quality and disagreement.
- Safety and helpfulness evaluation.
- Build a tiny pairwise preference experiment.
- Production preference-data governance and human review.
