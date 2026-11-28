# Session 021 — Tokenization, Embeddings and Next-Token Language Modeling

**Phase 5 · Transformers and Generative AI · 60-minute lesson**

Session 020 introduced attention and the Transformer block. This lesson connects raw text to the numerical tensors those blocks consume, then trains a tiny causal language model. The path is: text → token IDs → embeddings → shifted targets → cross-entropy → next-token prediction.

> **Central question:** How does a model turn text into numbers, learn relationships between those numbers, and predict the next piece of text?

## Learning objectives

By the end, you should be able to:

1. Distinguish characters, bytes, words and subword tokens.
2. Explain vocabulary IDs, normalization, pre-tokenization and special tokens.
3. Describe BPE, WordPiece and Unigram at a high level.
4. Explain why tokenization is a versioned model interface, not cosmetic preprocessing.
5. Calculate an embedding lookup and distinguish token embeddings from positional information.
6. Build shifted input/target sequences for causal language modeling.
7. Explain teacher forcing, logits, cross-entropy and perplexity.
8. Implement and test a small BPE tokenizer and decoder-only language model.
9. Identify tokenizer/data leakage, padding, context and output-vocabulary pitfalls.
10. Design a production-minded training and inference workflow with deterministic gates and bounded agent assistance.

## One-hour plan

| Time | Topic |
|---|---|
| 0–5 min | Recap: Transformer input and output |
| 5–15 min | Tokenization and subword algorithms |
| 15–23 min | Embedding matrices and positional information |
| 23–33 min | Next-token prediction, teacher forcing and loss |
| 33–40 min | Perplexity, generation and tokenizer trade-offs |
| 40–53 min | Python lab: toy BPE + tiny causal LM |
| 53–60 min | Production architecture and review |

## Key terminology

| Term | Meaning |
|---|---|
| Token | An indivisible unit emitted by a tokenizer: word, subword, byte or special symbol |
| Vocabulary | Mapping between token units and integer IDs |
| Pre-tokenization | Rule-based segmentation before a subword algorithm |
| Normalization | Canonical text transformations defined by a tokenizer |
| BPE | Byte Pair Encoding; repeatedly merges frequent adjacent symbol pairs |
| WordPiece | Subword method that chooses merges using a likelihood-oriented criterion |
| Unigram | Subword model that starts with candidate pieces and prunes them using a probabilistic objective |
| Special token | Reserved marker such as BOS, EOS, PAD or a model-specific control marker |
| Token ID | Integer index in a particular vocabulary; IDs have no inherent numeric meaning |
| Embedding | Learned dense vector associated with a token ID |
| Teacher forcing | Training with ground-truth preceding tokens as context |
| Logit | Unnormalized score for a possible next token |
| Cross-entropy | Negative log-likelihood objective for the correct next token |
| Perplexity | Exponential of average token-level negative log-likelihood |
| Context length | Maximum sequence length processed in one model context |
| Weight tying | Sharing input token embedding weights with the output vocabulary projection |

## 1. From text to tensors

A Transformer does not consume a string directly. A typical text path is:

~~~text
Raw text
   |
   v
Tokenizer normalization / pre-tokenization
   |
   v
Subword or byte-level segmentation
   |
   v
Token strings
   |
   v
Vocabulary lookup
   |
   v
Integer token IDs
   |
   v
Embedding lookup + positional information
   |
   v
Transformer blocks
   |
   v
Vocabulary logits at each position
~~~

For example, the sentence:

~~~text
dma retry rate high
~~~

might become pieces such as:

~~~text
["▁dma", "▁retry", "▁rate", "▁high"]
~~~

and then integer IDs:

~~~text
[418, 921, 77, 306]
~~~

Those IDs are illustrative only. The same text can produce different pieces and IDs with different tokenizers. ID 418 is not more meaningful than ID 77; it is an index into one specific vocabulary.

### Characters, bytes, words and subwords

| Unit | Advantage | Limitation |
|---|---|---|
| Character | Small vocabulary; composes unseen words | Long sequences; morphology is spread across many steps |
| UTF-8 byte | Fixed base alphabet of 256 values; represents any valid text | Some characters need multiple byte tokens; sequences can be longer |
| Whole word | Short sequences for common words | Huge vocabulary, rare-word/OOV problem, morphology and multilingual coverage |
| Subword | Reuses frequent pieces and balances vocabulary with sequence length | Segmentation is model-specific; rare strings may still split into many tokens |

Modern LLM tokenizers commonly use subword or byte-level subword methods. Tokenization is not simply splitting on spaces: punctuation, whitespace, Unicode and code can all affect segmentation.

### Why subwords?

A whole-word vocabulary has a problem: a new name, register identifier or inflected word may not exist in the vocabulary. Subword tokenization composes it from known pieces. Byte-level approaches can provide a fallback for arbitrary UTF-8 text without requiring a dedicated unknown token for every unseen character.

For multilingual text, compare token fertility (tokens per word, character or byte) across languages and scripts. A tokenizer efficient for English may split Telugu, other Indic scripts, source code or hardware identifiers less efficiently. Measure this rather than assuming equal coverage.

## 2. How Byte Pair Encoding works

BPE begins with small symbols and repeatedly merges the most frequent adjacent pair.

A simplified procedure:

1. Apply a defined normalization and pre-tokenization policy.
2. Represent each pre-tokenized word as a sequence of base symbols.
3. Count adjacent symbol pairs, weighted by word frequency.
4. Merge the most frequent pair into a new symbol.
5. Repeat until the requested merge count or vocabulary size is reached.
6. At inference, apply the learned merge rules in their learned order.

### Worked example

Suppose a tiny corpus contains:

~~~text
low low lower lowest
newer newer newest
~~~

Start from characters (a production tokenizer also has explicit boundary and byte/Unicode handling). If the pair l,o is frequent, it can be merged into lo. A later merge can combine lo,w into low. The learned merge list is part of the tokenizer artifact.

The exact result depends on pre-tokenization, boundary markers, corpus frequencies, tie-breaking and merge count. A hand-written toy tokenizer does not reproduce a production GPT tokenizer.

### BPE, WordPiece and Unigram

| Method | High-level behavior | Common context |
|---|---|---|
| BPE | Merge the most frequent adjacent pair | GPT-style byte-level BPE and many open models |
| WordPiece | Choose a merge using a likelihood-oriented score | BERT-family tokenizers |
| Unigram | Start with candidate pieces and prune those least useful under a probabilistic model | SentencePiece-based systems |

SentencePiece is a tokenizer framework that can operate directly on raw text and supports BPE or Unigram models. Byte-level BPE uses the 256 possible byte values as its base alphabet, then learns merges.

### Special tokens

Common examples:

- BOS: beginning of sequence.
- EOS: end of sequence.
- PAD: padding for rectangular batches.
- UNK: unknown token, if the tokenizer needs one.
- Additional task/chat markers: role boundaries, tool calls, image/audio markers or control tokens, depending on the model.

A special-token string is not necessarily interpreted as special merely because those characters appear in user text. The tokenizer's added-token configuration and escaping policy determine how it is handled. Chat templates and tool markers are model-specific interfaces.

**Engineering rule:** store tokenizer files, normalization settings, pre-tokenizer configuration, vocabulary, merge ranks, special-token IDs and a cryptographic artifact hash together. A model checkpoint paired with the wrong tokenizer can become unusable even when tensor shapes match.

## 3. Embedding matrices

A token ID is an index, not a useful continuous representation. An embedding layer maps it to a learned vector.

Let vocabulary size be V and embedding dimension be d:

\[
E\in\mathbb{R}^{V\times d}
\]

For token ID i:

\[
e_i=E[i]
\]

A batch of token IDs with shape [B,T] becomes embeddings with shape [B,T,d].

### Worked lookup

Suppose:

\[
E=
\begin{bmatrix}
0.1&0.2&0.3\\
-0.2&0.4&0.5\\
0.7&0.0&-0.1
\end{bmatrix}
\]

For token ID 1:

\[
E[1]=[-0.2,\;0.4,\;0.5]
\]

In PyTorch, `nn.Embedding(V, d)` performs this table lookup. It does not need to construct an explicit one-hot vector and multiply it by E.

~~~python
embedding = torch.nn.Embedding(num_embeddings=1000, embedding_dim=64)
token_ids = torch.tensor([[12, 7, 99]])  # [batch=1, time=3]
vectors = embedding(token_ids)          # [1, 3, 64]
~~~

Embedding vectors are learned by gradient descent as part of the language-model objective. Similar-looking token vectors may emerge, but embedding geometry is not guaranteed to be a perfect human-readable semantic map.

### Add position information

Token embeddings alone do not identify sequence order. A simple learned-position model uses:

\[
X=E_{token}[ids]+E_{position}[positions]
\]

Other Transformers inject position through sinusoidal encoding or rotary/relative methods. The position method must match the model architecture and checkpoint.

### Input embedding and output projection

At each position, the model produces a hidden vector h. A vocabulary projection maps it to V logits:

\[
logits=hW_{vocab}^{T}+b
\]

Some language models tie the output projection weights to the input embedding matrix E:

\[
W_{vocab}=E
\]

Weight tying reduces parameters and couples input/output token representations. It is a design choice, not a requirement.

## 4. Next-token language modeling

A causal language model learns:

\[
P(x_t\mid x_0,\ldots,x_{t-1})
\]

It predicts the next token from preceding tokens.

For the sequence:

~~~text
[BOS, dma, retry, EOS]
~~~

Training inputs and targets are shifted by one:

| Position | Input | Target |
|---:|---|---|
| 0 | BOS | dma |
| 1 | dma | retry |
| 2 | retry | EOS |

In tensor form:

~~~python
input_ids = sequence[:, :-1]
target_ids = sequence[:, 1:]
~~~

This shift is essential. If inputs and targets are not shifted, the model may be trained to copy the current token rather than predict the next one.

### Teacher forcing

During standard next-token training, each position receives the ground-truth preceding tokens, not the model's sampled predictions. This is teacher forcing. The causal mask ensures position t cannot inspect target tokens from positions after t.

The model can calculate logits for all input positions in parallel:

\[
logits\in\mathbb{R}^{B\times T\times V}
\]

The target tensor has shape [B,T]. Padding positions should be excluded from loss, commonly using an ignore index or explicit loss mask.

### Cross-entropy objective

For target token y_t and logits z_t:

\[
L_t=-\log softmax(z_t)_{y_t}
\]

Average over valid target tokens:

\[
L=-\frac{1}{N_{valid}}\sum_{t\in valid}\log P(y_t\mid context_t)
\]

PyTorch `CrossEntropyLoss` expects raw logits, not probabilities after softmax.

~~~python
logits = model(input_ids)  # [B, T, V]
loss = torch.nn.functional.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    target_ids.reshape(-1),
    ignore_index=pad_id,
)
~~~

The example assumes padding IDs are present and input/target alignment is correct. For a batch with no padding, the ignore index is unnecessary.

## 5. Perplexity and evaluation

Perplexity is:

\[
PPL=\exp\left(\frac{1}{N}\sum_{t=1}^{N}-\log P(x_t\mid x_{<t})\right)
\]

If average token negative log-likelihood is 0.5 nats:

\[
PPL=e^{0.5}\approx1.649
\]

Lower perplexity means the model assigns higher probability to observed next tokens on that evaluation stream.

Important caveats:

- Compare perplexity on the same evaluation text, tokenizer, tokenization policy and target masking.
- Different tokenizers segment text differently, so raw token-level perplexity is not directly comparable across tokenizers.
- Perplexity is not a complete measure of factuality, reasoning, safety, instruction following or usefulness.
- Report the number of valid target tokens and the exact evaluation corpus/split.
- For cross-tokenizer comparisons, consider byte-normalized or word-normalized likelihood measures with clearly stated conventions.

### Data split and tokenizer leakage

Split documents by source, conversation, device, time or another relevant group before fitting learned preprocessing. Fit the tokenizer on the declared training corpus only, unless a documented pretraining setup specifies otherwise. Deduplicate before splitting to reduce near-duplicate leakage.

For a production corpus:
- Keep validation and test documents out of tokenizer training when evaluating strict held-out generalization.
- Preserve document boundaries and EOS policy.
- Do not let windows cross train/validation/test boundaries.
- Record corpus manifest, tokenizer revision and preprocessing version.

## 6. From next-token training to generation

At inference:

1. Tokenize the prompt using the exact model tokenizer.
2. Check the context limit.
3. Run the model and obtain last-position logits.
4. Apply decoding policy (greedy, temperature, top-k or top-p).
5. Select/sample the next token.
6. Append it and repeat until EOS or a budget/stop condition.

~~~text
Prompt -> tokenizer -> token IDs -> model -> next-token logits
  ^                                                   |
  |                                                   v
  +----------- append selected token <--- decoder policy
~~~

Training is parallel over positions; autoregressive generation is sequential over newly generated tokens. KV caching, batching and optimized kernels improve serving throughput, but do not remove the dependency between generated tokens.

## 7. Worked systems example: hardware diagnostic language

A log excerpt might contain:

~~~text
soundwire port 2 retry count increased
dma queue depth high after clock change
~~~

A general-purpose tokenizer may split identifiers such as soundwire, port2, register names, hexadecimal values or error codes into many pieces. That affects:
- Context budget and truncation.
- Throughput and cost.
- Ability to preserve exact identifiers.
- Retrieval chunk sizes and alignment.
- Model's ability to learn recurring technical strings.

For engineering corpora, measure token counts for representative logs, C/C++ identifiers, register names, hexadecimal values, Telugu/English mixed text and protocol traces. Preserve raw source alongside normalized text. Never normalize away case or punctuation when semantically meaningful.

Do not invent a new special token for every register. First measure the effect of existing tokenization; if a vocabulary change is warranted, it changes the tokenizer/model interface and normally requires deliberate adaptation and compatibility testing.

## 8. Hands-on lab: toy BPE + tiny causal language model

The runnable `lab.py` contains:

1. A deterministic educational BPE trainer over a small whitespace-pretokenized corpus.
2. A vocabulary and token-to-ID mapping with BOS/EOS/PAD/UNK entries.
3. An encode/decode round-trip check.
4. An embedding lookup using `nn.Embedding`.
5. A tiny decoder-only Transformer with causal scaled-dot-product attention.
6. Shifted next-token inputs and labels.
7. Cross-entropy training with gradient clipping.
8. Validation/test loss and test perplexity.
9. A causal future-token leakage assertion.

The corpus is synthetic diagnostic-style text, not real silicon logs. The toy BPE implementation uses a visible word-start marker and character symbols. It is intentionally not production byte-level BPE: it omits robust Unicode normalization, byte fallback, production pre-tokenization, added-token matching and tokenizer serialization standards.

### Reference CPU run

Environment: PyTorch 2.10.0 CPU.

| Check | Result |
|---|---:|
| Training / validation / test documents | 192 / 24 / 24 |
| BPE merges learned | 80 |
| Toy vocabulary size | 108 |
| Round-trip sample | Exact |
| Embedding lookup shape | (1, 3, 48) |
| Training stream tokens | 3,662 |
| Validation loss before → after | 32.2449 → 0.3521 |
| Test loss | 0.3599 |
| Test perplexity | 1.433 |
| Causal future-token leakage | 0 |

The unusually high initial loss and low final loss are specific to a tiny synthetic, highly repetitive corpus. They are not a benchmark or a claim about general language understanding.

### Run locally

~~~bash
cd daily-lessons/day-021
python -m pip install -r requirements.txt
python lab.py
python lab.py --device cpu
python lab.py --device cuda
~~~

The reference run used CPU. The CUDA path is available with a compatible CUDA-enabled PyTorch installation and GPU, but was not hardware-tested in this run.

### Exercises

1. Change BPE merge count from 80 to 20 and 120. Compare token counts and vocabulary size.
2. Compare the same diagnostic text under character-level and toy-BPE tokenization.
3. Change embedding dimension from 48 to 64; inspect parameter count and memory.
4. Disable weight tying and compare parameter count.
5. Remove the one-position target shift and explain why the objective is wrong.
6. Add a padded batch and verify PAD targets do not contribute to loss.
7. Increase context length from 24 to 48 and observe attention cost.
8. Generate tokens autoregressively with greedy decoding and stop at EOS.
9. Build tokenizer coverage tests for Telugu, English, code identifiers and hexadecimal strings.
10. Add a checkpoint containing model, optimizer, tokenizer hash, RNG and training step; restore and verify identical next-token logits.

## 9. A27 — Tokenizer and Language-Model Training Governance Gate

### Architecture

~~~mermaid
flowchart TD
    A[Training objective] --> B[Read-only intake agent]
    B --> C[Typed corpus and tokenizer contract]
    C --> D[Corpus policy, dedup and split validator]
    D --> E[Deterministic tokenizer pipeline]
    E --> F[Versioned tokenizer artifact]
    F --> G[Tokenization coverage and round-trip tests]
    G --> H[Deterministic causal-LM trainer]
    H --> I[Checkpoint and token-level metrics]
    I --> J[Independent verifier]
    J --> K[Validation evaluator]
    K --> L{Acceptance policy}
    L -->|Reject| M[Bounded retry or reject]
    L -->|Accept| N[Frozen candidate]
    N --> O[Protected test gate]
    O --> P[Human approval]
    P --> Q[Model registry]
    Q --> R[Version-matched inference service]
~~~

### Component boundaries

| Component | Owns | Must not own |
|---|---|---|
| Corpus intake | Source allow-list, license/PII policy, immutable manifest | Model weights |
| Tokenizer pipeline | Normalization, pre-tokenization, vocab, merges, special IDs | Authorization policy |
| Dataset builder | Document splits, BOS/EOS, windows, PAD masks | Changing test labels |
| Trainer | Embeddings, Transformer, loss, optimizer, checkpoints | Release approval |
| Evaluator | Held-out token NLL, perplexity, coverage and regression tests | Rewriting training data |
| Registry | Immutable model/tokenizer pair and lineage | Unreviewed model mutation |
| Assistant | Explain reports, propose bounded experiments | Arbitrary corpus access or release authority |

### Seven reusable agent patterns

| Pattern | Application | Trade-off |
|---|---|---|
| Deterministic workflow | Corpus validation → tokenizer → dataset → training → evaluation | Most reproducible; less flexible for ambiguous research questions |
| Single agent + typed tools | Read-only corpus statistics and experiment reports | Low orchestration overhead; limited specialization |
| Router-specialist | Route tokenizer, data-quality, training and evaluation questions | Better focus; routing needs fallback |
| Planner–executor–verifier | Plan bounded experiments, execute, independently verify | Stronger control; more latency and state |
| Supervisor-worker | Parallel tokenizer configurations or model seeds | Higher throughput; requires isolated jobs and explicit aggregation |
| Event-driven | Queue tokenization/training jobs and publish checkpoint/metric events | Scales asynchronous work; needs idempotency and durable state |
| Human-in-the-loop | Approve corpus-policy exceptions, tokenizer compatibility changes and model promotion | Safer high-impact decisions; adds review latency |

Prefer the simplest pattern that satisfies the task. Parallel workers must use immutable manifests and disjoint job IDs; they must not race to mutate a shared tokenizer or model artifact.

### Typed tool contracts (illustrative)

~~~python
from dataclasses import dataclass

@dataclass(frozen=True)
class TokenizeRequest:
    text: str
    tokenizer_revision: str
    add_special_tokens: bool

@dataclass(frozen=True)
class TrainLMRequest:
    dataset_manifest_sha256: str
    tokenizer_sha256: str
    context_length: int
    batch_size: int
    max_steps: int
    seed: int
~~~

Validate bounds, revisions and caller permissions in deterministic services. A model-generated plan is data to validate, not executable authority.

### State and memory

Persist:
- Corpus snapshot and source/license manifest.
- Deduplication and split manifest.
- Tokenizer artifact and hash.
- Special-token IDs and normalization/pre-tokenizer configuration.
- Model config and tokenizer compatibility record.
- Optimizer, scheduler, RNG, sampler and training-step checkpoint state.
- Evaluation report and approval record.

Keep conversational memory separate from training state and inference KV cache. A tokenizer revision change invalidates token-ID compatibility; a model checkpoint must never be silently paired with a different vocabulary mapping.

### Security

- Treat corpus text as untrusted data, not instructions to the training agent.
- Scan for secrets, PII, malware-like payloads and licensing restrictions before ingestion.
- Do not execute source snippets or shell commands found in training documents.
- Sandbox tokenization and training jobs with resource and filesystem limits.
- Use least-privilege dataset access and tenant isolation.
- Protect test corpora and evaluation labels.
- Load only trusted checkpoints; use safe serialization and artifact verification.

### Evaluation and reliability

Required gates:
- Deterministic tokenizer output for a fixed revision.
- Encode/decode and special-token tests.
- Token ID range checks and vocabulary/model-size consistency.
- Shifted-target alignment and causal-mask tests.
- PAD exclusion from loss.
- Train/validation/test source separation and dedup checks.
- Held-out token NLL and perplexity with corpus, tokenizer and valid-token metadata.
- Checkpoint restore and reproducible evaluation.
- Regression coverage on code, identifiers and multilingual samples.

A low perplexity on a duplicated or narrow corpus is not sufficient evidence of a good assistant. Add task-specific evaluation later: factuality, retrieval grounding, instruction following, tool correctness and safety.

### Observability, reliability and cost/latency

Track:
- Documents, bytes and tokens ingested.
- Tokens per byte/word by language and data type.
- Unknown/fallback and truncation rates.
- Tokenizer throughput and CPU time.
- Training tokens/sec, loss, gradient norm and GPU memory.
- Checkpoint age, resume count and failed jobs.
- Time-to-first-token, inter-token latency and generated tokens/sec.
- Cost per million training/inference tokens and per successful task.

Use immutable job IDs, idempotent submissions, bounded retries, checkpoint recovery, deadlines and cancellation. For inference, cap context/output tokens and concurrency. Vocabulary size affects embedding and output-projection cost; context length affects attention and KV-cache memory.

**Release rule:** an LLM may explain tokenization and propose experiments, but deterministic pipelines own token IDs, masks, loss and training. Independent evaluation and human approval own model promotion.

## Five review questions

1. Why can the same text produce different token IDs with two different tokenizers?
2. What is the difference between BPE and WordPiece at a high level?
3. Why is embedding lookup more efficient than explicitly constructing a one-hot vector and multiplying by the embedding matrix?
4. For input tokens [BOS, A, B], what are the correctly shifted next-token targets?
5. Why is raw token-level perplexity not directly comparable across different tokenizers?

## Recommended references

1. [Hugging Face — Tokenization algorithms](https://huggingface.co/docs/transformers/main/en/tokenizer_summary) — BPE, WordPiece, Unigram and byte-level approaches.
2. [Hugging Face Tokenizers — Components](https://huggingface.co/docs/tokenizers/python/latest/components.html) — tokenizer pipeline, models, normalizers and pre-tokenizers.
3. [Sennrich, Haddow and Birch (2016), Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/) — foundational BPE-for-NLP paper.
4. [Kudo and Richardson (2018), SentencePiece](https://aclanthology.org/D18-2012/) — language-independent subword tokenizer.
5. [PyTorch nn.Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html) — embedding lookup.
6. [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) — logits and target contract.
7. [Radford et al. (2019), Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2 language-modeling report.
8. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer architecture from Session 020.

## Next lesson — Session 022

**Causal Language Model Pretraining, Sampling and Text Generation**

- Autoregressive decoding loop.
- Greedy decoding, temperature, top-k and top-p.
- EOS and stop conditions.
- Training corpus construction and packing.
- Learning-rate schedules and warmup.
- Gradient accumulation and mixed precision.
- Checkpointing and validation.
- Tiny GPT-style model training.
- Generation quality and repetition diagnostics.
- Training/inference cost and throughput.

**Milestone:** 21 sequential lessons published. Phase 5 now connects Transformer attention to the tokenizer, embedding and language-model objective that make a GPT-style system work.
