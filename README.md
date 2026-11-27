# AI Basics to Masterclass

A structured, hands-on learning journey from AI fundamentals to building and deploying modern AI systems.

**Format:** one focused 60-minute lesson per scheduled session, delivered sequentially  
**Approach:** concepts → architecture → worked examples → coding lab → review  
**Level:** beginner-friendly foundations, progressing to advanced AI engineering

## Learning outcomes
- Explain AI, machine learning, deep learning, generative AI and modern LLMs.
- Understand the mathematics and algorithms behind common AI systems.
- Build practical ML and deep-learning models.
- Explain transformers, attention, tokenization and embeddings.
- Build LLM applications using prompting, RAG and evaluation.
- Design tool-using AI agents, workflows and MCP integrations.
- Run, adapt, evaluate and deploy local and hosted models.

## Daily 60-minute format
| Time | Activity |
|---|---|
| 0–10 min | Recap and learning objectives |
| 10–25 min | Core concepts and terminology |
| 25–40 min | Architecture, diagrams and worked examples |
| 40–55 min | Hands-on coding or experiment |
| 55–60 min | Quiz, takeaways and next lesson |

Every lesson should contain objectives, prerequisites, detailed explanations, diagrams where useful, runnable code/labs, debugging tips, five review questions and further reading.

## Curriculum roadmap

### Phase 1 — AI foundations
AI history; AI vs ML vs deep learning vs generative AI; classification, regression, clustering and generation; data, features, labels, training, inference and evaluation; responsible AI.

### Phase 2 — Mathematics for AI
Linear algebra; vectors, matrices and tensors; derivatives, gradients and chain rule; probability and Bayes' theorem; statistics and distributions; loss functions, optimization and gradient descent.

### Phase 3 — Machine learning
Data preparation and feature engineering; linear/logistic regression; decision trees, random forests and boosting; k-nearest neighbors, SVM and naive Bayes; clustering; dimensionality reduction; train/validation/test splits; metrics, overfitting and regularization.

### Phase 4 — Deep learning
Neurons and multilayer networks; forward propagation and backpropagation; activations, losses and optimizers; CNNs, RNNs and LSTMs; GPU training, batches, epochs and checkpoints; model evaluation and debugging.

### Phase 5 — Transformers and generative AI
Tokenization and embeddings; attention and self-attention; transformer encoder/decoder architecture; positional information and context windows; pretraining and instruction tuning; LLM capabilities and limitations; multimodal text, image and audio models.

### Phase 6 — LLM application engineering
Prompt design and structured outputs; embedding models and semantic search; vector databases; RAG; chunking and reranking; tool/function calling; evaluation, tracing, guardrails, cost and latency.

### Phase 7 — AI agents and orchestration
Agent loop; planning, tools, state and memory; workflow and multi-agent patterns; MCP servers, tools and resources; context engineering; reliability, security and human approval; agent testing.

### Phase 8 — Reusable agent architectures
Agent design principles; single-agent and workflow patterns; router/specialist, planner–executor–verifier and supervisor/worker designs; durable state machines; typed tools; memory boundaries; security and policy enforcement; evaluation, observability, reliability and human approval; reusable agent specifications and reference implementations.

### Phase 9 — Model adaptation and deployment
Open-weight and hosted models; local inference; fine-tuning, LoRA and quantization; datasets and experiment tracking; APIs, containers and deployment; monitoring and lifecycle.

### Phase 10 — Capstone projects
Build a small ML application, document Q&A RAG application, tool-using AI agent, MCP-connected assistant and an evaluated end-to-end AI application.

## Repository structure
```text
ai-basics-masterclass/
├── README.md
├── LICENSE
├── LICENSE-CONTENT
├── curriculum/ (roadmap and progress)
├── daily-lessons/ (one folder per day)
├── fundamentals/
├── mathematics/
├── machine-learning/
├── deep-learning/
├── transformers/
├── llm/
├── rag/
├── ai-agents/
├── agent-architectures/ (patterns, templates and examples)
├── mcp/
├── hands-on-projects/
├── exercises/
└── resources/
```

## Progress
See [curriculum/progress.md](curriculum/progress.md). A lesson is complete when its notes, examples, lab and review questions are committed.

## Licensing
- Code and software examples: [MIT](LICENSE).
- Original educational prose and diagrams: [CC BY 4.0](LICENSE-CONTENT).
- Third-party materials remain under their original licenses; retain attribution and follow their terms.

Prefer primary sources, official specifications and original research. Label assumptions, distinguish conceptual examples from production-ready code, cite external material, and never commit API keys, credentials or private datasets.
