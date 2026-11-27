# Day 001 — What Is Artificial Intelligence?

**Duration:** 60 minutes  
**Level:** Beginner  
**Phase:** AI foundations

## Learning objectives

By the end of this session, you should be able to:
- Define artificial intelligence in practical terms.
- Distinguish AI, machine learning, deep learning and generative AI.
- Explain training versus inference.
- Identify a problem that is suitable for an AI approach.

## 60-minute lesson plan

| Time | Activity |
|---|---|
| 0–5 min | What you already know about AI |
| 5–20 min | Definition, history and examples |
| 20–35 min | AI vs ML vs deep learning vs generative AI |
| 35–45 min | Training and inference lifecycle |
| 45–55 min | Mini exercise |
| 55–60 min | Quiz and recap |

## 1. What is AI?

Artificial intelligence is the field of building computer systems that perform tasks which normally require capabilities associated with human intelligence: perception, language understanding, reasoning, prediction, planning and decision support.

AI is not one single algorithm. It is a broad collection of methods, from hand-written rules to statistical learning and neural networks.

Examples:
- Spam filtering: classify an email as spam or legitimate.
- Speech recognition: convert audio into text.
- Image recognition: identify objects in a picture.
- Recommendation: rank products or videos.
- Generative AI: create text, images, audio or code from a prompt.

## 2. AI, ML, deep learning and generative AI

Think of these as overlapping categories:

- **Artificial intelligence (AI):** the broad field of intelligent computer behavior.
- **Machine learning (ML):** methods that learn patterns from data instead of relying only on manually written rules.
- **Deep learning (DL):** machine learning based on multi-layer neural networks.
- **Generative AI:** systems that generate new content, often using deep-learning models such as transformers or diffusion models.

A rule-based chess program can be AI without being machine learning. A neural-network image classifier is both ML and deep learning. A large language model is a deep-learning model used for generative AI.

## 3. Training and inference

**Training** is the process of adjusting a model's parameters using examples and an objective (loss) so that its outputs improve.

**Inference** is using the trained model to make a prediction or generate an output for new input data.

Typical lifecycle:

```text
Collect data → Prepare data → Train model → Evaluate model
                                      ↓
User input → Preprocess → Inference → Output
```

Training can be computationally expensive. Inference is usually the deployed model's day-to-day operation, although its cost and latency depend on model size, hardware and workload.

## 4. A simple example: spam detection

Input: an email and its text/features.  
Output: spam or not spam.

A supervised-learning dataset contains examples with labels. The model learns statistical patterns from the examples. During inference, it predicts a label for a new email. The prediction is not guaranteed to be correct; evaluation measures how often and in what ways it fails.

## 5. Mini exercise

For each task, identify whether it is primarily classification, regression, clustering or generation:

1. Predict tomorrow's temperature.
2. Decide whether a transaction is fraudulent.
3. Group customers by similar purchasing behavior without predefined labels.
4. Generate a draft email from a short instruction.

**Suggested answers:** 1. Regression; 2. Classification; 3. Clustering; 4. Generation.

## 6. Knowledge check

1. Is every AI system a machine-learning system? Why?
2. What is the difference between training and inference?
3. How does deep learning relate to machine learning?
4. Give one example of generative AI.
5. Why should a model be evaluated before deployment?

## 7. Key terminology

| Term | Meaning |
|---|---|
| Model | A parameterized system that maps inputs to outputs |
| Dataset | A collection of examples used for analysis or learning |
| Feature | An input variable or measurable property |
| Label | The target answer in supervised learning |
| Training | Learning model parameters from data |
| Inference | Producing outputs using a trained model |
| Parameter | A learned internal value of a model |
| Evaluation | Measuring model quality on defined data and metrics |

## Takeaways

- AI is a broad field, not one model or product.
- ML learns patterns from data; deep learning uses multi-layer neural networks.
- Generative AI creates content.
- Training builds or adapts a model; inference uses it.
- Every AI solution should be evaluated against its intended task.

## Further reading

- [Artificial intelligence — Stanford Encyclopedia of Philosophy](https://plato.stanford.edu/entries/artificial-intelligence/)
- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
- [Elements of AI](https://www.elementsofai.com/)

## Completion checklist

- [ ] Read the lesson
- [ ] Explain the four AI terms without notes
- [ ] Complete the mini exercise
- [ ] Answer the five knowledge-check questions
- [ ] Write down one AI use case you want to build
