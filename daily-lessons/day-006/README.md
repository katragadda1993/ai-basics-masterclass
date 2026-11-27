# Day 006 — Information Theory: Entropy, Cross-Entropy and KL Divergence

**Duration:** 60 minutes  
**Level:** Beginner → intermediate  
**Phase:** 2 — Mathematics for AI  
**Previous lesson:** [Day 005 — Probability, Statistics and Bayes' Theorem](../day-005/README.md)

## Learning objectives

By the end of this session, you should be able to:
1. Explain information content and why rare events carry more information.
2. Calculate Shannon entropy in bits and nats.
3. Distinguish target entropy, cross-entropy and KL divergence.
4. Explain why cross-entropy trains classifiers and language models.
5. Convert logits to probabilities using numerically stable softmax.
6. Explain the softmax-plus-cross-entropy gradient.
7. Calculate information gain for a decision-tree split.
8. Use predictive entropy as one signal in agent abstention without confusing it with correctness or authorization.

## 60-minute lesson plan

| Time | Activity |
|---|---|
| 0–5 min | Recall probability, distributions and Bayes |
| 5–15 min | Information content and Shannon entropy |
| 15–27 min | Cross-entropy and KL divergence |
| 27–38 min | Softmax, logits and classification loss |
| 38–45 min | Information gain and next-token prediction |
| 45–55 min | Python lab |
| 55–58 min | Entropy-aware agent routing |
| 58–60 min | Five review questions |

## 1. Why information theory matters in AI

Day 005 introduced probability distributions and Bayes' theorem. Today we ask: how uncertain is a distribution, and how costly is it when a model predicts the wrong distribution?

Information theory quantifies surprise, uncertainty, prediction cost and information gained from observations. These ideas appear in decision trees, neural-network classification, language-model training, compression and probabilistic evaluation.

## 2. Information content: rare events are more surprising

For an event with probability \(p(x)\), its self-information is:

\[
I(x)=-\log_2 p(x)
\]

The unit is the **bit** when the logarithm base is 2.

| Event probability | Information |
|---:|---:|
| 1/2 | 1 bit |
| 1/4 | 2 bits |
| 1/8 | 3 bits |
| 1/16 | 4 bits |

Halving an event's probability adds one bit of surprise. An event with probability 1 has zero surprise; an event approaching probability zero has increasingly large surprise.

With natural logarithms, information is measured in **nats**:

\[
I(x)=-\ln p(x), \qquad 1\text{ nat}=\log_2(e)\text{ bits}
\]

## 3. Shannon entropy: uncertainty before observing an outcome

For a discrete distribution \(P\):

\[
H(P)=-\sum_i p_i\log_2 p_i
\]

Terms with zero probability contribute zero by continuity.

Entropy is the expected self-information of an outcome. It is high when probability is spread across many plausible outcomes and low when probability is concentrated.

### Example A — Fair coin

\[
P=[0.5,0.5]
\]

\[
H(P)=-(0.5\log_2 0.5+0.5\log_2 0.5)=1\text{ bit}
\]

### Example B — Biased coin

\[
P=[0.9,0.1],\qquad H(P)\approx0.469\text{ bits}
\]

The fair coin has greater uncertainty than the biased coin.

For a distribution over \(K\) possible outcomes, maximum entropy is:

\[
H_{\max}=\log_2 K
\]

It occurs for a uniform distribution. A deterministic distribution such as \([1,0,0,0]\) has zero entropy.

**Important:** entropy measures uncertainty in a distribution, not whether a prediction is correct. A model can be confidently wrong and have low predictive entropy.

## 4. Cross-entropy: how well does a predicted distribution match the target?

Let \(P\) be the target distribution and \(Q\) the model's predicted distribution.

\[
H(P,Q)=-\sum_i p_i\log q_i
\]

This is the average negative log probability that model \(Q\) assigns to outcomes drawn from \(P\).

For a one-hot classification target, only the true class has target probability 1. The per-example loss becomes:

\[
L=-\log q_{\text{true class}}
\]

A model is penalized when it assigns low probability to the correct class.

### Example C — Compare predictions

Suppose:

\[
P=[0.8,0.2], \qquad Q=[0.6,0.4]
\]

Then:

\[
H(P,Q)=-0.8\log_2(0.6)-0.2\log_2(0.4)
\approx0.854\text{ bits}
\]

A perfect prediction \(Q=P\) has cross-entropy equal to target entropy:

\[
H(P,P)=H(P)\approx0.722\text{ bits}
\]

For a one-hot target, assigning probability 0.9 to the correct class gives loss about 0.105 nats; assigning 0.001 gives about 6.908 nats. Confident mistakes are expensive.

## 5. KL divergence: the extra cost of using the wrong distribution

Kullback–Leibler divergence from target \(P\) to model \(Q\) is:

\[
D_{KL}(P\parallel Q)=\sum_i p_i\log\frac{p_i}{q_i}
\]

The same logarithm base determines whether units are bits or nats.

The key identity is:

\[
H(P,Q)=H(P)+D_{KL}(P\parallel Q)
\]

- \(H(P)\): uncertainty intrinsic to the target distribution.
- \(D_{KL}(P\parallel Q)\): extra cost from using \(Q\) instead of \(P\).
- \(H(P,Q)\): total expected coding/prediction cost.

For \(P=[0.8,0.2]\) and \(Q=[0.6,0.4]\):

\[
D_{KL}(P\parallel Q)\approx0.13203\text{ bits}
\]

### Properties and cautions

1. KL divergence is non-negative for valid distributions.
2. It is zero when the distributions match on the support of \(P\).
3. It is not symmetric: \(D_{KL}(P\parallel Q)\ne D_{KL}(Q\parallel P)\).
4. It is not a metric: it does not satisfy symmetry or the usual triangle inequality.
5. If \(p_i>0\) but \(q_i=0\), divergence is infinite.

## 6. Logits and softmax

Neural networks commonly output **logits**, unnormalized real-valued scores. They are not probabilities and need not sum to one.

For logits \(z_1,\ldots,z_K\), softmax gives:

\[
q_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
\]

For \(z=[2,1,0]\), the probabilities are approximately \([0.6652,0.2447,0.0900]\).

### Numerical stability

Exponentials can overflow for large logits. Softmax is invariant to adding the same constant to every logit. Subtract the maximum:

\[
q_i=\frac{e^{z_i-m}}{\sum_j e^{z_j-m}},\qquad m=\max_j z_j
\]

For loss calculations, use log-sum-exp:

\[
\log\sum_j e^{z_j}=m+\log\sum_j e^{z_j-m}
\]

Production libraries use stable fused operations.

## 7. Softmax plus cross-entropy gradient

For a one-hot target vector \(y\), predicted softmax vector \(q\), and logits \(z\):

\[
L=-\sum_i y_i\log q_i
\]

The derivative with respect to each logit is:

\[
\frac{\partial L}{\partial z_i}=q_i-y_i
\]

This compact result tells the model how each class score should change. Backpropagation propagates these gradients through model parameters, connecting this lesson to the chain rule from Day 004.

PyTorch example:

~~~python
import torch
import torch.nn as nn

logits = torch.tensor([[2.0, 1.0, 0.0]])
target = torch.tensor([0])
loss = nn.CrossEntropyLoss()(logits, target)
print(loss.item())  # approximately 0.4076 nats
~~~

**Important:** PyTorch CrossEntropyLoss expects raw logits and class-index targets (or valid target probability distributions), not probabilities already passed through softmax. It combines log-softmax and negative log likelihood in a numerically stable implementation.

## 8. Language models: predicting the next token

A causal language model predicts the next token conditioned on previous tokens:

\[
P(x_t\mid x_1,\ldots,x_{t-1})
\]

For a sequence, negative log-likelihood is:

\[
L=-\sum_{t=1}^{T}\log P_\theta(x_t\mid x_{<t})
\]

Training commonly averages over valid target tokens. Padding and ignored positions must be excluded correctly.

~~~text
Input tokens
    |
    v
Embedding + positional information
    |
    v
Transformer layers
    |
    v
Logits for each vocabulary token
    |
    v
Cross-entropy against next-token target
    |
    v
Loss -> backpropagation -> optimizer update
~~~

The model learns parameters that assign higher probability to observed next-token targets.

**Perplexity** is derived from average negative log-likelihood:

\[
\text{Perplexity}=\exp(\text{average NLL in nats})
\]

Lower perplexity means better average next-token prediction on the same evaluation data and tokenization. It does not by itself measure factuality, reasoning or usefulness.

## 9. Information gain: decision-tree feature selection

Decision trees can choose a split by measuring how much it reduces class uncertainty:

\[
IG=H(\text{parent})-\sum_j w_jH(\text{child}_j)
\]

Here \(w_j\) is the fraction of samples in child \(j\).

Suppose a balanced parent has entropy 1 bit. A feature divides it into two equally sized children, each 90% one class and 10% the other:

\[
H(\text{child})=H([0.9,0.1])\approx0.469
\]

\[
IG=1-0.469=0.531\text{ bits}
\]

The feature reduces uncertainty by about 0.531 bits in this simplified example. Real tree training must also control overfitting and validate generalization.

## 10. Practical lab

The standard-library Python lab in [lab.py](lab.py) implements:

1. Shannon entropy in bits and nats.
2. Cross-entropy between target and predicted distributions.
3. KL divergence and the identity \(H(P,Q)=H(P)+D_{KL}(P\parallel Q)\).
4. Numerically stable softmax from logits.
5. Cross-entropy from logits and a target class.
6. Softmax-cross-entropy gradient \(q-y\).
7. Decision-tree information gain.
8. Input validation and deterministic assertions.

Run:

~~~bash
cd daily-lessons/day-006
python lab.py
~~~

No third-party packages are required.

Expected values:
- Entropy of [0.5, 0.5]: 1 bit.
- Entropy of [0.9, 0.1]: about 0.469 bits.
- Cross-entropy for P=[0.8,0.2], Q=[0.6,0.4]: about 0.854 bits.
- KL divergence for the same distributions: about 0.132 bits.
- Softmax of [2,1,0]: approximately [0.6652,0.2447,0.0900].
- Cross-entropy for target class 0: about 0.4076 nats.
- Information gain for the balanced 90/10 split: about 0.531 bits.

## 11. Agent architecture connection: entropy-aware routing with abstention

Day 005 introduced probability-aware specialist routing. Today we add predictive entropy as a measure of how spread out the classifier's output distribution is:

\[
H(q)=-\sum_i q_i\log q_i
\]

Consider three specialist domains: DMA, PCIe and Audio.

Clear distribution: \(q_A=[0.95,0.03,0.02]\)

Ambiguous distribution: \(q_B=[0.38,0.34,0.28]\)

The second distribution is more ambiguous and is a stronger candidate for abstention or human triage.

### Reference architecture

~~~mermaid
flowchart TD
  E[Diagnostic event] --> V[Schema, identity and provenance validation]
  V --> M[Versioned classifier]
  M --> P[Calibrated class probabilities]
  P --> U[Entropy, margin and OOD checks]
  U --> R[Deterministic routing policy]
  R -->|Clear and in distribution| S[Read-only specialist]
  R -->|Ambiguous or OOD| H[Human triage]
  R -->|High-impact operation| A[Approval workflow]
  S --> Q[Independent evidence verifier]
  Q --> O[Audit record and telemetry]
  H --> O
  A --> O
~~~

### Entropy is not a complete uncertainty detector

- **Predictive entropy:** spread in the predicted class distribution.
- **Epistemic uncertainty:** uncertainty in the model itself, sometimes approximated using ensembles or Bayesian methods.
- **Aleatoric uncertainty:** inherent noise or ambiguity in data.
- **Out-of-distribution (OOD):** input differs from the model's training/development distribution.

A model can be confidently wrong on an OOD input. Low entropy is not proof of correctness.

### Production routing policy

1. Validate event schema and source identity.
2. Calculate probabilities using a versioned model.
3. Check calibration, entropy, top-two margin and OOD indicators.
4. Apply thresholds selected on representative validation data.
5. Route clear, low-risk cases to read-only specialists.
6. Abstain and request human triage for ambiguous or OOD cases.
7. Require independent authorization and approval for consequential actions.
8. Record model version, feature schema, decision, evidence references and approval identity.

### Engineering trade-offs

| Concern | Production design |
|---|---|
| Latency | Use a cheap deterministic filter before a larger model where appropriate |
| Cost | Avoid invoking several specialists when one bounded route is enough |
| Reliability | Use explicit states, timeouts, idempotency and bounded retries |
| Memory | Persist case state and evidence references outside the prompt |
| Security | Treat logs, telemetry and retrieved documents as untrusted inputs |
| Evaluation | Track cross-entropy, calibration, entropy, abstention, false routing and task success |
| Human control | Approval remains independent of model confidence |

The router is a decision-support component. It does not gain permission to execute tools because its entropy is low.

## 12. Key terminology

| Term | Meaning |
|---|---|
| Self-information | Surprise of an individual event, negative log probability |
| Shannon entropy | Expected self-information of a distribution |
| Bit / nat | Information units using log base 2 / natural log |
| Cross-entropy | Expected negative log probability under a predicted distribution |
| KL divergence | Relative entropy \(D_{KL}(P\parallel Q)\) |
| Logit | Unnormalized model score |
| Softmax | Converts logits into a categorical probability distribution |
| Log-sum-exp | Stable logarithm of a sum of exponentials |
| Negative log-likelihood | Negative log probability assigned to observed targets |
| Information gain | Reduction in entropy after a feature/split |
| Perplexity | Exponential of average next-token negative log-likelihood |
| Predictive entropy | Entropy of a model's predictive distribution |
| Abstention | Choosing not to automate a decision |
| OOD | Out-of-distribution input |

## 13. Five review questions

1. Calculate the entropy in bits of [0.5, 0.5] and [0.9, 0.1]. Which distribution is more uncertain and why?
2. If P=[0.8,0.2] and Q=[0.6,0.4], explain the relationship between cross-entropy, entropy and KL divergence.
3. Why does PyTorch CrossEntropyLoss expect logits rather than probabilities already passed through softmax?
4. A three-class agent router produces [0.38, 0.34, 0.28]. Why might it abstain, and why is entropy alone insufficient to establish whether the input is safe or in-distribution?
5. Explain how next-token cross-entropy trains a language model and how it connects to the chain rule and backpropagation from Day 004.

## Takeaways

- Entropy quantifies uncertainty in a distribution.
- Cross-entropy measures the cost of predicting target outcomes with a model distribution.
- KL divergence is the extra expected coding cost relative to the target distribution.
- Softmax converts logits to probabilities; stable log-sum-exp avoids numerical overflow.
- Softmax plus cross-entropy produces the simple logit gradient \(q-y\).
- LLM next-token training minimizes cross-entropy over target tokens.
- Information gain measures uncertainty reduction and helps decision trees choose splits.
- Entropy can support agent abstention, but it is not a correctness guarantee, an OOD detector by itself, or an authorization mechanism.

## Further reading

- [Shannon (1948), A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)
- [PyTorch — CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.CrossEntropyLoss.html)
- [SciPy — Shannon entropy and relative entropy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html)
- [scikit-learn — Log loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)
- [scikit-learn — Decision trees](https://scikit-learn.org/stable/modules/tree.html)

## Completion checklist

- [ ] Calculate entropy of a fair and biased coin by hand.
- [ ] Verify the cross-entropy/KL identity with the lab.
- [ ] Explain why logits are not probabilities.
- [ ] Run stable softmax and gradient checks.
- [ ] Explain next-token cross-entropy in your own words.
- [ ] Explain when an agent router should abstain.
- [ ] Answer all five review questions.
