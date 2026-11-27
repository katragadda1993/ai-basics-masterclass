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
