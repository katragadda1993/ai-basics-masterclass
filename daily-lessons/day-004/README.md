# Day 004 — Functions, Derivatives, Gradients and the Chain Rule

**Duration:** 60 minutes  
**Level:** Beginner → intermediate  
**Phase:** 2 — Mathematics for AI  
**Previous lesson:** [Day 003 — Vectors, Matrices and Tensors](../day-003/README.md)

## Learning objectives

By the end of this session, you should be able to:

1. Explain a function as a mapping from inputs to outputs.
2. Interpret a derivative as an instantaneous rate of change and local slope.
3. Calculate simple derivatives using power, sum and constant-multiple rules.
4. Explain partial derivatives and assemble them into a gradient vector.
5. Apply the chain rule to a composed function.
6. Explain how backpropagation uses the chain rule on a computational graph.
7. Calculate a gradient-descent update for a simple regression model.
8. Detect common optimization issues such as an excessive learning rate and an incorrect gradient sign.

## 60-minute lesson plan

| Time | Activity |
|---|---|
| 0–5 min | Recall vectors, matrices and model parameters from Day 003 |
| 5–15 min | Functions and derivatives: what changes when an input changes? |
| 15–25 min | Partial derivatives and gradients |
| 25–35 min | Chain rule and computational graphs |
| 35–45 min | Gradient descent and a worked regression example |
| 45–55 min | Python lab: numerical gradients and training a tiny model |
| 55–58 min | Connect gradients to backpropagation and agent engineering |
| 58–60 min | Five review questions |

## 1. Why calculus matters in AI

In Day 003, we saw that a dense neural-network layer performs weighted sums such as:

\[
z = Wx+b
\]

The model has parameters W and b. How should we change them so that predictions become more accurate?

We need three things:

1. A **loss function** that measures how wrong the model is.
2. A **derivative or gradient** that tells us how the loss changes when parameters change.
3. An **optimization algorithm** that uses this information to update parameters.

Calculus provides derivatives and gradients. Optimization uses them to improve model parameters.

Not every AI algorithm uses gradient descent. Decision trees and many other methods use different optimization procedures. But gradients are foundational to training neural networks and most modern deep-learning models.

## 2. Functions: inputs mapped to outputs

A function assigns an output to each valid input.

\[
y=f(x)
\]

For example:

\[
f(x)=x^2
\]

If x=3, then f(3)=9. If x=4, then f(4)=16.

In machine learning, a model is a parameterized function:

\[
\hat{y}=f_\theta(x)
\]

- x: input features
- θ: model parameters (weights and biases)
- ŷ: model prediction

A loss function compares predictions with targets:

\[
L(\theta)=\frac{1}{n}\sum_{i=1}^{n}(\hat{y}^{(i)}-y^{(i)})^2
\]

Training tries to find parameter values that make this loss small.

## 3. Derivatives: how fast does the output change?

For a function y=f(x), the derivative measures its instantaneous rate of change with respect to x.

\[
f'(x)=\frac{df}{dx}
=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}
\]

The expression before the limit is the slope between two nearby points (a secant slope). As h approaches zero, it becomes the tangent slope at x, when the derivative exists.

### 3.1 A simple example

\[
f(x)=x^2
\]

Using the power rule:

\[
\frac{d}{dx}x^n=nx^{n-1}
\]

Therefore:

\[
f'(x)=2x
\]

At x=3:

\[
f'(3)=6
\]

Interpretation: near x=3, increasing x by a small amount of 0.01 increases f(x) by approximately 6 × 0.01 = 0.06. This is a local approximation, not an exact prediction for a large input change.

### 3.2 Derivative rules

| Rule | Formula | Example |
|---|---|---|
| Constant | \(\frac{d}{dx}c=0\) | derivative of 7 is 0 |
| Power | \(\frac{d}{dx}x^n=nx^{n-1}\) | derivative of x³ is 3x² |
| Constant multiple | \(\frac{d}{dx}[cf]=cf'\) | derivative of 4x² is 8x |
| Sum | \((f+g)'=f'+g'\) | derivative of x²+3x is 2x+3 |

Worked example:

\[
f(x)=3x^2+4x+7
\]

\[
f'(x)=6x+4
\]

At x=2, the derivative is 16.

### 3.3 Numerical derivative

When an analytical derivative is inconvenient, approximate it using a small value ε:

\[
f'(x)\approx
\frac{f(x+\epsilon)-f(x-\epsilon)}{2\epsilon}
\]

This is a central finite-difference approximation. A very large epsilon gives a coarse estimate; an extremely tiny epsilon can amplify floating-point round-off. Choose epsilon for the scale and numeric precision of the problem.

The Python lab compares analytical gradients with finite-difference estimates. This is a useful way to catch mistakes in hand-written gradient code.

## 4. Partial derivatives and gradients

A model usually has many parameters, not just one. Consider:

\[
f(x,y)=x^2+3y^2
\]

A **partial derivative** measures change with respect to one variable while holding the other fixed.

\[
\frac{\partial f}{\partial x}=2x
\qquad
\frac{\partial f}{\partial y}=6y
\]

At x=2 and y=1:

\[
\frac{\partial f}{\partial x}=4,
\qquad
\frac{\partial f}{\partial y}=6
\]

The **gradient** collects all partial derivatives into a vector:

\[
\nabla f(x,y)=
\begin{bmatrix}
\partial f/\partial x\\
\partial f/\partial y
\end{bmatrix}
=
\begin{bmatrix}
2x\\
6y
\end{bmatrix}
\]

At (2,1), the gradient is [4,6].

The gradient points in the direction of steepest local increase under the usual Euclidean geometry. The negative gradient points in the direction of steepest local decrease.

If a model has parameters θ=[w₁,w₂,...,wₘ], then:

\[
\nabla_\theta L=
\left[
\frac{\partial L}{\partial w_1},
\frac{\partial L}{\partial w_2},
\ldots,
\frac{\partial L}{\partial w_m}
\right]
\]

This connects to Day 003: a gradient normally has the same shape as the parameter with respect to which it is calculated.

## 5. Chain rule: the foundation of backpropagation

Many models are compositions of functions. If:

\[
y=f(u),\qquad u=g(x)
\]

then:

\[
\frac{dy}{dx}
=
\frac{dy}{du}\cdot\frac{du}{dx}
\]

This is the chain rule.

### 5.1 Worked example

Let:

\[
u=3x+2,\qquad y=u^2
\]

First:

\[
\frac{dy}{du}=2u
\]

Then:

\[
\frac{du}{dx}=3
\]

Therefore:

\[
\frac{dy}{dx}=2u\cdot3=6(3x+2)=18x+12
\]

At x=1, the derivative is 30.

We can also expand the original function:

\[
y=(3x+2)^2=9x^2+12x+4
\]

Differentiating gives 18x+12, confirming the chain-rule result.

### 5.2 Computational graph

A computational graph represents a calculation as connected operations:

~~~text
x → u = 3x + 2 → y = u²
~~~

Forward pass:
- Calculate u from x.
- Calculate y from u.

Backward pass:
- Start with dy/dy=1.
- Calculate dy/du=2u.
- Calculate du/dx=3.
- Multiply local derivatives to obtain dy/dx.

For x=1, u=5, so dy/du=10, and the final derivative is 10 × 3 = 30.

### 5.3 Why backpropagation is efficient

A neural network can contain millions or billions of parameters. Computing each parameter's derivative independently by repeatedly evaluating the whole model would be very expensive.

Backpropagation applies the chain rule in reverse through the computation graph and reuses intermediate results. In a typical scalar-loss training setup, it efficiently computes gradients for all trainable parameters.

For a node that contributes to multiple downstream paths, gradient contributions are accumulated. This is important in branching computational graphs.

## 6. Gradient descent: use gradients to improve a model

The basic gradient-descent update is:

\[
\theta_{t+1}=\theta_t-\eta\nabla_\theta L(\theta_t)
\]

Where:
- θ: current model parameters
- η: learning rate (step size)
- ∇L: gradient of the loss
- t: current update step

The minus sign matters: we move against the gradient to reduce the loss locally.

### 6.1 A one-parameter example

Let:

\[
L(w)=(w-3)^2
\]

Then:

\[
\frac{dL}{dw}=2(w-3)
\]

Start with w=0 and learning rate η=0.1:

\[
\frac{dL}{dw}=2(0-3)=-6
\]

\[
w_{new}=0-0.1(-6)=0.6
\]

The loss changes from:

\[
L(0)=9
\]

to:

\[
L(0.6)=5.76
\]

The loss decreased. Repeating the update moves w toward 3, the minimum of this function.

### 6.2 Learning-rate trade-offs

- Too small: training may make very slow progress.
- Reasonable: updates can make steady progress.
- Too large: updates may overshoot, oscillate or diverge.

A learning rate is not a cosmetic setting. It changes optimization behavior.

### 6.3 Batch, stochastic and mini-batch gradient descent

- **Batch gradient descent:** compute a gradient using the entire training dataset for each update.
- **Stochastic gradient descent (SGD):** compute an update using one training example at a time.
- **Mini-batch gradient descent:** compute each update using a small subset of examples.

Mini-batches are common in deep learning because they balance computation efficiency, memory usage and noisy gradient estimates.

## 7. Worked example: train a tiny linear regression model

Suppose our dataset is:

\[
x=[1,2],\qquad y=[2,4]
\]

Use a model:

\[
\hat y=wx+b
\]

Mean squared error:

\[
L(w,b)=\frac{1}{2}\sum_{i=1}^{2}(wx_i+b-y_i)^2
\]

The factor 1/2 is the average over two examples. The gradient formulas are:

\[
\frac{\partial L}{\partial w}
=\frac{2}{2}\sum_{i=1}^{2}(wx_i+b-y_i)x_i
\]

\[
\frac{\partial L}{\partial b}
=\frac{2}{2}\sum_{i=1}^{2}(wx_i+b-y_i)
\]

Start with w=0, b=0. Predictions are [0,0] and errors are [-2,-4].

Loss:

\[
L=\frac{(-2)^2+(-4)^2}{2}=10
\]

Gradients:

\[
\frac{\partial L}{\partial w}
=(-2)(1)+(-4)(2)=-10
\]

\[
\frac{\partial L}{\partial b}=(-2)+(-4)=-6
\]

With learning rate 0.1:

\[
w_{new}=0-0.1(-10)=1
\]

\[
b_{new}=0-0.1(-6)=0.6
\]

New predictions are [1.6,2.6]. The new loss is:

\[
L_{new}=\frac{(1.6-2)^2+(2.6-4)^2}{2}=1.06
\]

The loss fell from 10 to 1.06 after one update. This demonstrates one successful step; it does not guarantee that every update decreases loss for every learning rate.

## 8. Hands-on lab — implement gradients and gradient descent

### Setup

This lab uses only the Python standard library:

~~~bash
cd daily-lessons/day-004
python lab.py
~~~

### Lab tasks

1. Implement the derivative of a quadratic function.
2. Approximate a derivative using central finite differences.
3. Implement MSE loss and analytical gradients for a one-feature linear model.
4. Compare analytical gradients against numerical estimates.
5. Train the tiny regression model with gradient descent.
6. Confirm that loss falls and parameters move toward the expected solution.

Expected key output:

~~~text
quadratic derivative at x=3: 6.0
finite difference check: PASS
chain rule derivative at x=1: 30.0
initial loss: 10.0
gradient dw: -10.0
gradient db: -6.0
parameters after one step: w=1.0, b=0.6
loss after one step: 1.06
training loss decreased: PASS
all checks passed
~~~

### Exercises

1. Change the learning rate to 0.01. How many iterations are needed to reach a similar loss?
2. Change the learning rate to 1.0. Observe whether the loss decreases consistently.
3. Change the dataset to x=[1,2,3], y=[3,5,7]. What values of w and b fit it exactly?
4. Intentionally reverse the gradient-descent sign. Explain why the loss grows for this example.
5. Compare analytical derivatives with finite-difference approximations for several epsilon values.

## 9. Optional PyTorch experiment

After understanding the manual lab, try automatic differentiation:

~~~python
import torch

w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

x = torch.tensor([1.0, 2.0])
y = torch.tensor([2.0, 4.0])

prediction = w * x + b
loss = ((prediction - y) ** 2).mean()

loss.backward()

print("loss:", loss.item())
print("dw:", w.grad.item())
print("db:", b.grad.item())
~~~

Expected gradients are approximately dw=-10.0 and db=-6.0.

PyTorch records the operations that contribute to the loss. Calling backward computes gradients through that graph using automatic differentiation. In ordinary training loops, clear or reset accumulated gradients between updates, for example with optimizer.zero_grad().

## 10. Agent engineering connection

Gradients optimize differentiable model parameters. They do not directly decide whether a workflow should call a database, wait for human approval, retry an API or deploy a patch.

For production agent systems, distinguish two optimization layers:

1. **Model optimization:** train or fine-tune a neural model using losses and gradients.
2. **System optimization:** improve routing, prompts, tool selection, workflow steps and retry policies using evaluations, traces, controlled experiments and cost/latency measurements.

A reusable agent architecture should keep these concerns separate:

~~~text
Request
  → deterministic policy and workflow
  → model inference (where useful)
  → typed tool execution
  → independent verification
  → evaluation and telemetry
~~~

Do not use a model's confidence or a similarity score as an authorization decision. Do not let an agent alter its own production policy based only on generated reasoning. Measure changes against a fixed evaluation set and require approval for high-impact changes.

## 11. Key terminology

| Term | Meaning |
|---|---|
| Function | Mapping from inputs to outputs |
| Parameter | A value the model can learn or adjust |
| Derivative | Instantaneous rate of change of a scalar function |
| Partial derivative | Derivative with respect to one variable while holding others fixed |
| Gradient | Vector of partial derivatives |
| Chain rule | Rule for differentiating composed functions |
| Computational graph | Directed graph of operations and dependencies |
| Backpropagation | Efficient reverse application of the chain rule |
| Loss function | Numerical measure of model error |
| Gradient descent | Parameter update against the loss gradient |
| Learning rate | Step size used in a gradient update |
| Epoch | One complete pass through the training dataset |
| Batch | Subset of examples used for one training step |
| Automatic differentiation | Systematic computation of derivatives by applying chain rules to program operations |
| Numerical gradient check | Comparison of an analytical gradient with a finite-difference estimate |

## 12. Five review questions

1. What does the derivative of a function tell us, and what is the derivative of f(x)=3x²+4x+7?
2. What is the difference between a partial derivative and a gradient?
3. For u=3x+2 and y=u², use the chain rule to calculate dy/dx.
4. Why does gradient descent subtract the gradient, and what can happen when the learning rate is too large?
5. How does backpropagation reuse the chain rule through a computational graph, and why is this different from an agent's workflow orchestration?

## Takeaways

- A model is a parameterized function; training tries to reduce a loss.
- Derivatives describe local sensitivity to input or parameter changes.
- Gradients combine partial derivatives for all parameters.
- The chain rule is the mathematical foundation of backpropagation.
- Gradient descent updates parameters in the negative-gradient direction.
- Learning rate, batch strategy and numerical precision affect training behavior.
- Automatic differentiation computes gradients through recorded operations, but numerical checks remain valuable.
- Model training and production agent orchestration are different engineering concerns.

## Further reading

1. MIT OpenCourseWare — [Single Variable Calculus](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/).
2. MIT OpenCourseWare — [Gradient Descent: Downhill to a Minimum](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/resources/lecture-22-gradient-descent-downhill-to-a-minimum/).
3. PyTorch — [Automatic Differentiation with torch.autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html).
4. PyTorch — [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html).
5. Stanford CS229 — [Machine Learning course materials](https://cs229.stanford.edu/).

## Completion checklist

- [ ] Run the Python lab.
- [ ] Explain the derivative and gradient without notes.
- [ ] Work through the chain-rule example by hand.
- [ ] Compare analytical and numerical gradients.
- [ ] Complete at least three gradient-descent experiments.
- [ ] Answer all five review questions.
- [ ] Explain the difference between optimizing model parameters and optimizing an agent workflow.
