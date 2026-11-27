# Day 003 — Vectors, Matrices and Tensors: The Mathematics Behind AI

**Duration:** 60 minutes  
**Level:** Beginner → intermediate  
**Phase:** 2 — Mathematics for AI  
**Previous lesson:** [Day 002 — How AI Systems Evolved](../day-002/README.md)

## Learning objectives

By the end of this session, you should be able to:

1. Explain scalars, vectors, matrices and higher-dimensional tensors.
2. Read a tensor's shape and rank and explain what each axis represents.
3. Calculate a dot product and explain why it appears in neural networks and embedding search.
4. Perform matrix multiplication and check its dimensions before coding.
5. Explain how a batch of AI examples is represented as a matrix.
6. Identify common tensor bugs involving shape, broadcasting, data type and memory layout.
7. Explain how vector similarity can support retrieval and specialist routing in an agent system.

## 60-minute lesson plan

| Time | Activity |
|---|---|
| 0–5 min | Connect Day 002 problem formulation to mathematical representation |
| 5–15 min | Scalars, vectors and feature representation |
| 15–25 min | Dot product, norms and cosine similarity |
| 25–35 min | Matrices, matrix multiplication and linear transformations |
| 35–43 min | Tensors, shapes, axes, broadcasting and data types |
| 43–55 min | NumPy hands-on lab |
| 55–58 min | AI retrieval and agent-architecture connection |
| 58–60 min | Five-question review |

## 1. Why AI needs linear algebra

In Day 002 we learned that machine learning finds patterns in data. But a computer model does not directly process the human meaning of a phrase such as "DMA timeout." It receives numerical representations.

A simplified neural-network layer performs:

```text
Input numbers → weighted sums → activation → output numbers
```

The weighted sums are mostly vector and matrix operations. Deep-learning frameworks generalize these operations to tensors and execute them efficiently on CPUs, GPUs and specialized accelerators.

Think of linear algebra as the language used to represent data and transform it.

## 2. Scalars, vectors, matrices and tensors

### 2.1 Scalar

A scalar is one number.

Examples:
- Temperature: 42.5
- Learning rate: 0.001
- One model parameter: -0.27

A scalar has no axes. In NumPy its shape is `()`.

### 2.2 Vector

A vector is an ordered collection of numbers.

Example: a telemetry feature vector:

[
x = [4, 2, 0.08]
]

Suppose the fields are:
- Retry count = 4
- Timeout count = 2
- Bus error rate = 0.08

The vector contains three features. Its length is 3 and its one-dimensional array shape is `(3,)`.

A vector is not just a list: its order and units matter. If a model was trained with the order `[retry_count, timeout_count, bus_error_rate]`, inference must use the same order and compatible preprocessing.

**Important NumPy detail:** shape `(3,)` is a one-dimensional array. It is neither explicitly a row matrix nor a column matrix. Use `reshape(1, 3)` for a row and `reshape(3, 1)` for a column when the distinction matters.

### 2.3 Matrix

A matrix is a two-dimensional rectangular arrangement of numbers.

[
X =
egin{bmatrix}
1 & 2 \
3 & 4
end{bmatrix}
]

Its shape is `(2, 2)`: two rows and two columns.

A common ML convention is:
- Rows = examples in a batch
- Columns = features

For example, three telemetry samples with three features each have shape `(3, 3)`.

### 2.4 Tensor

A tensor is a generalization of scalar, vector and matrix to multiple dimensions. In common ML libraries, the term usually refers to an n-dimensional array.

Example: a batch of audio feature frames:

[
	ext{audio features shape} = (B,T,C)
]

where:
- (B) = number of examples in a batch
- (T) = frames or time steps per example
- (C) = feature channels per frame

If shape = `(2, 4, 3)`, it means 2 examples, 4 frames per example and 3 features per frame.

An image batch might use `(B,H,W,C)` or `(B,C,H,W)`, depending on the framework and model. Audio may use `(B,T,C)` or `(B,C,T)`. Neither convention should be guessed: inspect the model/API contract and document each axis.

| Object | Example shape | Number of axes |
|---|---:|---:|
| Scalar | `()` | 0 |
| Vector | `(3,)` | 1 |
| Matrix | `(2, 3)` | 2 |
| Tensor | `(2, 4, 3)` | 3 |

**Terminology note:** ML libraries commonly call the number of dimensions the tensor's rank or `ndim`. In abstract mathematics, tensor rank has a more specialized meaning; here we use the array-library convention.

## 3. Dot product — the weighted sum inside a neuron

Given two vectors:

[
a=[a_1,a_2,ldots,a_n],quad b=[b_1,b_2,ldots,b_n]
]

Their dot product is:

[
acdot b = sum_{i=1}^{n}a_i b_i
]

Worked example:

[
a=[2,1,0.05],quad w=[0.5,1.2,10]
]

[
acdot w = (2)(0.5)+(1)(1.2)+(0.05)(10)=2.2
]

The result is a scalar. In a neural-network unit, the dot product is commonly followed by a bias:

[
z = wcdot x + b
]

Then an activation function may transform (z).

**Why it matters:** a neuron combines input features using learned weights. The weights determine how strongly each feature contributes to the result.

### 3.1 Vector length (norm)

The Euclidean or L2 norm measures a vector's magnitude:

[
|a|_2 = sqrt{sum_i a_i^2}
]

For ([3,4]), the L2 norm is 5.

Norms are used in normalization, regularization and similarity calculations.

### 3.2 Cosine similarity

Cosine similarity measures the angle-based alignment of two non-zero vectors:

[
operatorname{cosine}(a,b)=
rac{acdot b}{|a|_2|b|_2}
]

For (a=[1,0,1]) and (b=[1,1,0]):

- Dot product = 1
- Each norm = (sqrt{2})
- Cosine similarity = (1/2=0.5)

For non-zero vectors, the value is between -1 and 1. Some embedding systems normalize vectors and then use a dot product as a similarity score.

**Do not confuse similarity with probability.** A cosine score of 0.8 does not mean an 80% probability that a retrieved document is correct. Thresholds need evaluation on the application's own data.

## 4. Matrices and matrix multiplication

### 4.1 Matrix-vector multiplication

A matrix can transform an input vector:

[
y=Ax
]

For example:

[
A =
egin{bmatrix}
2 & 1\
1 & 3
end{bmatrix},
quad
x =
egin{bmatrix}
4\
2
end{bmatrix}
]

[
y =
egin{bmatrix}
2(4)+1(2)\
1(4)+3(2)
end{bmatrix}
=
egin{bmatrix}
10\10
end{bmatrix}
]

Each output is a weighted combination of the input values.

Geometrically, a matrix represents a linear transformation: it can rotate, scale, reflect, shear or project vectors, depending on the matrix.

### 4.2 Matrix-matrix multiplication

If (A) has shape ((m,n)) and (B) has shape ((n,p)), then:

[
A_{m	imes n}B_{n	imes p}=C_{m	imes p}
]

The inner dimensions must match. The result uses the outer dimensions.

Example:

[
X =
egin{bmatrix}
1&2\
3&4
end{bmatrix},
quad
W =
egin{bmatrix}
2&0&1\
1&1&0
end{bmatrix}
]

[
XW =
egin{bmatrix}
4&2&1\
10&4&3
end{bmatrix}
]

Shapes:
- (X): ((2,2))
- (W): ((2,3))
- (XW): ((2,3))

The matrix product is not element-by-element multiplication. In NumPy, `A * B` is elementwise for arrays of compatible shapes; `A @ B` performs matrix multiplication.

### 4.3 How a dense neural-network layer uses this

A batch of examples is represented by (X), where rows are examples and columns are input features:

[
X_{B	imes F}W_{F	imes O}+b_{O}=Y_{B	imes O}
]

- (B): batch size
- (F): input features
- (O): output units
- (W): learned weights
- (b): bias vector, broadcast across the batch
- (Y): output values or logits

One matrix multiplication processes many examples together. This is a major reason GPUs and AI accelerators are optimized for matrix multiply and multiply-accumulate (MAC) operations.

## 5. Tensor shapes, axes and broadcasting

### 5.1 Shape is a contract

Suppose a model expects `(batch, features) = (32, 128)`. Supplying `(32, 1, 128)` is not automatically equivalent. The extra axis may change how an operation is interpreted.

At every model boundary, document:
- Shape and meaning of each axis
- Data type (dtype)
- Units and normalization
- Layout/axis order
- Whether values are finite and within expected ranges

### 5.2 Broadcasting

Broadcasting allows compatible arrays of different shapes to participate in an elementwise operation without explicitly copying the smaller array.

For example:
- Matrix shape: `(2, 3)`
- Bias shape: `(3,)`
- Result after addition: `(2, 3)`

The bias is applied to each row.

NumPy/PyTorch broadcasting compares dimensions from the right. Dimensions are compatible when they are equal or one of them is 1 (or an axis is absent).

A common bug: data shape `(B,T,C)` plus a per-frame scale of shape `(T,)` is usually not what you intend. To apply one scalar per frame across channels and batches, use a shape such as `(1,T,1)`.

Read the official [NumPy broadcasting guide](https://numpy.org/doc/stable/user/basics.broadcasting.html) and [PyTorch broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html).

### 5.3 Data types and memory

A tensor's shape does not tell you how much memory it uses. Dtype matters too.

[
	ext{bytes} = left(prod_i 	ext{shape}_iight)	imes 	ext{bytes per element}
]

Example: a `(2,1024,80)` float32 tensor contains 163,840 elements. At 4 bytes per element it uses 655,360 bytes (640 KiB) for the element storage, excluding framework and allocator overhead.

Common types include:
- float32: 4 bytes per element
- float16 / bfloat16: 2 bytes per element
- int8: 1 byte per element

Lower precision can reduce memory traffic and increase throughput, but may affect numerical accuracy. Quantized inference often uses scales/zero-points and may accumulate products in a wider type. Exact behavior depends on the model, hardware and kernel.

### 5.4 Strides and layout

A tensor can be viewed as a shape, dtype and a mapping into underlying storage. Strides describe how many storage elements (or bytes, depending on API) to step when moving along each axis.

Transposing may change shape and strides without moving the underlying values. Some operations require contiguous memory or may create a copy. This matters when optimizing GPU workloads and debugging unexpected memory use.

PyTorch documents tensor storage and views in its [Tensor reference](https://docs.pytorch.org/docs/stable/tensors.html) and [view documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html).

## 6. Connection to embeddings and agent systems

A retrieval-backed assistant often stores one embedding vector per document chunk. At query time:

1. Encode the user query into a vector using the same embedding model/version.
2. Search for the nearest stored vectors.
3. Retrieve the source chunks and metadata.
4. Apply access-control and relevance checks.
5. Give the selected evidence to the answer-generation step.

A router can use embeddings to suggest which specialist is relevant, but should use an allow-list, evaluated thresholds and a fallback path. Similarity scores are ranking signals, not authorization.

See the companion reference design: [Embedding Retrieval and Specialist Routing](../../agent-architectures/patterns/embedding-retrieval-router.md).

## 7. Hands-on lab — vectors, matrices and tensors with NumPy

### Setup

Use Python 3.9 or newer:

```bash
python -m pip install -r requirements.txt
python lab.py
```

The lab uses NumPy. It is intentionally small enough to inspect every value.

### What the program demonstrates

1. A telemetry feature vector and a weighted dot product.
2. A batch matrix multiplied by a weight matrix.
3. Bias broadcasting.
4. Cosine similarity between two embeddings.
5. A rank-3 tensor with explicit batch, frame and feature axes.
6. Shape and element-storage checks.

Expected key outputs:

```text
dot score: 2.2
matrix product:
[[ 4.  2.  1.]
 [10.  4.  3.]]
cosine similarity: 0.5
audio feature tensor shape: (2, 4, 3)
all checks passed
```

### Exercises

1. Change the feature vector and calculate its dot product by hand before running the code.
2. Change the batch from 2 samples to 4. Keep the feature count unchanged and verify the output shape.
3. Add a bias vector of shape `(3,)` and explain why it broadcasts across rows.
4. Create a tensor of shape `(8, 100, 16)` and calculate its float32 storage size in bytes and KiB.
5. Intentionally use incompatible matrix dimensions and explain the error before fixing it.
6. Normalize two vectors and compare their dot product with cosine similarity.

## 8. Production engineering checklist

Before passing tensors between components, validate:

- [ ] Expected shape and named axis order
- [ ] Dtype and device
- [ ] Feature ordering, units and normalization version
- [ ] Finite values (no unexpected NaN or infinity)
- [ ] Batch and sequence length limits
- [ ] Memory budget and copy/contiguity behavior
- [ ] Model/embedding version compatibility
- [ ] Safe handling of malformed or oversized inputs

For an agent that calls a vector-search tool, define a typed contract such as:
- Input: query vector, embedding model ID, namespace, top-k and filters
- Output: candidate IDs, scores, source metadata and index version
- Policy: enforce tenant/namespace access outside the model
- Verification: validate dimensions, score range and source permissions

Do not let the LLM decide whether a caller is authorized to access a namespace.

## 9. Key terminology

| Term | Meaning |
|---|---|
| Scalar | A single numeric value |
| Vector | An ordered one-dimensional collection of values |
| Matrix | A two-dimensional array |
| Tensor | An n-dimensional numerical array in ML frameworks |
| Shape | Size of each axis/dimension |
| Rank / ndim | Number of dimensions in the array-library sense |
| Axis | One dimension with a defined meaning |
| Dot product | Sum of pairwise products of two vectors |
| Norm | A measure of vector magnitude |
| Cosine similarity | Normalized dot product measuring angular alignment |
| Matrix multiplication | Composition of weighted linear transformations |
| Broadcasting | Shape-compatible expansion for elementwise operations |
| Dtype | Numeric representation such as float32 or int8 |
| Stride | Step through storage when advancing along an axis |
| Embedding | Learned vector representation of an item |
| Batch | Multiple examples processed together |

## 10. Five review questions

1. What is the difference between a vector of shape `(4,)` and a matrix of shape `(1,4)`?
2. If (A) has shape `(5,3)` and (B) has shape `(3,2)`, what is the shape of (AB), and why?
3. Why is a dot product used in a neuron and in embedding retrieval? Are the meanings identical?
4. Explain the shape `(B,T,C)` for audio features. What bug can happen if a component expects `(B,C,T)`?
5. Why is an embedding similarity score not automatically a probability or an access-control decision?

## Takeaways

- AI data is represented numerically as vectors, matrices and tensors.
- Shape and axis meaning are part of an interface contract, not cosmetic metadata.
- Dot products combine weighted features and compare vector representations.
- Matrix multiplication is the core operation in dense neural-network layers.
- Broadcasting is convenient but can hide axis mistakes.
- Dtype, strides and memory layout affect accuracy, performance and memory use.
- Retrieval and agent routing can use vector similarity, but still need permission checks, calibrated thresholds and verification.

## Further reading

1. MIT OpenCourseWare — [Vectors and Matrices](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/pages/1.-vectors-and-matrices/).
2. MIT OpenCourseWare — [18.06 Linear Algebra lecture videos](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-videos/).
3. NumPy — [Array objects and fundamentals](https://numpy.org/doc/stable/user/basics.html).
4. NumPy — [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html).
5. PyTorch — [Tensor reference](https://docs.pytorch.org/docs/stable/tensors.html).
6. PyTorch — [Broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html).

## Completion checklist

- [ ] Run the NumPy lab successfully.
- [ ] Explain every shape in the lab without guessing.
- [ ] Complete at least four exercises.
- [ ] Answer all five review questions.
- [ ] Write down one example where tensor shape is part of an API contract.
