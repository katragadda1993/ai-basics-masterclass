"""Day 003 lab: vectors, matrices and tensors with NumPy.

Setup:
    python -m pip install -r requirements.txt
Run:
    python lab.py

All values are synthetic educational examples.
"""

import numpy as np


def main() -> None:
    # 1) A feature vector and a weighted dot product.
    features = np.array([2.0, 1.0, 0.05], dtype=np.float32)
    weights = np.array([0.5, 1.2, 10.0], dtype=np.float32)
    score = features @ weights
    assert features.shape == (3,)
    assert np.isclose(score, 2.7)
    print(f"dot score: {score:.1f}")

    # 2) Two examples (rows) with two features each, projected to 3 outputs.
    batch = np.array([[1, 2], [3, 4]], dtype=np.float32)
    weight_matrix = np.array(
        [[2, 0, 1], [1, 1, 0]], dtype=np.float32
    )
    logits = batch @ weight_matrix
    expected = np.array([[4, 2, 1], [10, 4, 3]], dtype=np.float32)
    assert batch.shape == (2, 2)
    assert weight_matrix.shape == (2, 3)
    assert logits.shape == (2, 3)
    assert np.array_equal(logits, expected)
    print("matrix product:")
    print(logits)

    # 3) A bias vector broadcasts over the batch dimension.
    bias = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    biased_logits = logits + bias
    assert biased_logits.shape == (2, 3)
    assert np.allclose(biased_logits[0], [4.1, 2.2, 1.3])

    # 4) Cosine similarity between two small embedding vectors.
    embedding_a = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    embedding_b = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    denominator = np.linalg.norm(embedding_a) * np.linalg.norm(embedding_b)
    cosine = float(embedding_a @ embedding_b) / float(denominator)
    assert np.isclose(cosine, 0.5)
    print(f"cosine similarity: {cosine:.1f}")

    # 5) A rank-3 tensor: batch, frames, features.
    audio_features = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    assert audio_features.ndim == 3
    assert audio_features.shape == (2, 4, 3)
    print(f"audio feature tensor shape: {audio_features.shape}")
    print(f"float32 element storage: {audio_features.nbytes} bytes")

    # 6) Demonstrate an incompatible shape and check that NumPy rejects it.
    try:
        _ = batch @ np.ones((4, 2), dtype=np.float32)
    except ValueError:
        print("incompatible matrix dimensions correctly rejected")
    else:
        raise AssertionError("Expected incompatible matrix dimensions to fail")

    print("all checks passed")


if __name__ == "__main__":
    main()
