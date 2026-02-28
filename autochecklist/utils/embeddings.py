"""Text embedding utilities for checklist refinement.

Uses OpenAI's text-embedding-3-large model for computing
semantic similarity between checklist questions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    import numpy as np


def _require_numpy():
    """Lazy-import numpy, raising a helpful error if missing."""
    try:
        import numpy as np
        return np
    except ImportError:
        raise ImportError(
            "numpy is required for embedding operations. "
            "Install with: pip install 'autochecklist[ml]'"
        ) from None



def get_embeddings(
    texts: List[str],
    model: str = "text-embedding-3-large",
    api_key: Optional[str] = None,
) -> np.ndarray:
    """Get embeddings for a list of texts.

    Uses OpenAI embeddings via direct API call.

    Args:
        texts: List of text strings to embed
        model: Embedding model to use (default: text-embedding-3-large)
        api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    import httpx
    import os

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key required for embeddings. "
            "Set OPENAI_API_KEY environment variable or pass api_key parameter."
        )

    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "input": texts,
                "model": model,
            },
        )
        response.raise_for_status()
        data = response.json()

    # Extract embeddings in order
    np = _require_numpy()
    embeddings = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
    return np.array(embeddings)


def cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix.

    Args:
        embeddings: numpy array of shape (n, embedding_dim)

    Returns:
        numpy array of shape (n, n) with cosine similarities
    """
    np = _require_numpy()
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms

    # Compute cosine similarity matrix
    return np.dot(normalized, normalized.T)


def find_similar_pairs(
    similarity_matrix: np.ndarray,
    threshold: float = 0.85,
) -> List[Tuple[int, int, float]]:
    """Find pairs of items with similarity above threshold.

    Args:
        similarity_matrix: Pairwise similarity matrix
        threshold: Minimum similarity to include

    Returns:
        List of (i, j, similarity) tuples for similar pairs
    """
    n = similarity_matrix.shape[0]
    similar_pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= threshold:
                similar_pairs.append((i, j, similarity_matrix[i, j]))

    return similar_pairs


def build_similarity_graph(
    questions: List[str],
    threshold: float = 0.85,
    embeddings: Optional[np.ndarray] = None,
    api_key: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, float]]]:
    """Build similarity graph from questions.

    Args:
        questions: List of question strings
        threshold: Similarity threshold for edges
        embeddings: Pre-computed embeddings (optional)
        api_key: OpenAI API key for embeddings

    Returns:
        Tuple of (embeddings, similarity_matrix, similar_pairs)
    """
    if embeddings is None:
        embeddings = get_embeddings(questions, api_key=api_key)

    similarity_matrix = cosine_similarity(embeddings)
    similar_pairs = find_similar_pairs(similarity_matrix, threshold)

    return embeddings, similarity_matrix, similar_pairs
