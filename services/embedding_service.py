import os
import logging
import numpy as np
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_client = None


def _get_client():
    global _client
    if _client is None:
        token = os.getenv("HF_API_TOKEN") or os.getenv("HF_TOKEN", "")
        _client = InferenceClient(provider="hf-inference", api_key=token)
    return _client


def get_embedding(text: str) -> list:
    """Get embedding for a single text using HF Inference API."""
    if not text or not text.strip():
        return [0.0] * EMBEDDING_DIM

    try:
        result = _get_client().feature_extraction(text, model=MODEL)
        vec = np.array(result, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()
    except Exception as e:
        logger.error(f"Embedding API error: {e}")
        return [0.0] * EMBEDDING_DIM


def get_embeddings_batch(texts: list) -> list:
    """Get embeddings for a batch of texts."""
    if not texts:
        return []
    return [get_embedding(t) for t in texts]


def cosine_similarity(vec1, vec2) -> float:
    """Compute cosine similarity between two vectors using numpy dot product."""
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(v1 / norm1, v2 / norm2))
