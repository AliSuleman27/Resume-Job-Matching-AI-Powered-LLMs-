import os
import hashlib
import logging
import numpy as np
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM = 768

_client = None
_cache_collection = None


def init_cache(collection):
    """Set the MongoDB collection used for embedding caching."""
    global _cache_collection
    _cache_collection = collection


def _get_client():
    global _client
    if _client is None:
        token = os.getenv("HF_API_TOKEN") or os.getenv("HF_TOKEN", "")
        _client = InferenceClient(provider="hf-inference", api_key=token)
    return _client


def get_embedding(text: str) -> list:
    """Get embedding for a single text, with MongoDB caching."""
    if not text or not text.strip():
        return [0.0] * EMBEDDING_DIM

    cache_key = hashlib.sha256((MODEL + ":" + text.strip()).encode()).hexdigest()

    # Check cache
    if _cache_collection is not None:
        try:
            cached = _cache_collection.find_one({'_k': cache_key})
            if cached:
                return cached['emb']
        except Exception as e:
            logger.debug(f"Cache read error: {e}")

    try:
        result = _get_client().feature_extraction(text, model=MODEL)
        vec = np.array(result, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        embedding = vec.tolist()

        # Store in cache
        if _cache_collection is not None:
            try:
                _cache_collection.update_one(
                    {'_k': cache_key},
                    {'$set': {'_k': cache_key, 'emb': embedding}},
                    upsert=True
                )
            except Exception as e:
                logger.debug(f"Cache write error: {e}")

        return embedding
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
