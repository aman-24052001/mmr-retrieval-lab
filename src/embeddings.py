"""Pluggable embedding backends. Defaults to sentence-transformers (purpose-
trained for semantic similarity), with a zero-dependency hashing fallback so
the pipeline runs anywhere."""
from __future__ import annotations
import numpy as np


def get_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)
    except Exception:
        return _hashing_embeddings(texts)


def _hashing_embeddings(texts, dim=256):
    vecs = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            vecs[i, hash(tok) % dim] += 1.0
    norms = np.linalg.norm(vecs, axis=1, keepdims=True); norms[norms == 0] = 1.0
    return vecs / norms


def cosine_matrix(emb):
    norms = np.linalg.norm(emb, axis=1, keepdims=True); norms[norms == 0] = 1.0
    unit = emb / norms
    return unit @ unit.T


def query_doc_similarity(query_emb, doc_emb):
    q = query_emb / (np.linalg.norm(query_emb) or 1.0)
    norms = np.linalg.norm(doc_emb, axis=1, keepdims=True); norms[norms == 0] = 1.0
    return (doc_emb / norms) @ q
