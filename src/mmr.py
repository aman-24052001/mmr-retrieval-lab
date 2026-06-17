"""
Vectorized Maximal Marginal Relevance.

Unlike the textbook implementation that recomputes max() over the selected
set on every iteration (O(n*k) similarity lookups inside a Python loop), this
keeps a running vector of "max similarity to anything already selected" and
updates it in O(n) per pick.
"""
from __future__ import annotations
import numpy as np


def mmr(query_doc_sim, doc_doc_sim, k, lambda_param=0.7):
    """Return indices of k documents ordered by MMR selection.

    lambda_param = 1.0 -> pure relevance (standard ranking)
    lambda_param = 0.0 -> pure diversity
    """
    n = query_doc_sim.shape[0]
    if n == 0 or k <= 0:
        return []
    k = min(k, n)

    selected = [int(np.argmax(query_doc_sim))]
    max_sim_to_selected = doc_doc_sim[selected[0]].copy()
    candidate_mask = np.ones(n, dtype=bool)
    candidate_mask[selected[0]] = False

    while len(selected) < k:
        scores = lambda_param * query_doc_sim - (1 - lambda_param) * max_sim_to_selected
        scores[~candidate_mask] = -np.inf
        nxt = int(np.argmax(scores))
        selected.append(nxt)
        candidate_mask[nxt] = False
        max_sim_to_selected = np.maximum(max_sim_to_selected, doc_doc_sim[nxt])

    return selected


def mmr_scores_trace(query_doc_sim, doc_doc_sim, k, lambda_param=0.7):
    """Same as mmr() but returns per-step diagnostics: which doc was picked,
    its relevance term, its redundancy penalty, and final score."""
    n = query_doc_sim.shape[0]
    if n == 0 or k <= 0:
        return []
    k = min(k, n)

    first = int(np.argmax(query_doc_sim))
    trace = [{"step": 0, "index": first,
              "relevance": float(query_doc_sim[first]),
              "redundancy": 0.0, "mmr": float(query_doc_sim[first])}]
    selected = [first]
    max_sim = doc_doc_sim[first].copy()
    mask = np.ones(n, dtype=bool); mask[first] = False

    while len(selected) < k:
        scores = lambda_param * query_doc_sim - (1 - lambda_param) * max_sim
        scores[~mask] = -np.inf
        nxt = int(np.argmax(scores))
        trace.append({"step": len(selected), "index": nxt,
                      "relevance": float(query_doc_sim[nxt]),
                      "redundancy": float(max_sim[nxt]),
                      "mmr": float(scores[nxt])})
        selected.append(nxt)
        mask[nxt] = False
        max_sim = np.maximum(max_sim, doc_doc_sim[nxt])

    return trace


def top_k_relevance(query_doc_sim, k):
    """Plain relevance ranking baseline (equivalent to MMR with lambda=1)."""
    return list(np.argsort(query_doc_sim)[::-1][:k])
