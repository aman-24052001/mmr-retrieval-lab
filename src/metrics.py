"""Quantitative metrics for evaluating a retrieved set. Putting numbers on
relevance and redundancy so configurations can be compared objectively."""
from __future__ import annotations
import numpy as np


def mean_relevance(query_doc_sim, selected):
    if not selected: return 0.0
    return float(np.mean([query_doc_sim[i] for i in selected]))


def redundancy(doc_doc_sim, selected):
    """Mean pairwise similarity within the set. The number MMR pushes down."""
    if len(selected) < 2: return 0.0
    sims = [doc_doc_sim[a, b] for idx, a in enumerate(selected) for b in selected[idx+1:]]
    return float(np.mean(sims))


def diversity(doc_doc_sim, selected):
    return 1.0 - redundancy(doc_doc_sim, selected)


def max_pairwise_similarity(doc_doc_sim, selected):
    """The single most-similar pair: a near-duplicate detector."""
    if len(selected) < 2: return 0.0
    return float(max(doc_doc_sim[a, b] for idx, a in enumerate(selected) for b in selected[idx+1:]))


def summarize(query_doc_sim, doc_doc_sim, selected):
    return {
        "mean_relevance": round(mean_relevance(query_doc_sim, selected), 4),
        "redundancy": round(redundancy(doc_doc_sim, selected), 4),
        "diversity": round(diversity(doc_doc_sim, selected), 4),
        "max_pair_sim": round(max_pairwise_similarity(doc_doc_sim, selected), 4),
    }
