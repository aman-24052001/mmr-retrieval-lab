"""End-to-end retriever: embeddings + MMR + metrics. Drop-in for a RAG
pipeline. Give it a corpus once, then call retrieve(query, k, lambda_param)."""
from __future__ import annotations
import numpy as np
from .embeddings import get_embeddings, cosine_matrix, query_doc_similarity
from .mmr import mmr, top_k_relevance, mmr_scores_trace
from .metrics import summarize


class MMRRetriever:
    def __init__(self, documents, model_name="all-MiniLM-L6-v2"):
        self.documents = documents
        self.model_name = model_name
        self.doc_emb = get_embeddings(documents, model_name)
        self.doc_doc_sim = cosine_matrix(self.doc_emb)

    def _query_sim(self, query):
        q_emb = get_embeddings([query], self.model_name)[0]
        return query_doc_similarity(q_emb, self.doc_emb)

    def retrieve(self, query, k=7, lambda_param=0.7):
        q_sim = self._query_sim(query)
        mmr_idx = mmr(q_sim, self.doc_doc_sim, k, lambda_param)
        base_idx = top_k_relevance(q_sim, k)
        return {
            "query": query, "lambda": lambda_param,
            "mmr": {"indices": mmr_idx,
                    "documents": [self.documents[i] for i in mmr_idx],
                    "metrics": summarize(q_sim, self.doc_doc_sim, mmr_idx)},
            "baseline": {"indices": list(base_idx),
                         "documents": [self.documents[i] for i in base_idx],
                         "metrics": summarize(q_sim, self.doc_doc_sim, base_idx)},
        }

    def explain(self, query, k=7, lambda_param=0.7):
        q_sim = self._query_sim(query)
        trace = mmr_scores_trace(q_sim, self.doc_doc_sim, k, lambda_param)
        for step in trace:
            step["document"] = self.documents[step["index"]]
        return trace
