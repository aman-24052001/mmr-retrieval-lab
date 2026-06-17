"""Exports TF-IDF query/doc similarities to JSON so the static web demo can run
MMR client-side with zero backend."""
import json, os, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = os.path.join(os.path.dirname(__file__), "..")
with open(os.path.join(ROOT, "data", "corpus.json")) as f:
    corpus = json.load(f)
query = corpus["query"]
docs = corpus["documents"]
titles = [d["title"] for d in docs]

vec = TfidfVectorizer(stop_words="english")
tfidf = vec.fit_transform([query] + titles).toarray()
q_emb, doc_emb = tfidf[0], tfidf[1:]

def cos_mat(m):
    n = np.linalg.norm(m, axis=1, keepdims=True); n[n==0]=1
    u = m/n; return u@u.T

doc_doc = cos_mat(doc_emb)
qn = np.linalg.norm(q_emb) or 1.0
dn = np.linalg.norm(doc_emb, axis=1, keepdims=True); dn[dn==0]=1
q_sim = ((doc_emb/dn) @ (q_emb/qn))

payload = {
    "query": query,
    "documents": [{"id": d["id"], "title": d["title"], "category": d["category"]} for d in docs],
    "query_doc_sim": [round(float(x), 5) for x in q_sim],
    "doc_doc_sim": [[round(float(x), 5) for x in row] for row in doc_doc],
}
out = os.path.join(ROOT, "web", "public", "data.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(payload, f)
print(f"[saved] {out}  ({len(docs)} docs)")
