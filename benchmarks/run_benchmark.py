"""Runs MMR vs baseline on the London corpus. Produces results/lambda_sweep.png,
results/metrics.json, and the duplicate-pair demonstration to console.

Uses TF-IDF embeddings: lightweight, deterministic, effective here because the
near-duplicate docs share surface vocabulary. Swap in sentence-transformers
(src/embeddings.py) for semantic production use."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.mmr import mmr, top_k_relevance
from src.metrics import summarize, redundancy, mean_relevance, max_pairwise_similarity

ROOT = os.path.join(os.path.dirname(__file__), "..")
with open(os.path.join(ROOT, "data", "corpus.json")) as f:
    corpus = json.load(f)
query = corpus["query"]
titles = [d["title"] for d in corpus["documents"]]

vec = TfidfVectorizer(stop_words="english")
tfidf = vec.fit_transform([query] + titles).toarray()
q_emb, doc_emb = tfidf[0], tfidf[1:]

def cos_mat(m):
    n = np.linalg.norm(m, axis=1, keepdims=True); n[n==0]=1
    u = m/n; return u@u.T

doc_doc = cos_mat(doc_emb)
qn = np.linalg.norm(q_emb) or 1.0
dn = np.linalg.norm(doc_emb, axis=1, keepdims=True); dn[dn==0]=1
q_sim = (doc_emb/dn) @ (q_emb/qn)
K = 7

base = top_k_relevance(q_sim, K)
print("="*64); print("BASELINE — pure relevance (top-k cosine)"); print("="*64)
for i in base: print(f"  {q_sim[i]:.4f}  {titles[i]}")
print(f"\n  redundancy={redundancy(doc_doc, base):.4f}  max_pair_sim={max_pairwise_similarity(doc_doc, base):.4f}")

mmr_idx = mmr(q_sim, doc_doc, K, 0.7)
print("\n" + "="*64); print("MMR — lambda=0.7"); print("="*64)
for i in mmr_idx: print(f"  {q_sim[i]:.4f}  {titles[i]}")
print(f"\n  redundancy={redundancy(doc_doc, mmr_idx):.4f}  max_pair_sim={max_pairwise_similarity(doc_doc, mmr_idx):.4f}")

lambdas = np.linspace(0, 1, 21)
rel_curve, div_curve = [], []
for lam in lambdas:
    idx = mmr(q_sim, doc_doc, K, float(lam))
    rel_curve.append(mean_relevance(q_sim, idx))
    div_curve.append(1 - redundancy(doc_doc, idx))

plt.rcParams["font.family"] = "monospace"
fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor("#FFFDF5"); ax.set_facecolor("#FFFDF5")
ax.plot(lambdas, rel_curve, color="#1A1A1A", lw=3, marker="s", ms=6,
        markerfacecolor="#FBE311", markeredgecolor="#1A1A1A", markeredgewidth=2, label="Mean Relevance")
ax.plot(lambdas, div_curve, color="#1A1A1A", lw=3, marker="o", ms=6,
        markerfacecolor="#F15153", markeredgecolor="#1A1A1A", markeredgewidth=2, label="Diversity (1 - redundancy)")
ax.axvline(0.7, color="#013E37", lw=2, ls="--", alpha=0.7)
ax.text(0.71, 0.05, "lambda=0.7\n(recommended)", fontsize=9, color="#013E37", fontweight="bold", transform=ax.get_xaxis_transform())
ax.set_xlabel("lambda  (0 = diversity  -->  1 = relevance)", fontsize=12, fontweight="bold")
ax.set_ylabel("score", fontsize=12, fontweight="bold")
ax.set_title("MMR: the relevance / diversity tradeoff", fontsize=15, fontweight="bold", pad=14)
for s in ax.spines.values(): s.set_linewidth(2.5); s.set_color("#1A1A1A")
ax.tick_params(width=2); ax.grid(True, alpha=0.15, color="#1A1A1A")
leg = ax.legend(fontsize=10, frameon=True, loc="center right")
leg.get_frame().set_edgecolor("#1A1A1A"); leg.get_frame().set_linewidth(2); leg.get_frame().set_facecolor("#FFFDF5")
plt.tight_layout()
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
plt.savefig(os.path.join(ROOT, "results", "lambda_sweep.png"), dpi=130, facecolor="#FFFDF5")
print("\n[saved] results/lambda_sweep.png")

out = {"query": query, "k": K,
       "baseline": {"indices": list(map(int, base)), **summarize(q_sim, doc_doc, base)},
       "mmr_0.7": {"indices": list(map(int, mmr_idx)), **summarize(q_sim, doc_doc, mmr_idx)},
       "sweep": {"lambda": [round(float(x),2) for x in lambdas],
                 "relevance": [round(float(x),4) for x in rel_curve],
                 "diversity": [round(float(x),4) for x in div_curve]}}
with open(os.path.join(ROOT, "results", "metrics.json"), "w") as f:
    json.dump(out, f, indent=2)
print("[saved] results/metrics.json")
