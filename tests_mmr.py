"""Minimal correctness tests for the MMR core. Run: python3 tests_mmr.py"""
import numpy as np
from src.mmr import mmr, top_k_relevance

def test_lambda_one_equals_relevance():
    """lambda=1 must reproduce pure relevance ranking."""
    q = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    dd = np.eye(5)
    assert mmr(q, dd, 5, 1.0) == top_k_relevance(q, 5)

def test_first_pick_is_most_relevant():
    """First selection is always the most relevant doc, regardless of lambda."""
    q = np.array([0.1, 0.9, 0.3])
    dd = np.eye(3)
    for lam in (0.0, 0.5, 1.0):
        assert mmr(q, dd, 1, lam)[0] == 1

def test_diversity_avoids_duplicates():
    """With two near-identical top docs, low lambda should skip the duplicate."""
    q = np.array([0.9, 0.89, 0.5, 0.5])
    dd = np.array([
        [1.0, 0.99, 0.1, 0.1],
        [0.99, 1.0, 0.1, 0.1],
        [0.1, 0.1, 1.0, 0.2],
        [0.1, 0.1, 0.2, 1.0],
    ])
    picks = mmr(q, dd, 2, 0.3)
    assert picks[0] == 0          # most relevant first
    assert picks[1] != 1          # second pick avoids the 0.99-similar duplicate

def test_k_larger_than_n():
    q = np.array([0.5, 0.4]); dd = np.eye(2)
    assert len(mmr(q, dd, 10, 0.7)) == 2

def test_empty():
    assert mmr(np.array([]), np.zeros((0,0)), 5, 0.7) == []

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
