"""
FAISS Index
-----------
Builds an in-memory vector index over all candidates.
Finds the top-K semantically closest candidates to a JD vector
in milliseconds — even for 100K candidates.

Uses IndexFlatIP (inner product) because vectors are L2-normalized,
so inner product == cosine similarity.
"""

import faiss
import numpy as np
import os
import pickle

INDEX_PATH = "output/faiss.index"
META_PATH  = "output/faiss_meta.pkl"


def build_index(vectors: np.ndarray, candidate_ids: list) -> tuple:
    """Build FAISS flat index and save to disk."""
    vectors = np.array(vectors, dtype=np.float32)
    dim = vectors.shape[1]

    print(f"[FAISS] Building index: {len(candidate_ids):,} candidates, dim={dim}")
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    print(f"[FAISS] Index ready. Total: {index.ntotal:,}")

    os.makedirs("output", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(candidate_ids, f)

    return index, candidate_ids


def load_index() -> tuple:
    """Load saved FAISS index from disk."""
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        candidate_ids = pickle.load(f)
    print(f"[FAISS] Loaded index: {index.ntotal:,} candidates")
    return index, candidate_ids


def search(index, candidate_ids: list, query_vec: np.ndarray, top_k: int = 500) -> list:
    """
    Search top-K candidates by cosine similarity.
    Returns: [(candidate_id, similarity_score), ...]
    """
    query = np.array([query_vec], dtype=np.float32)
    scores, indices = index.search(query, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append((candidate_ids[idx], float(score)))
    return results
