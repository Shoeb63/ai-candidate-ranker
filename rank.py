"""
AI Candidate Ranking System
============================
Entry point — reads candidates.jsonl + JD, runs full pipeline,
outputs top 100 ranked candidates to submission.csv

Pipeline:
  1. Load candidates + JD
  2. Embed all candidate profiles (sentence-transformers, offline)
  3. Build FAISS index
  4. Embed JD + semantic search → top 500
  5. Hybrid score each candidate (semantic + trajectory + skills + behavioral)
  6. Generate offline reasoning
  7. Write submission.csv

Usage:
  python rank.py --candidates ./data/candidates.jsonl --jd ./data/job_description.txt
  python rank.py --candidates ./data/candidates.jsonl --jd ./data/job_description.txt --verbose
"""

import argparse
import json
import csv
import os
from tqdm import tqdm

from src.embedder import build_candidate_text, embed_texts, embed_single
from src.faiss_index import build_index, search
from src.scorer import hybrid_score
from src.reasoner import generate_reasoning


DEFAULT_JD = """
Senior AI Engineer — Founding Team at Redrob AI.
5-9 years experience. Must have: production embeddings, vector search (FAISS/Pinecone/Weaviate),
hybrid retrieval, RAG pipelines, ranking systems, NLP, LLMs, PyTorch.
Strong Python. Evaluation frameworks (NDCG, MRR). Product company experience preferred.
Pune/Noida location. Sub-30-day notice preferred.
NOT looking for: pure consulting background, non-AI titles, inactive candidates,
computer vision/speech specialists without NLP experience.
"""


def load_candidates(path: str) -> list:
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"[Main] Loaded {len(candidates):,} candidates.")
    return candidates


def load_jd(path: str) -> str:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        print(f"[Main] Loaded JD from {path} ({len(content)} chars)")
        return content
    print("[Main] Using default JD.")
    return DEFAULT_JD


def main():
    parser = argparse.ArgumentParser(description="AI Candidate Ranking System")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--jd", default=None, help="Path to job_description.txt")
    parser.add_argument("--out", default="./output/submission.csv", help="Output CSV path")
    parser.add_argument("--top", type=int, default=100, help="Top N candidates to output")
    parser.add_argument("--pool", type=int, default=500, help="Semantic search pool size before re-scoring")
    parser.add_argument("--verbose", action="store_true", help="Print top 10 to console")
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  AI Candidate Ranking System")
    print("="*55)

    # ── Step 1: Load ───────────────────────────────────────────────
    print("\n[Step 1] Loading data...")
    candidates = load_candidates(args.candidates)
    jd = load_jd(args.jd)
    candidate_map = {c["candidate_id"]: c for c in candidates}

    # ── Step 2: Embed candidates ───────────────────────────────────
    print("\n[Step 2] Embedding candidate profiles...")
    texts = [build_candidate_text(c) for c in candidates]
    candidate_ids = [c["candidate_id"] for c in candidates]
    vectors = embed_texts(texts)

    # ── Step 3: Build FAISS index ──────────────────────────────────
    print("\n[Step 3] Building FAISS index...")
    index, ids = build_index(vectors, candidate_ids)

    # ── Step 4: Embed JD + Semantic search ────────────────────────
    print("\n[Step 4] Semantic search against JD...")
    jd_vector = embed_single(jd)
    pool_size = min(args.pool, len(candidates))
    top_results = search(index, ids, jd_vector, top_k=pool_size)
    print(f"[Main] Retrieved top {len(top_results):,} candidates from semantic search.")

    # ── Step 5: Hybrid scoring ─────────────────────────────────────
    print("\n[Step 5] Hybrid scoring (semantic + trajectory + skills + behavioral)...")
    scored = []
    for cid, semantic_sim in tqdm(top_results, desc="Scoring"):
        candidate = candidate_map.get(cid)
        if not candidate:
            continue
        scores = hybrid_score(semantic_sim, candidate)
        scored.append((cid, scores))

    scored.sort(key=lambda x: x[1]["final_score"], reverse=True)
    top_candidates = scored[:args.top]

    # ── Step 6: Generate reasoning ─────────────────────────────────
    print(f"\n[Step 6] Generating reasoning for top {args.top}...")
    results = []
    for rank, (cid, scores) in enumerate(tqdm(top_candidates, desc="Reasoning"), start=1):
        candidate = candidate_map[cid]
        reasoning = generate_reasoning(candidate, scores)
        results.append({
            "rank":               rank,
            "candidate_id":       cid,
            "final_score":        scores["final_score"],
            "semantic_score":     scores["semantic_score"],
            "trajectory_score":   scores["trajectory_score"],
            "trajectory_pattern": scores["trajectory_pattern"],
            "skill_score":        scores["skill_score"],
            "behavioral_score":   scores["behavioral_score"],
            "reasoning":          reasoning,
        })

    # ── Step 7: Write CSV ──────────────────────────────────────────
    print(f"\n[Step 7] Writing output to {args.out}...")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fieldnames = [
        "rank", "candidate_id", "final_score",
        "semantic_score", "trajectory_score", "trajectory_pattern",
        "skill_score", "behavioral_score", "reasoning"
    ]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Done! Top {args.top} candidates → {args.out}")

    # ── Verbose: print top 10 ──────────────────────────────────────
    if args.verbose:
        print("\n" + "="*55)
        print("  TOP 10 CANDIDATES")
        print("="*55)
        for r in results[:10]:
            print(f"\n#{r['rank']:>3}  {r['candidate_id']}  |  Score: {r['final_score']:.4f}")
            print(f"      Semantic: {r['semantic_score']:.3f}  |  Trajectory: {r['trajectory_score']:.3f} [{r['trajectory_pattern']}]")
            print(f"      Skills:   {r['skill_score']:.3f}  |  Behavioral: {r['behavioral_score']:.3f}")
            print(f"      {r['reasoning'][:120]}...")


if __name__ == "__main__":
    main()
