"""
Embedder
--------
Converts candidate profiles into dense vectors using sentence-transformers.
Runs fully OFFLINE — no API calls, no internet needed during ranking.

Model: all-MiniLM-L6-v2
  - Fast, small (~80MB), accurate enough for this task
  - Produces 384-dim normalized vectors (cosine similarity = dot product)
"""

from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[Embedder] Loading model '{MODEL_NAME}' ...")
        _model = SentenceTransformer(MODEL_NAME)
        print("[Embedder] Model ready.")
    return _model


def build_candidate_text(c: dict) -> str:
    """
    Convert a candidate JSON into a SHORT, dense text string for embedding.
    Kept short intentionally — shorter text = faster embedding, still accurate.
    """
    parts = []
    profile = c.get("profile", {})

    # Current role (most important signal)
    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    yoe = profile.get("years_of_experience", 0)
    parts.append(f"{title} at {company} {yoe}yr")

    # Headline (if exists)
    headline = profile.get("headline", "")
    if headline:
        parts.append(headline[:100])

    # Top skills only (expert/advanced, max 8)
    skills = c.get("skills", [])
    top_skills = [s["name"] for s in skills
                  if s.get("proficiency") in ("expert", "advanced")][:8]
    if top_skills:
        parts.append(f"Skills: {', '.join(top_skills)}")

    # Most recent 2 roles only (descriptions trimmed to 150 chars)
    for role in c.get("career_history", [])[:2]:
        r_title = role.get("title", "")
        r_desc = role.get("description", "")[:150]
        parts.append(f"{r_title}: {r_desc}")

    return " | ".join(filter(None, parts))


def embed_texts(texts: list) -> np.ndarray:
    """Embed a list of strings. Returns (N, 384) float32 array."""
    model = get_model()
    return model.encode(
        texts,
        show_progress_bar=True,
        batch_size=256,
        normalize_embeddings=True,  # cosine sim = dot product
        convert_to_numpy=True
    )


def embed_single(text: str) -> np.ndarray:
    """Embed one string. Returns (384,) float32 array."""
    model = get_model()
    return model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
