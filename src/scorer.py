"""
Hybrid Scorer
-------------
Combines all scoring signals into a final score.

Final Score =
    35% Semantic similarity  (embedding cosine sim vs JD)
    30% Career Trajectory    ← OUR UNIQUE DIFFERENTIATOR
    20% Skills quality       (depth, assessments, endorsements)
    15% Behavioral signals   (availability, activity, responsiveness)

JD explicitly says:
  - Down-weight candidates not active / low recruiter response rate
  - Penalize pure consulting backgrounds
  - Penalize title-chasers (job-hopping)
  - Penalize non-AI titles even if skills list looks good
"""

from datetime import date, datetime
from src.trajectory import analyze_trajectory

# Skills the JD specifically mentions as "must have"
TIER1_AI_SKILLS = {
    "embeddings", "sentence-transformers", "vector search", "faiss", "pinecone",
    "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "retrieval", "rag", "semantic search", "hybrid search",
    "ranking", "re-ranking", "information retrieval", "ndcg", "mrr",
    "llm", "large language models", "fine-tuning", "lora", "qlora", "peft",
    "bert", "gpt", "llama", "transformers", "nlp",
    "pytorch", "tensorflow", "huggingface"
}

# Skills the JD mentions as "nice to have"
TIER2_AI_SKILLS = {
    "learning to rank", "xgboost", "a/b testing", "mlflow",
    "weights & biases", "recommendation systems", "distributed systems",
    "mlops", "bge", "e5", "openai embeddings"
}

# Skills that sound AI but JD says are NOT the focus
IRRELEVANT_SKILLS = {
    "computer vision", "speech recognition", "robotics", "image classification",
    "object detection", "tts", "asr", "photoshop", "excel", "tableau"
}

PROFICIENCY_WEIGHT = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.5, "beginner": 0.2}


def skill_score(candidate: dict) -> float:
    """Score skill quality: depth, relevance, assessment scores, endorsements."""
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0

    tier1_score = 0.0
    tier2_score = 0.0
    irrelevant_penalty = 0.0

    for skill in skills:
        name = skill.get("name", "").lower().strip()
        prof = PROFICIENCY_WEIGHT.get(skill.get("proficiency", "beginner"), 0.2)
        endorsements = min(skill.get("endorsements", 0), 100) / 100
        duration = min(skill.get("duration_months", 0), 48) / 48

        # Skill trust score = proficiency * (0.5 + 0.3*endorsements + 0.2*duration)
        trust = prof * (0.5 + 0.3 * endorsements + 0.2 * duration)

        if any(t in name for t in TIER1_AI_SKILLS):
            tier1_score += trust
        elif any(t in name for t in TIER2_AI_SKILLS):
            tier2_score += trust * 0.5
        elif any(t in name for t in IRRELEVANT_SKILLS):
            irrelevant_penalty += 0.02  # slight penalty for irrelevant claiming

    # Platform assessment scores (verified — more trustworthy)
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    assessment_bonus = 0.0
    for skill_name, score in assessments.items():
        if any(t in skill_name.lower() for t in TIER1_AI_SKILLS):
            assessment_bonus += (score / 100) * 0.15

    raw = (
        min(tier1_score / 3.0, 0.6) +   # up to 0.6 from tier1 skills
        min(tier2_score / 2.0, 0.15) +  # up to 0.15 from tier2
        min(assessment_bonus, 0.25) -    # up to 0.25 from verified assessments
        min(irrelevant_penalty, 0.1)     # up to -0.1 penalty
    )

    return max(min(raw, 1.0), 0.0)


def behavioral_score(candidate: dict) -> float:
    """Score behavioral / availability signals."""
    signals = candidate.get("redrob_signals", {})
    score = 0.0

    # Open to work (binary — most important availability signal)
    if signals.get("open_to_work_flag", False):
        score += 0.25

    # Notice period (shorter = more available)
    notice = signals.get("notice_period_days", 90)
    if notice == 0:
        score += 0.20
    elif notice <= 30:
        score += 0.15
    elif notice <= 60:
        score += 0.08
    # 90+ days: no bonus (JD says bar gets higher)

    # Recruiter response rate (JD explicitly calls this out)
    rr = signals.get("recruiter_response_rate", 0.0)
    score += rr * 0.20

    # Recent activity — last active date
    last_active_str = signals.get("last_active_date", "")
    if last_active_str:
        try:
            last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
            days_ago = (date.today() - last_active).days
            if days_ago <= 7:
                score += 0.15
            elif days_ago <= 30:
                score += 0.10
            elif days_ago <= 90:
                score += 0.05
            # 6+ months inactive: no bonus (JD says down-weight these)
        except Exception:
            pass

    # Interview completion rate (shows commitment)
    icr = signals.get("interview_completion_rate", 0.0)
    score += icr * 0.10

    # GitHub activity (signals engineering passion)
    github = signals.get("github_activity_score", -1)
    if github > 0:
        score += min(github / 100, 1.0) * 0.10

    return min(score, 1.0)


def education_bonus(candidate: dict) -> float:
    """Small bonus for top-tier institutions (tier_1, tier_2)."""
    for edu in candidate.get("education", []):
        tier = edu.get("tier", "")
        if tier == "tier_1":
            return 0.08
        elif tier == "tier_2":
            return 0.04
    return 0.0


def jd_disqualifier_check(candidate: dict) -> tuple:
    """
    Check for explicit JD disqualifiers.
    Returns (is_disqualified: bool, reason: str, penalty: float)
    
    JD explicitly says to down-weight:
    1. Pure consulting background
    2. Non-AI titles (marketing manager, etc.) even if skills look good
    3. Candidates inactive 6+ months with low response rate
    4. Title chasers (many short stints)
    """
    trajectory = analyze_trajectory(candidate)
    signals = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    reasons = []
    total_penalty = 0.0

    # 1. Pure consulting background
    if trajectory["consulting_only"]:
        reasons.append("Entire career in IT consulting (JD explicit disqualifier)")
        total_penalty += 0.40

    # 2. Non-tech current title (Marketing Manager etc.)
    from src.trajectory import classify_title
    current_cat = classify_title(profile.get("current_title", ""))
    if current_cat == "non_tech":
        reasons.append(f"Current title '{profile.get('current_title','')}' is non-technical")
        total_penalty += 0.35

    # 3. Inactive + unresponsive (the "perfect on paper but not available" trap)
    last_active_str = signals.get("last_active_date", "")
    recruiter_rr = signals.get("recruiter_response_rate", 1.0)
    if last_active_str:
        try:
            last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
            days_ago = (date.today() - last_active).days
            if days_ago > 180 and recruiter_rr < 0.1:
                reasons.append("Inactive 6+ months and very low recruiter response rate")
                total_penalty += 0.30
            elif days_ago > 180:
                reasons.append("Inactive for 6+ months")
                total_penalty += 0.15
        except Exception:
            pass

    # 4. Title chaser — many short stints (< 12 months each)
    short_stints = [r for r in career if r.get("duration_months", 12) < 12 and not r.get("is_current")]
    if len(short_stints) >= 3:
        reasons.append(f"{len(short_stints)} roles under 12 months (possible title-chasing)")
        total_penalty += 0.15

    return bool(reasons), " | ".join(reasons) if reasons else "", min(total_penalty, 0.6)


def hybrid_score(semantic_sim: float, candidate: dict) -> dict:
    """
    Compute final hybrid score combining all signals.

    Weights:
        35% semantic similarity
        30% career trajectory  ← unique differentiator
        20% skill quality
        15% behavioral signals
    """
    # Trajectory analysis (our differentiator)
    traj = analyze_trajectory(candidate)

    # Component scores
    s_skill = skill_score(candidate)
    s_behavioral = behavioral_score(candidate)
    s_edu_bonus = education_bonus(candidate)

    # JD disqualifier check (applies penalties)
    is_flagged, flag_reason, penalty = jd_disqualifier_check(candidate)

    # Weighted combination
    raw = (
        0.35 * semantic_sim +
        0.30 * traj["score"] +
        0.20 * s_skill +
        0.15 * s_behavioral +
        s_edu_bonus
    )

    # Apply disqualifier penalties
    final = max(raw - penalty, 0.0)

    return {
        "final_score":       round(final, 4),
        "semantic_score":    round(semantic_sim, 4),
        "trajectory_score":  round(traj["score"], 4),
        "trajectory_pattern": traj["pattern"],
        "skill_score":       round(s_skill, 4),
        "behavioral_score":  round(s_behavioral, 4),
        "ai_focus_ratio":    round(traj["ai_focus_ratio"], 4),
        "is_flagged":        is_flagged,
        "flag_reason":       flag_reason,
    }
