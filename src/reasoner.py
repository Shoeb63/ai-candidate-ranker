"""
Reasoner
--------
Generates human-readable reasoning for why a candidate fits the role.
Runs FULLY OFFLINE — no API calls during ranking (hackathon requirement).

Reasoning is built from actual score components, so it's meaningful,
not just a generic template.
"""

from src.trajectory import classify_title


PATTERN_DESCRIPTIONS = {
    "FOCUSED_GROWTH":     "Consistent upward trajectory in AI/ML roles",
    "STEADY_CONTRIBUTOR": "Stable AI/ML career with solid domain focus",
    "PIVOTING_IN":        "Engineering background pivoting intentionally into AI/ML",
    "MIXED_AI_FOCUS":     "Mixed background with meaningful AI/ML experience",
    "ADJACENT":           "Adjacent technical background with some AI exposure",
    "STAGNANT":           "Limited career progression observed",
    "UNFOCUSED":          "Career spans multiple unrelated domains",
    "CONSULTING_ONLY":    "Entire career in IT consulting/services",
    "PIVOTING_OUT":       "Recent roles moving away from technical work",
    "NO_HISTORY":         "No career history available",
}


def generate_reasoning(candidate: dict, scores: dict) -> str:
    """
    Build a 2-3 sentence reasoning string from score components.
    No API calls — purely from data.
    """
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    title = profile.get("current_title", "N/A")
    company = profile.get("current_company", "N/A")
    yoe = profile.get("years_of_experience", 0)

    # Top skills (expert/advanced first)
    top_skills = [
        s["name"] for s in sorted(skills, key=lambda x: {"expert": 3, "advanced": 2, "intermediate": 1, "beginner": 0}.get(x.get("proficiency", "beginner"), 0), reverse=True)
    ][:4]
    skill_str = ", ".join(top_skills) if top_skills else "general engineering skills"

    # Trajectory sentence
    pattern = scores.get("trajectory_pattern", "")
    traj_desc = PATTERN_DESCRIPTIONS.get(pattern, "Career trajectory analyzed")
    traj_score = scores.get("trajectory_score", 0)

    # Availability sentence
    notice = signals.get("notice_period_days", 90)
    open_to_work = signals.get("open_to_work_flag", False)
    rr = signals.get("recruiter_response_rate", 0)

    if notice == 0 and open_to_work:
        avail = "immediately available and actively looking"
    elif notice <= 30 and open_to_work:
        avail = f"open to work with {notice}-day notice"
    elif notice <= 60:
        avail = f"{notice}-day notice period"
    else:
        avail = f"{notice}-day notice period"

    # Build sentences
    sentence1 = f"{title} at {company} with {yoe} years of experience; key skills include {skill_str}."

    sentence2 = f"{traj_desc} (trajectory score: {traj_score:.2f}/1.0), with {scores.get('ai_focus_ratio', 0)*100:.0f}% of career in AI/ML roles."

    sentence3 = f"Candidate is {avail}; recruiter response rate {rr*100:.0f}%."

    # Add flag warning if flagged
    if scores.get("is_flagged") and scores.get("flag_reason"):
        sentence3 += f" ⚠️ Note: {scores['flag_reason']}."

    return f"{sentence1} {sentence2} {sentence3}"
