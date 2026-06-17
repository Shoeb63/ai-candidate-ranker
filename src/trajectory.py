"""
Career Trajectory Analyzer  ★ UNIQUE DIFFERENTIATOR ★
------------------------------------------------------
Most rankers look at WHERE a candidate is today.
We look at HOW they got there — direction, momentum, focus.

A great recruiter doesn't just ask "What's your current title?"
They ask "How have you grown? Are you moving toward this role or away from it?"

Trajectory patterns we detect:
  ✅ FOCUSED_GROWTH    — AI/ML roles getting more senior over time
  ✅ PIVOTING_IN       — Engineering background pivoting toward AI (promising)
  ✅ STEADY_CONTRIBUTOR — Consistent AI work, solid but not climbing fast
  ⚠️  STAGNANT         — Same role/level for too long without growth
  ⚠️  UNFOCUSED        — Jumping between unrelated domains
  ❌ PIVOTING_OUT      — Moving AWAY from technical/AI roles (into management etc.)
  ❌ CONSULTING_ONLY   — Entire career in TCS/Infosys/Wipro/etc. (JD explicitly flags this)
"""

# Title classification
AI_ML_TITLES = {
    "ai engineer", "ml engineer", "machine learning engineer", "nlp engineer",
    "applied scientist", "research scientist", "computer vision engineer",
    "applied ml engineer", "lead ai engineer", "senior ai engineer",
    "senior ml engineer", "senior nlp engineer", "data scientist",
    "llm engineer", "ai researcher", "deep learning engineer",
    "recommendation engineer", "search engineer", "ranking engineer",
    "staff ml engineer", "principal ai engineer", "ml scientist"
}

ENGINEERING_TITLES = {
    "software engineer", "backend engineer", "frontend engineer",
    "full stack engineer", "data engineer", "platform engineer",
    "sre", "devops engineer", "cloud engineer", "systems engineer",
    "software developer", "backend developer", "senior software engineer",
    "senior engineer", "staff engineer", "principal engineer"
}

MANAGEMENT_TITLES = {
    "engineering manager", "tech lead", "vp engineering", "cto",
    "director of engineering", "head of engineering", "product manager",
    "program manager", "scrum master", "project manager"
}

NON_TECH_TITLES = {
    "marketing manager", "sales manager", "hr manager", "operations manager",
    "business analyst", "content writer", "graphic designer", "accountant",
    "qa engineer", "manual tester", "business development"
}

CONSULTING_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "hcl", "tech mahindra", "capgemini", "mphasis",
    "hexaware", "l&t infotech", "ltimindtree", "mindtree"
}

SENIORITY_ORDER = {
    "intern": 0, "trainee": 0, "associate": 1, "junior": 1,
    "": 2,  # mid-level (no prefix)
    "senior": 3, "lead": 4, "staff": 4, "principal": 5,
    "architect": 5, "director": 5, "vp": 6, "head": 6, "cto": 7
}


def classify_title(title: str) -> str:
    """Classify a job title into a category."""
    t = title.lower().strip()
    if any(ai in t for ai in AI_ML_TITLES):
        return "ai_ml"
    if any(eng in t for eng in ENGINEERING_TITLES):
        return "engineering"
    if any(mgmt in t for mgmt in MANAGEMENT_TITLES):
        return "management"
    if any(nt in t for nt in NON_TECH_TITLES):
        return "non_tech"
    return "other"


def get_seniority(title: str) -> int:
    """Extract seniority level from title (0=intern, 7=C-suite)."""
    t = title.lower()
    for keyword, level in sorted(SENIORITY_ORDER.items(), key=lambda x: -x[1]):
        if keyword and keyword in t:
            return level
    return 2  # default: mid-level


def is_consulting_company(company: str) -> bool:
    """Check if company is a pure IT services/consulting firm."""
    c = company.lower()
    return any(cons in c for cons in CONSULTING_COMPANIES)


def analyze_trajectory(candidate: dict) -> dict:
    """
    Analyze a candidate's career trajectory.
    
    Returns:
        {
            "pattern": str,           # trajectory pattern name
            "score": float,           # 0.0 to 1.0
            "momentum": float,        # is seniority increasing? 0.0-1.0
            "ai_focus_ratio": float,  # fraction of career in AI/ML roles
            "consulting_only": bool,  # entire career in consulting?
            "explanation": str        # human-readable summary
        }
    """
    career = candidate.get("career_history", [])
    if not career:
        return _default_result("NO_HISTORY", 0.2, "No career history available.")

    # Sort by date (most recent first — career_history is usually already sorted)
    # We'll work oldest-to-newest for trajectory analysis
    roles = list(reversed(career))  # oldest first

    # Classify each role
    classified = []
    for role in roles:
        title = role.get("title", "")
        company = role.get("company", "")
        months = role.get("duration_months", 0)
        classified.append({
            "title": title,
            "company": company,
            "months": months,
            "category": classify_title(title),
            "seniority": get_seniority(title),
            "is_consulting": is_consulting_company(company)
        })

    total_months = sum(r["months"] for r in classified) or 1

    # ── Consulting-only check ──────────────────────────────────────────
    consulting_months = sum(r["months"] for r in classified if r["is_consulting"])
    consulting_ratio = consulting_months / total_months
    consulting_only = consulting_ratio > 0.85

    # ── AI focus ratio ─────────────────────────────────────────────────
    ai_months = sum(r["months"] for r in classified if r["category"] == "ai_ml")
    ai_focus_ratio = ai_months / total_months

    # ── Seniority momentum ─────────────────────────────────────────────
    # Compare last 2 roles vs first 2 roles seniority
    seniorities = [r["seniority"] for r in classified]
    if len(seniorities) >= 2:
        early_avg = sum(seniorities[:2]) / 2
        recent_avg = sum(seniorities[-2:]) / 2
        seniority_delta = recent_avg - early_avg
        momentum = min(max((seniority_delta + 3) / 6, 0.0), 1.0)  # normalize to 0-1
    else:
        momentum = 0.5

    # ── Pattern detection ──────────────────────────────────────────────
    recent_categories = [r["category"] for r in classified[-2:]]
    all_categories = [r["category"] for r in classified]

    if consulting_only:
        pattern = "CONSULTING_ONLY"
        score = 0.15
        explanation = f"Entire career ({consulting_ratio:.0%}) in IT services/consulting. JD explicitly de-prioritizes this."

    elif all(c == "ai_ml" for c in all_categories) and momentum >= 0.5:
        pattern = "FOCUSED_GROWTH"
        score = 1.0
        explanation = f"Consistent AI/ML career with upward seniority trend. {ai_focus_ratio:.0%} of career in AI roles."

    elif all(c == "ai_ml" for c in all_categories):
        pattern = "STEADY_CONTRIBUTOR"
        score = 0.75
        explanation = f"Consistent AI/ML career across {len(classified)} roles. Strong domain focus."

    elif recent_categories[-1] == "ai_ml" and "engineering" in all_categories:
        pattern = "PIVOTING_IN"
        score = 0.70
        explanation = f"Engineering background pivoting into AI/ML. {ai_focus_ratio:.0%} AI experience. Shows intentional transition."

    elif recent_categories[-1] in ("management", "non_tech"):
        pattern = "PIVOTING_OUT"
        score = 0.20
        explanation = f"Recent roles moving away from technical work toward {recent_categories[-1]}. Not a fit for IC role."

    elif ai_focus_ratio < 0.2 and "non_tech" in all_categories:
        pattern = "UNFOCUSED"
        score = 0.10
        explanation = f"Career spans unrelated domains. Only {ai_focus_ratio:.0%} in AI/ML roles."

    elif ai_focus_ratio >= 0.5:
        pattern = "MIXED_AI_FOCUS"
        score = 0.60
        explanation = f"{ai_focus_ratio:.0%} of career in AI/ML roles with some diversity."

    else:
        pattern = "ADJACENT"
        score = 0.35
        explanation = f"Primarily non-AI background with some adjacent experience."

    # Boost for upward momentum in AI roles
    if pattern in ("FOCUSED_GROWTH", "STEADY_CONTRIBUTOR", "PIVOTING_IN", "MIXED_AI_FOCUS"):
        score = min(score + momentum * 0.15, 1.0)

    # Penalty for consulting even if AI roles exist
    if consulting_ratio > 0.5 and not consulting_only:
        score = max(score - 0.15, 0.0)
        explanation += f" Note: {consulting_ratio:.0%} of career in consulting."

    return {
        "pattern": pattern,
        "score": round(score, 4),
        "momentum": round(momentum, 4),
        "ai_focus_ratio": round(ai_focus_ratio, 4),
        "consulting_only": consulting_only,
        "explanation": explanation
    }


def _default_result(pattern, score, explanation):
    return {
        "pattern": pattern,
        "score": score,
        "momentum": 0.5,
        "ai_focus_ratio": 0.0,
        "consulting_only": False,
        "explanation": explanation
    }
