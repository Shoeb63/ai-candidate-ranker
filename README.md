# AI Candidate Ranking System

An AI-powered candidate ranking system that understands candidates the way a great recruiter would — through semantic understanding + career trajectory analysis, not keyword matching.

## What makes this unique: Career Trajectory Analysis

Most systems ask "Where is the candidate today?"
This system asks "How did they get here, and where are they heading?"

```
Junior Dev (2yr) → ML Engineer (2yr) → Senior AI Engineer (3yr)
= FOCUSED_GROWTH ✅ — intentional, upward, AI-focused

AI Engineer (7yr, same role, same company)
= STAGNANT ⚠️ — technically qualified but no growth signal

Backend → Frontend → Data Analyst → AI Engineer (recent)
= PIVOTING_IN ✅ — intentional transition, shows initiative

Marketing Manager with Python listed as skill
= NON_TECH ❌ — JD explicitly calls this out as a trap
```

## Architecture

```
candidates.jsonl + job_description.txt
          ↓
  Sentence Transformers          ← semantic understanding (offline)
  (all-MiniLM-L6-v2)
          ↓
  FAISS Vector Index             ← fast search over 100K candidates
          ↓
  Hybrid Scoring:
    35% Semantic similarity
    30% Career Trajectory        ← unique differentiator
    20% Skill quality
    15% Behavioral signals
          ↓
  Offline Reasoning Generator    ← no API calls during ranking
          ↓
  submission.csv (Top 100)
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add data
```
data/
├── candidates.jsonl        # candidate profiles (one JSON per line)
└── job_description.txt     # job description text
```

### 3. Run
```bash
python rank.py --candidates ./data/candidates.jsonl --jd ./data/job_description.txt --out ./output/submission.csv
```

With verbose output (shows top 10):
```bash
python rank.py --candidates ./data/candidates.jsonl --jd ./data/job_description.txt --verbose
```

## Output

```
rank, candidate_id, final_score, semantic_score, trajectory_score,
trajectory_pattern, skill_score, behavioral_score, reasoning
```

## Scoring Breakdown

| Component | Weight | What it measures |
|---|---|---|
| Semantic Score | 35% | Embedding cosine similarity between profile and JD |
| Trajectory Score | 30% | Career direction, momentum, AI focus ratio |
| Skill Score | 20% | Depth, assessments, endorsements, JD-specific skills |
| Behavioral Score | 15% | Availability, activity, recruiter response rate |

## JD-Specific Rules (from actual job description)

- ❌ Pure consulting background (TCS/Infosys/Wipro) → penalty
- ❌ Non-technical current title (Marketing/Operations) → heavy penalty
- ❌ Inactive 6+ months + low response rate → penalty
- ❌ Title-chasing (3+ short stints < 12mo) → penalty
- ✅ Product company AI experience → bonus
- ✅ Top-tier institution (IIT/NIT) → small bonus
- ✅ Sub-30-day notice + open to work → bonus
