"""
AI Candidate Ranking System — Recruiter Web App
================================================
Simple browser-based UI for recruiters.
No command line, no code — just upload and click.

Run locally:
    streamlit run app.py

Deploy free:
    streamlit.io/cloud
"""

import streamlit as st
import json
import csv
import io
import time
import numpy as np
import faiss
from datetime import date, datetime
from sentence_transformers import SentenceTransformer

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Candidate Ranker",
    page_icon="🤖",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { padding: 1rem 2rem; }
    .stButton > button {
        background-color: #1E2761;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 0.7rem 2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
    }
    .stButton > button:hover { background-color: #0D9488; }
    .metric-card {
        background: #F4F6FB;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .metric-number { font-size: 2.5rem; font-weight: bold; color: #1E2761; }
    .metric-label { font-size: 0.9rem; color: #6B7A99; }
    .pattern-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ── Trajectory colors ─────────────────────────────────────────────
PATTERN_COLORS = {
    "FOCUSED_GROWTH":    ("✅", "#16A34A", "#F0FDF4"),
    "STEADY_CONTRIBUTOR":("✅", "#0D9488", "#F0FDFA"),
    "PIVOTING_IN":       ("✅", "#2563EB", "#EFF6FF"),
    "MIXED_AI_FOCUS":    ("🟡", "#D97706", "#FFFBEB"),
    "ADJACENT":          ("🟡", "#9333EA", "#FAF5FF"),
    "CONSULTING_ONLY":   ("❌", "#DC2626", "#FEF2F2"),
    "PIVOTING_OUT":      ("❌", "#EA580C", "#FFF7ED"),
    "UNFOCUSED":         ("❌", "#6B7280", "#F9FAFB"),
    "NO_HISTORY":        ("⚠️",  "#6B7280", "#F9FAFB"),
}

# ── Cached model loader ───────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ── Core pipeline functions ───────────────────────────────────────
def build_text(c):
    parts = []
    p = c.get("profile", {})
    parts.append(f"{p.get('current_title','')} at {p.get('current_company','')} {p.get('years_of_experience',0)}yr")
    if p.get("headline"): parts.append(p["headline"][:100])
    skills = c.get("skills", [])
    top = [s["name"] for s in skills if s.get("proficiency") in ("expert","advanced")][:8]
    if top: parts.append(f"Skills: {', '.join(top)}")
    for role in c.get("career_history", [])[:2]:
        parts.append(f"{role.get('title','')}: {role.get('description','')[:150]}")
    return " | ".join(filter(None, parts))

AI_ML_TITLES = {"ai engineer","ml engineer","machine learning engineer","nlp engineer",
    "applied scientist","research scientist","data scientist","llm engineer",
    "senior ai engineer","senior ml engineer","lead ai engineer","applied ml engineer",
    "deep learning engineer","staff ml engineer"}
MGMT_NONTECH = {"engineering manager","vp engineering","product manager","marketing manager",
    "operations manager","hr manager","sales manager","business analyst","project manager"}
CONSULTING = {"tcs","infosys","wipro","accenture","cognizant","hcl","tech mahindra",
              "capgemini","mphasis","hexaware","ltimindtree"}

def classify(title):
    t = title.lower()
    if any(x in t for x in AI_ML_TITLES): return "ai_ml"
    if any(x in t for x in MGMT_NONTECH): return "non_tech"
    return "other"

def get_seniority(title):
    t = title.lower()
    for kw, lvl in [("principal",5),("staff",4),("lead",4),("senior",3),("junior",1),("intern",0)]:
        if kw in t: return lvl
    return 2

def trajectory(c):
    career = list(reversed(c.get("career_history", [])))
    if not career: return {"score": 0.2, "pattern": "NO_HISTORY", "ai_focus_ratio": 0}
    total = sum(r.get("duration_months",0) for r in career) or 1
    ai_months = sum(r.get("duration_months",0) for r in career if classify(r.get("title",""))=="ai_ml")
    cons_months = sum(r.get("duration_months",0) for r in career if any(x in r.get("company","").lower() for x in CONSULTING))
    ai_ratio = ai_months / total
    cons_ratio = cons_months / total
    cats = [classify(r.get("title","")) for r in career]
    senl = [get_seniority(r.get("title","")) for r in career]
    momentum = min(max(((sum(senl[-2:])/max(len(senl[-2:]),1))-(sum(senl[:2])/max(len(senl[:2]),1))+3)/6,0),1) if len(senl)>=2 else 0.5

    if cons_ratio > 0.85:
        return {"score": 0.15, "pattern": "CONSULTING_ONLY", "ai_focus_ratio": ai_ratio}
    if cats and cats[-1] == "non_tech":
        return {"score": 0.20, "pattern": "PIVOTING_OUT", "ai_focus_ratio": ai_ratio}
    if all(x=="ai_ml" for x in cats) and momentum >= 0.5:
        return {"score": min(1.0, 0.9 + momentum*0.1), "pattern": "FOCUSED_GROWTH", "ai_focus_ratio": ai_ratio}
    if all(x=="ai_ml" for x in cats):
        return {"score": 0.75, "pattern": "STEADY_CONTRIBUTOR", "ai_focus_ratio": ai_ratio}
    if cats and cats[-1]=="ai_ml" and ai_ratio >= 0.3:
        return {"score": 0.70, "pattern": "PIVOTING_IN", "ai_focus_ratio": ai_ratio}
    if ai_ratio >= 0.5:
        return {"score": 0.60, "pattern": "MIXED_AI_FOCUS", "ai_focus_ratio": ai_ratio}
    return {"score": max(0.10, ai_ratio), "pattern": "ADJACENT", "ai_focus_ratio": ai_ratio}

TIER1 = {"embeddings","vector search","faiss","pinecone","rag","semantic search","llm",
         "transformers","nlp","pytorch","fine-tuning","bert","huggingface","ranking","re-ranking"}
PROF_W = {"expert":1.0,"advanced":0.8,"intermediate":0.5,"beginner":0.2}

def skill_score(c):
    score = 0.0
    for s in c.get("skills",[]):
        name = s.get("name","").lower()
        prof = PROF_W.get(s.get("proficiency","beginner"), 0.2)
        end = min(s.get("endorsements",0),100)/100
        trust = prof * (0.6 + 0.4*end)
        if any(t in name for t in TIER1): score += trust
    for sk, val in c.get("redrob_signals",{}).get("skill_assessment_scores",{}).items():
        if any(t in sk.lower() for t in TIER1): score += (val/100)*0.15
    return min(score/3.0, 1.0)

def behavioral_score(c):
    sig = c.get("redrob_signals",{})
    score = 0.0
    if sig.get("open_to_work_flag"): score += 0.25
    notice = sig.get("notice_period_days",90)
    score += 0.20 if notice==0 else 0.15 if notice<=30 else 0.08 if notice<=60 else 0
    score += sig.get("recruiter_response_rate",0)*0.20
    la = sig.get("last_active_date","")
    if la:
        try:
            days = (date.today() - datetime.strptime(la,"%Y-%m-%d").date()).days
            score += 0.15 if days<=7 else 0.10 if days<=30 else 0.05 if days<=90 else 0
        except: pass
    score += sig.get("interview_completion_rate",0)*0.10
    score += min(sig.get("github_activity_score",0)/100,1.0)*0.10
    return min(score, 1.0)

def compute_penalty(c, traj):
    pen = 0.0
    if traj["pattern"] == "CONSULTING_ONLY": pen += 0.40
    if traj["pattern"] == "PIVOTING_OUT": pen += 0.35
    sig = c.get("redrob_signals",{})
    la = sig.get("last_active_date","")
    if la:
        try:
            days = (date.today()-datetime.strptime(la,"%Y-%m-%d").date()).days
            if days>180 and sig.get("recruiter_response_rate",1)<0.1: pen += 0.30
        except: pass
    short = [r for r in c.get("career_history",[]) if r.get("duration_months",12)<12 and not r.get("is_current")]
    if len(short)>=3: pen += 0.15
    return min(pen, 0.6)

def hybrid_score(sem, c):
    traj = trajectory(c)
    ss = skill_score(c)
    bs = behavioral_score(c)
    pen = compute_penalty(c, traj)
    raw = 0.35*sem + 0.30*traj["score"] + 0.20*ss + 0.15*bs
    return {
        "final_score": round(max(raw-pen,0), 4),
        "semantic_score": round(sem, 4),
        "trajectory_score": round(traj["score"], 4),
        "trajectory_pattern": traj["pattern"],
        "skill_score": round(ss, 4),
        "behavioral_score": round(bs, 4),
        "ai_focus_ratio": round(traj["ai_focus_ratio"], 4),
    }

def make_reasoning(c, sc):
    p = c.get("profile",{})
    sig = c.get("redrob_signals",{})
    skills = [s["name"] for s in sorted(c.get("skills",[]),
              key=lambda x: {"expert":3,"advanced":2,"intermediate":1,"beginner":0}.get(x.get("proficiency","beginner"),0),
              reverse=True)][:4]
    notice = sig.get("notice_period_days",90)
    avail = "immediately available" if notice==0 and sig.get("open_to_work_flag") else f"{notice}d notice"
    s1 = f"{p.get('current_title','N/A')} at {p.get('current_company','N/A')} | {p.get('years_of_experience',0)}yr exp | Skills: {', '.join(skills)}"
    s2 = f"Trajectory: {sc['trajectory_pattern']} ({sc['trajectory_score']:.2f}) | AI focus: {sc['ai_focus_ratio']*100:.0f}%"
    s3 = f"Availability: {avail} | Response rate: {sig.get('recruiter_response_rate',0)*100:.0f}%"
    return f"{s1}\n{s2}\n{s3}"

# ════════════════════════════════════════════════════════════════
# UI STARTS HERE
# ════════════════════════════════════════════════════════════════

st.title("🤖 AI Candidate Ranking System")
st.markdown("*Understanding candidates the way a great recruiter would — not by keywords, but by career trajectories.*")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_n = st.slider("Top N candidates to show", 10, 100, 100, 10)
    pool_size = st.slider("Semantic search pool", 100, 1000, 500, 100)
    st.divider()
    st.markdown("**Scoring Weights**")
    st.markdown("🔵 Semantic: **35%**")
    st.markdown("🟢 Trajectory: **30%**")
    st.markdown("🔴 Skills: **20%**")
    st.markdown("🟡 Behavioral: **15%**")
    st.divider()
    st.markdown("**Trajectory Patterns**")
    for pat, (emoji, color, _) in PATTERN_COLORS.items():
        st.markdown(f"{emoji} `{pat}`")

# ── Main layout ───────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📁 Step 1: Upload Candidates")
    candidates_file = st.file_uploader(
        "Upload candidates.jsonl",
        type=["jsonl", "json"],
        help="One candidate JSON per line"
    )
    if candidates_file:
        st.success(f"✅ File uploaded: {candidates_file.name}")

with col2:
    st.subheader("📝 Step 2: Job Description")
    jd_option = st.radio("Choose input method", ["Paste text", "Upload file"], horizontal=True)

    if jd_option == "Paste text":
        jd_text = st.text_area(
            "Paste Job Description here",
            height=200,
            placeholder="Senior AI Engineer role requiring embeddings, RAG, vector search..."
        )
    else:
        jd_file = st.file_uploader("Upload job_description.txt", type=["txt", "docx"])
        jd_text = ""
        if jd_file:
            jd_text = jd_file.read().decode("utf-8")
            st.success(f"✅ JD uploaded: {jd_file.name}")

st.divider()

# ── Run button ────────────────────────────────────────────────────
run_col = st.columns([1, 2, 1])[1]
with run_col:
    run_clicked = st.button("▶ Run AI Ranking", use_container_width=True)

# ── Pipeline execution ────────────────────────────────────────────
if run_clicked:
    if not candidates_file:
        st.error("❌ Please upload candidates.jsonl file first!")
        st.stop()
    if not jd_text or len(jd_text.strip()) < 20:
        st.error("❌ Please provide a job description!")
        st.stop()

    # Load candidates
    with st.spinner("📂 Loading candidates..."):
        candidates = []
        content = candidates_file.read().decode("utf-8")
        for line in content.strip().split("\n"):
            if line.strip():
                candidates.append(json.loads(line.strip()))
        candidate_map = {c["candidate_id"]: c for c in candidates}
        st.success(f"✅ Loaded **{len(candidates):,}** candidates")

    # Load model
    with st.spinner("🧠 Loading AI model (first time takes 30s)..."):
        model = load_model()
        st.success("✅ AI model ready")

    # Embed
    with st.spinner(f"⚡ Embedding {len(candidates):,} candidates..."):
        t0 = time.time()
        texts = [build_text(c) for c in candidates]
        candidate_ids = [c["candidate_id"] for c in candidates]
        vectors = model.encode(texts, batch_size=512, show_progress_bar=False,
                               normalize_embeddings=True, convert_to_numpy=True)
        t1 = time.time()
        st.success(f"✅ Embedded in {t1-t0:.1f}s")

    # FAISS
    with st.spinner("🔍 Building search index..."):
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors.astype(np.float32))
        jd_vec = model.encode([jd_text], normalize_embeddings=True, convert_to_numpy=True)[0]
        k = min(pool_size, len(candidates))
        scores_arr, indices_arr = index.search(np.array([jd_vec], dtype=np.float32), k)
        top_results = [(candidate_ids[idx], float(sc)) for sc, idx in zip(scores_arr[0], indices_arr[0]) if idx >= 0]
        st.success(f"✅ Retrieved top {len(top_results)} candidates")

    # Score
    with st.spinner("📊 Scoring candidates..."):
        scored = []
        for cid, sem in top_results:
            c = candidate_map.get(cid)
            if c:
                sc = hybrid_score(sem, c)
                scored.append((cid, sc))
        scored.sort(key=lambda x: x[1]["final_score"], reverse=True)
        top_candidates = scored[:top_n]

    # Reasoning
    with st.spinner("✍️ Generating reasoning..."):
        results = []
        for rank, (cid, sc) in enumerate(top_candidates, 1):
            c = candidate_map[cid]
            p = c.get("profile", {})
            results.append({
                "rank":               rank,
                "candidate_id":       cid,
                "name":               p.get("name", cid),
                "current_title":      p.get("current_title", "N/A"),
                "current_company":    p.get("current_company", "N/A"),
                "years_exp":          p.get("years_of_experience", 0),
                "final_score":        sc["final_score"],
                "semantic_score":     sc["semantic_score"],
                "trajectory_pattern": sc["trajectory_pattern"],
                "trajectory_score":   sc["trajectory_score"],
                "skill_score":        sc["skill_score"],
                "behavioral_score":   sc["behavioral_score"],
                "ai_focus_ratio":     sc["ai_focus_ratio"],
                "reasoning":          make_reasoning(c, sc),
            })

    total_time = time.time() - t0

    # ── RESULTS ───────────────────────────────────────────────────
    st.divider()
    st.header("🏆 Results")

    # Metric cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-number">{len(candidates):,}</div><div class="metric-label">Candidates Processed</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-number">{total_time:.0f}s</div><div class="metric-label">Total Runtime</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-number">{len(results)}</div><div class="metric-label">Top Candidates</div></div>', unsafe_allow_html=True)
    with m4:
        top_pattern = results[0]["trajectory_pattern"] if results else "N/A"
        emoji = PATTERN_COLORS.get(top_pattern, ("✅","",""))[0]
        st.markdown(f'<div class="metric-card"><div class="metric-number">{emoji}</div><div class="metric-label">Top Candidate: {top_pattern}</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # Download button
    output = io.StringIO()
    fieldnames = ["rank","candidate_id","final_score","semantic_score","trajectory_score",
                  "trajectory_pattern","skill_score","behavioral_score","reasoning"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow({k: r[k] for k in fieldnames})

    st.download_button(
        label="⬇️ Download submission.csv",
        data=output.getvalue(),
        file_name="submission.csv",
        mime="text/csv",
        use_container_width=False
    )

    st.divider()

    # Results table
    st.subheader(f"Top {len(results)} Candidates")

    for r in results:
        emoji, color, bg = PATTERN_COLORS.get(r["trajectory_pattern"], ("⚪","#6B7280","#F9FAFB"))
        with st.expander(f"#{r['rank']}  {r['current_title']} @ {r['current_company']}  |  Score: {r['final_score']:.4f}  |  {emoji} {r['trajectory_pattern']}"):

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Final Score", f"{r['final_score']:.3f}")
            c2.metric("Semantic", f"{r['semantic_score']:.3f}")
            c3.metric("Trajectory", f"{r['trajectory_score']:.3f}")
            c4.metric("Skills", f"{r['skill_score']:.3f}")

            st.markdown(f"**Candidate ID:** `{r['candidate_id']}` | **Experience:** {r['years_exp']} years | **AI Focus:** {r['ai_focus_ratio']*100:.0f}%")
            st.markdown(f"**Reasoning:**")
            st.code(r["reasoning"], language=None)

else:
    # Instructions when nothing is running
    st.info("👆 Upload your candidates file and job description above, then click **Run AI Ranking**")

    st.markdown("### How it works")
    cols = st.columns(4)
    steps = [
        ("1️⃣", "Upload", "candidates.jsonl file with all candidate profiles"),
        ("2️⃣", "Describe", "Paste or upload the Job Description"),
        ("3️⃣", "Run", "Click the button — AI does the rest"),
        ("4️⃣", "Download", "Get ranked top-100 CSV instantly"),
    ]
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"### {num} {title}")
            st.markdown(desc)
