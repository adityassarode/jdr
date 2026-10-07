from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from config.skills import ALL_SKILLS, GENERIC_TFIDF_EXCLUSIONS, SKILL_VOCABULARY
from frequency_analysis import calculate_word_frequency
from preprocessing import ensure_nltk_resources, preprocess_stages
from regex_extractor import extract_contact_information
from similarity import calculate_similarity
from skill_matcher import classify_skill_priority, compare_skills, extract_skills
from tfidf_analysis import calculate_tfidf


st.set_page_config(page_title="Resume Match", page_icon="✦", layout="wide", initial_sidebar_state="expanded")
ensure_nltk_resources()

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
    :root { color-scheme:light; --ink:#121a24; --muted:#657080; --paper:#f4f7fb; --card:#ffffff; --line:#dfe6ef; --accent:#1468e8; --accent-soft:#e9f1ff; --coral:#f26051; }
    html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
    .stApp { background:linear-gradient(135deg,#f6f8fb 0%,#eef3f8 52%,#f9fbfd 100%); color:var(--ink); }
    [data-testid="stHeader"] { background:rgba(246,248,251,.86); }
    [data-testid="stToolbar"] { visibility:hidden; }
    [data-testid="stSidebar"] { background:#f8fafc !important; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] * { color:var(--ink) !important; }
    [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small { color:var(--muted) !important; }
    .block-container { max-width:1240px; padding:3.4rem 3.2rem 5rem; }
    h1,h2,h3 { font-family:'Manrope', sans-serif; letter-spacing:-.045em; color:var(--ink) !important; }
    h1 { font-size:clamp(2.5rem,5vw,4.4rem); line-height:.98; margin:.25rem 0 .8rem; }
    h2 { font-size:1.5rem; }
    h3 { font-size:1.05rem; letter-spacing:-.02em; }
    p, label, .stMarkdown, [data-testid="stCaptionContainer"] { color:var(--ink); }
    .eyebrow { color:var(--accent); font-weight:700; font-size:.72rem; letter-spacing:.16em; text-transform:uppercase; }
    .subtitle { color:var(--muted) !important; font-size:1.05rem; max-width:680px; line-height:1.6; }
    .hero-note { color:var(--muted); font-size:.8rem; margin-top:1.6rem; }
    .input-heading { display:flex; align-items:center; gap:.6rem; margin:.2rem 0 .7rem; }
    .input-number { display:grid; place-items:center; width:1.65rem; height:1.65rem; border-radius:50%; background:var(--ink); color:#fff; font-size:.78rem; font-weight:700; }
    [data-testid="stFileUploader"] { margin:.25rem 0 .65rem; }
    [data-testid="stFileUploader"] label { color:var(--muted) !important; font-size:.78rem !important; font-weight:600 !important; }
    [data-testid="stFileUploaderDropzone"] { min-height:3.3rem; background:#f4f7fb !important; border:1px dashed #aebed1 !important; border-radius:13px !important; }
    [data-testid="stFileUploaderDropzone"] > div { background:#f4f7fb !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] { color:#536274 !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] span, [data-testid="stFileUploaderDropzoneInstructions"] small { color:#536274 !important; }
    [data-testid="stFileUploaderDropzone"] button { background:#fff !important; border:1px solid #c5d2e0 !important; color:var(--ink) !important; }
    [data-testid="stFileUploaderDropzone"] button span { color:var(--ink) !important; }
    [data-testid="stFileUploaderFile"] { background:#fff !important; color:var(--ink) !important; }
    [data-testid="stTextArea"] textarea { background:#fff !important; color:var(--ink) !important; border:1px solid #cbd6e2 !important; border-radius:14px !important; min-height:210px; }
    [data-testid="stTextArea"] textarea:focus { border-color:var(--accent) !important; box-shadow:0 0 0 3px rgba(20,104,232,.12) !important; }
    [data-testid="stTextArea"] textarea::placeholder { color:#8894a3 !important; opacity:1; }
    .stButton > button { border-radius:12px; font-weight:700; min-height:3rem; transition:transform .15s ease, box-shadow .15s ease; }
    .stButton > button[kind="primary"] { background:var(--ink) !important; color:#fff !important; border:1px solid var(--ink) !important; }
    .stButton > button:not([kind="primary"]) { background:#fff !important; color:var(--ink) !important; border:1px solid #ccd7e4 !important; }
    .stButton > button:hover { transform:translateY(-1px); box-shadow:0 9px 20px rgba(18,26,36,.14); }
    .action-copy { color:var(--muted); font-size:.85rem; padding-top:.8rem; }
    .glass { background:rgba(255,255,255,.82); border:1px solid rgba(218,226,236,.95); box-shadow:0 18px 50px rgba(28,30,34,.07); border-radius:18px; padding:1.15rem 1.25rem; }
    .metric { font-family:'Manrope'; font-size:2rem; font-weight:800; letter-spacing:-.04em; }
    .metric-label { color:var(--muted); font-size:.82rem; margin-top:.25rem; }
    .metric-note { color:var(--muted); font-size:.7rem; line-height:1.35; margin-top:.35rem; }
    .pill { display:inline-block; background:var(--accent-soft); color:#1257bc !important; border-radius:999px; padding:.35rem .7rem; margin:.2rem .2rem .2rem 0; font-size:.82rem; font-weight:600; }
    .empty { color:var(--muted) !important; font-style:italic; }
    [data-testid="stAlert"] { border-radius:14px; }
    [data-testid="stTabs"] button { color:var(--muted) !important; }
    [data-testid="stTabs"] button[aria-selected="true"] { color:var(--ink) !important; }
    @media (prefers-reduced-motion: reduce) { *,*::before,*::after { scroll-behavior:auto !important; transition:none !important; animation:none !important; } }
    @media (max-width: 800px) { .block-container { padding:2rem 1.1rem 3rem; } h1 { font-size:2.8rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_upload(uploaded_file) -> str:
    if not uploaded_file:
        return ""
    if Path(uploaded_file.name).suffix.lower() != ".txt":
        st.error("Please upload a .txt file.")
        return ""
    try:
        return uploaded_file.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        st.error("This file is not valid UTF-8 text.")
        return ""


def show_pills(items: set[str], empty_message: str = "None detected") -> None:
    if items:
        st.markdown(" ".join(f'<span class="pill">{item}</span>' for item in sorted(items)), unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="empty">{empty_message}</span>', unsafe_allow_html=True)


def frequency_chart(items: list[tuple[str, int]], title: str) -> None:
    if not items:
        st.info("No terms available to chart.")
        return
    frame = pd.DataFrame(items, columns=["Term", "Frequency"]).set_index("Term")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    frame.sort_values("Frequency").plot.barh(ax=ax, legend=False, color="#0b6cff")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=.15)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def important_job_terms(frame: pd.DataFrame, resume_text: str, job_text: str, resume_tokens: list[str]) -> list[str]:
    job_skills = extract_skills(job_text)
    resume_skills = extract_skills(resume_text)
    missing_skill_phrases = sorted(job_skills - resume_skills)
    phrase_components = {word.lower() for phrase in job_skills for word in phrase.split()}
    resume_vocabulary = set(resume_tokens)
    candidates = frame[(frame["Job Description TF-IDF"] > 0) & (~frame["Term"].isin(resume_vocabulary))]
    candidates = candidates[~candidates["Term"].isin(GENERIC_TFIDF_EXCLUSIONS)]
    candidates = candidates[~candidates["Term"].isin(phrase_components)]
    candidates = candidates[candidates["Term"].str.len() >= 3]
    raw_terms = candidates.sort_values("Job Description TF-IDF", ascending=False)["Term"].head(12).tolist()
    return missing_skill_phrases + [term for term in raw_terms if term not in missing_skill_phrases]


def common_vocabulary(resume_tokens: list[str], job_tokens: list[str], resume_text: str, job_text: str) -> set[str]:
    shared = (set(resume_tokens) & set(job_tokens)) - GENERIC_TFIDF_EXCLUSIONS
    phrase_components = {word.lower() for phrase in ALL_SKILLS for word in phrase.split() if " " in phrase}
    shared_skills = extract_skills(resume_text) & extract_skills(job_text)
    return shared_skills | {term for term in shared if term not in phrase_components and len(term) >= 3}


def render_skill_coverage(analysis: dict) -> None:
    total = len(analysis["job_skills"])
    matched = len(analysis["comparison"]["common"])
    missing = len(analysis["comparison"]["missing"])
    st.markdown("### Job skill coverage")
    st.caption(f"Total job skills: {total}  ·  Matched: {matched}  ·  Missing: {missing}")
    st.progress((matched / total) if total else 0.0)


def render_skill_group(title: str, skills: set[str], empty_message: str) -> None:
    st.markdown(f"**{title}**")
    show_pills(skills, empty_message)


with st.sidebar:
    st.markdown("## Resume Match")
    st.caption("Group 9 · Academic NLP project")
    st.divider()
    st.markdown("### Techniques")
    st.markdown("Preprocessing  ·  Regex  ·  Frequency  ·  TF-IDF  ·  Similarity")
    st.markdown("### Skill vocabulary")
    st.caption(f"{len(ALL_SKILLS)} editable terms across {len(SKILL_VOCABULARY)} categories")
    for category, skills in SKILL_VOCABULARY.items():
        st.caption(f"**{category}**: {', '.join(skills)}")
    st.divider()
    st.caption("No supervised model is used. Scores are computed from the text you provide.")

st.markdown('<div class="eyebrow">Group 9 · NLP laboratory</div>', unsafe_allow_html=True)
st.title("Resume, meet the role.")
st.markdown('<p class="subtitle">See where a candidate’s experience overlaps with a role, using transparent NLP instead of invented scores.</p>', unsafe_allow_html=True)
st.markdown('<div class="hero-note">Two documents in · one clear comparison out</div>', unsafe_allow_html=True)
st.write("")

left, right = st.columns(2, gap="large")
with left:
    st.markdown("### 1 / Candidate resume")
    st.caption("Paste text or upload a plain-text file")
    resume_upload = st.file_uploader("Optional text file", type=["txt"], key="resume_file")
    resume_text = st.text_area("Resume text", height=220, placeholder="Paste the candidate's resume here…", label_visibility="collapsed")
    if resume_upload:
        resume_text = read_upload(resume_upload)
with right:
    st.markdown("### 2 / Job description")
    st.caption("Paste the role requirements or upload a plain-text file")
    job_upload = st.file_uploader("Optional text file", type=["txt"], key="job_file")
    job_text = st.text_area("Job description text", height=220, placeholder="Paste the role's job description here…", label_visibility="collapsed")
    if job_upload:
        job_text = read_upload(job_upload)

analyze, clear = st.columns([1, 1])
with analyze:
    analyze_clicked = st.button("Analyze resume", type="primary", use_container_width=True)
with clear:
    if st.button("Clear", use_container_width=True):
        st.session_state.clear()
        st.rerun()
st.markdown('<div class="action-copy">Your text stays local to this session. Analysis begins only when you choose the button.</div>', unsafe_allow_html=True)

if analyze_clicked:
    if not resume_text.strip():
        st.error("Please enter or upload a resume.")
    elif not job_text.strip():
        st.error("Please enter or upload a job description.")
    elif len(resume_text.split()) < 3 or len(job_text.split()) < 3:
        st.error("Please provide more text in both documents so the comparison is meaningful.")
    else:
        with st.spinner("Processing your documents…"):
            resume_stages = preprocess_stages(resume_text)
            job_stages = preprocess_stages(job_text)
            resume_tokens = list(resume_stages["lemmatized"])
            job_tokens = list(job_stages["lemmatized"])
            resume_skills = extract_skills(resume_text)
            job_skills = extract_skills(job_text)
            skill_comparison = compare_skills(resume_skills, job_skills)
            job_priority = classify_skill_priority(job_text, job_skills)
            resume_frequency = calculate_word_frequency(resume_tokens, top_n=20)
            job_frequency = calculate_word_frequency(job_tokens, top_n=20)
            tfidf_frame, matrix = calculate_tfidf(resume_tokens, job_tokens)
            similarity = calculate_similarity(matrix[0:1], matrix[1:2])
            skill_match = (len(skill_comparison["common"]) / len(job_skills) * 100) if job_skills else 0.0
            st.session_state["analysis"] = {
                "resume_text": resume_text, "job_text": job_text, "resume_tokens": resume_tokens, "job_tokens": job_tokens,
                "resume_stages": resume_stages, "job_stages": job_stages,
                "contacts": extract_contact_information(resume_text), "resume_skills": resume_skills, "job_skills": job_skills,
                "comparison": skill_comparison, "resume_frequency": resume_frequency, "job_frequency": job_frequency,
                "job_priority": job_priority, "tfidf": tfidf_frame, "similarity": similarity, "skill_match": skill_match,
                "important_job_terms": important_job_terms(tfidf_frame, resume_text, job_text, resume_tokens),
                "common_vocabulary": common_vocabulary(resume_tokens, job_tokens, resume_text, job_text),
            }

analysis = st.session_state.get("analysis")
if analysis:
    st.divider()
    st.markdown('<div class="eyebrow">Analysis complete</div>', unsafe_allow_html=True)
    st.header("Match summary")
    metrics = st.columns(5)
    job_skill_count = len(analysis["job_skills"])
    matched_skill_count = len(analysis["comparison"]["common"])
    skill_value = f"{analysis['skill_match']:.1f}%" if job_skill_count else "N/A"
    skill_note = f"{matched_skill_count} of {job_skill_count} detected skills" if job_skill_count else "No recognized job skills"
    metric_values = [
        (f"{analysis['similarity']:.1%}", "TF-IDF similarity", "Overall textual similarity"),
        (skill_value, "Job skill match", skill_note),
        (str(matched_skill_count), "Common skills", "Detected overlap"),
        (str(len(analysis["comparison"]["missing"])), "Missing skills", "Detected job gaps"),
        (str(len(analysis["resume_skills"])), "Resume skills", "Configured skills found"),
    ]
    for column, (value, label, note) in zip(metrics, metric_values):
        with column:
            st.markdown(f'<div class="glass"><div class="metric">{value}</div><div class="metric-label">{label}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

    tabs = st.tabs(["Overview", "Skill analysis", "Resume information", "Preprocessing", "Frequency", "TF-IDF", "Methodology"])
    with tabs[0]:
        st.markdown("### Readout")
        missing_names = ", ".join(sorted(analysis["comparison"]["missing"]))
        if job_skill_count:
            readout = f"{matched_skill_count} of {job_skill_count} job-related skills detected in the job description were also found in the resume."
            if missing_names:
                readout += f" Missing detected skills include {missing_names}."
        else:
            readout = "No recognized job-related skills were detected, so skill match cannot be calculated."
        readout += f" TF-IDF cosine similarity between the two documents is {analysis['similarity']:.1%}."
        st.markdown("### Overall analysis")
        st.write(readout)
        total = len(analysis["job_skills"])
        matched = len(analysis["comparison"]["common"])
        st.markdown("### Job skill coverage")
        st.caption(f"Total job skills: {total}  ·  Matched: {matched}  ·  Missing: {total - matched}")
        st.progress((matched / total) if total else 0.0)
        st.info("**TF-IDF similarity vs job skill match**\n\nTF-IDF similarity measures overall textual similarity, while job skill match measures the proportion of detected job-related skills found in the resume. The two percentages measure different aspects and are not expected to be equal.")
        st.markdown("### How is job skill match calculated?")
        if job_skill_count:
            st.code(f"matched job skills / detected job skills × 100\n{matched_skill_count} / {job_skill_count} × 100 = {analysis['skill_match']:.1f}%")
        else:
            st.warning("No recognized job-related skills were detected, so skill match cannot be calculated.")
        st.markdown("### Strong match areas")
        show_pills(analysis["comparison"]["common"], "No common skills detected")
        st.markdown("### Skill gaps")
        show_pills(analysis["comparison"]["missing"], "No missing skills were detected from the configured skill vocabulary.")
        if analysis["comparison"]["common"]:
            st.markdown("### Resume strengths")
            st.write(f"The resume matches {matched_skill_count} of {job_skill_count} detected job-related skills, including {', '.join(sorted(analysis['comparison']['common']))}.")
        st.markdown("### Common vocabulary")
        st.caption("These are words shared by both documents after preprocessing. They are not necessarily skills.")
        show_pills(analysis["common_vocabulary"], "No meaningful common vocabulary detected")
        st.markdown("### Important job terms not found in resume")
        st.caption("Actual TF-IDF terms only, with generic recruitment language excluded.")
        show_pills(set(analysis["important_job_terms"]), "No filtered TF-IDF terms detected")
    with tabs[1]:
        st.markdown("### Required skills")
        show_pills(analysis["job_priority"]["required"], "No reliable required-skill cue detected")
        st.markdown("### Preferred skills")
        show_pills(analysis["job_priority"]["preferred"], "No reliable preferred-skill cue detected")
        st.markdown("### Common skills")
        show_pills(analysis["comparison"]["common"], "None detected")
        st.markdown("### Missing job skills")
        show_pills(analysis["comparison"]["missing"], "None detected")
        st.markdown("### Additional resume skills")
        show_pills(analysis["comparison"]["additional"], "None detected")
        st.markdown("### Job skill coverage")
        st.progress((len(analysis["comparison"]["common"]) / len(analysis["job_skills"])) if analysis["job_skills"] else 0.0)
    with tabs[2]:
        st.markdown("### Contact details detected with regex")
        st.dataframe(pd.DataFrame(analysis["contacts"].items(), columns=["Field", "Value"]), hide_index=True, use_container_width=True)
    with tabs[3]:
        st.markdown("### Actual preprocessing pipeline")
        st.caption("Each stage below is calculated from the current resume and job description.")
        a, b = st.columns(2)
        with a:
            stages = analysis["resume_stages"]
            st.markdown("**Resume**")
            st.markdown("**Original text**"); st.code(stages["original"])
            st.markdown("**Lowercase** · standardizes letter case"); st.code(stages["lowercase"])
            st.markdown("**Cleaned text** · removes punctuation and extra whitespace"); st.code(stages["cleaned"])
            st.markdown("**Tokens** · splits text into individual terms"); st.code(" | ".join(stages["tokens"]))
            st.markdown("**Stop-word removal** · removes common words such as the, is, and"); st.code(" ".join(stages["without_stopwords"]))
            st.markdown("**Lemmatized tokens** · reduces words to base forms where available"); st.code(" ".join(stages["lemmatized"]))
        with b:
            stages = analysis["job_stages"]
            st.markdown("**Job description**")
            st.markdown("**Original text**"); st.code(stages["original"])
            st.markdown("**Lowercase** · standardizes letter case"); st.code(stages["lowercase"])
            st.markdown("**Cleaned text** · removes punctuation and extra whitespace"); st.code(stages["cleaned"])
            st.markdown("**Tokens** · splits text into individual terms"); st.code(" | ".join(stages["tokens"]))
            st.markdown("**Stop-word removal** · removes common words such as the, is, and"); st.code(" ".join(stages["without_stopwords"]))
            st.markdown("**Lemmatized tokens** · reduces words to base forms where available"); st.code(" ".join(stages["lemmatized"]))
    with tabs[4]:
        top_n = st.select_slider("Show top terms", options=[5, 10, 15, 20], value=10)
        a, b = st.columns(2)
        with a:
            items = analysis["resume_frequency"][:top_n]
            st.markdown("### Resume word frequency"); st.dataframe(pd.DataFrame(items, columns=["Term", "Frequency"]), hide_index=True, use_container_width=True); frequency_chart(items, "Resume terms")
        with b:
            items = analysis["job_frequency"][:top_n]
            st.markdown("### Job description word frequency"); st.dataframe(pd.DataFrame(items, columns=["Term", "Frequency"]), hide_index=True, use_container_width=True); frequency_chart(items, "Job terms")
    with tabs[5]:
        st.markdown("### Top resume terms")
        resume_terms = analysis["tfidf"].sort_values("Resume TF-IDF", ascending=False).head(20)
        st.dataframe(resume_terms[["Term", "Resume TF-IDF", "Job Description TF-IDF"]], hide_index=True, use_container_width=True)
        st.markdown("### Top job-description terms")
        job_terms = analysis["tfidf"].sort_values("Job Description TF-IDF", ascending=False).head(20)
        st.dataframe(job_terms[["Term", "Resume TF-IDF", "Job Description TF-IDF"]], hide_index=True, use_container_width=True)
    with tabs[6]:
        st.markdown("### Transparent pipeline")
        st.code("INPUT\n  ↓\nPREPROCESSING\n  ↓\nREGEX EXTRACTION\n  ↓\nWORD FREQUENCY\n  ↓\nTF-IDF\n  ↓\nCOSINE SIMILARITY\n  ↓\nSKILL COMPARISON\n  ↓\nRESULTS")
        st.info("No supervised machine-learning model is used. TF-IDF and cosine similarity compare the supplied documents. Skill detection depends on the editable vocabulary, exact wording affects matching, and regex depends on recognizable patterns.")
        st.markdown("### Limitations")
        st.markdown("- Skill detection depends on the configured skill vocabulary.\n- TF-IDF is based on statistical word importance.\n- Cosine similarity measures textual similarity, not candidate ability.\n- Synonyms may not be recognized automatically.\n- Regex depends on recognizable patterns.\n- Results should not be used as an automatic hiring decision.")
else:
    st.info("Add both documents, then choose Analyze resume. Nothing is calculated until you ask for it.")
