from dotenv import load_dotenv

load_dotenv()  # must run BEFORE project modules are imported (they read the API key)

import os
import sys

# `streamlit run utils/app.py` only puts utils/ on sys.path. Add the project root (the folder
# that contains core/ and utils/) so the project imports below always resolve.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import time
import traceback
from core import job_store as js
from core.pipeline import run_pipeline, load_saved_result, stage_summary
from core.rag_engine import ask_question

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface-2: #1a1a25;
    --border: #2a2a3a;
    --accent: #7c3aed;
    --accent-glow: #9f67ff;
    --accent-2: #06b6d4;
    --text: #e8e8f0;
    --text-muted: #7070a0;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp {
    background: var(--bg) !important;
}

/* Animated grid background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(124, 58, 237, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124, 58, 237, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

/* ── Headings ── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

/* ── Hero Title ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-glow) 50%, var(--accent-2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}

.card:hover {
    border-color: var(--accent);
}

.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--accent-2));
}

.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.card-content {
    font-size: 0.875rem;
    line-height: 1.7;
    color: var(--text);
}

/* ── Accent Badge ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.badge-purple { background: rgba(124,58,237,0.2); color: var(--accent-glow); border: 1px solid rgba(124,58,237,0.3); }
.badge-cyan   { background: rgba(6,182,212,0.15); color: var(--accent-2);    border: 1px solid rgba(6,182,212,0.3); }
.badge-green  { background: rgba(16,185,129,0.15); color: var(--success);    border: 1px solid rgba(16,185,129,0.3); }

/* ── Input & Buttons ── */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #5b21b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.4) !important;
}

/* Secondary button */
.stButton > button[kind="secondary"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
}

/* ── Progress / Status ── */
.status-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    background: var(--surface-2);
    border-radius: 8px;
    margin: 0.4rem 0;
    border: 1px solid var(--border);
    font-size: 0.8rem;
}

.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

.dot-active   { background: var(--accent-glow); box-shadow: 0 0 8px var(--accent-glow); animation: pulse 1.5s infinite; }
.dot-done     { background: var(--success); }
.dot-pending  { background: var(--border); }
.dot-failed   { background: var(--danger); }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ── Chat ── */
.chat-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    max-height: 420px;
    overflow-y: auto;
    margin-bottom: 1rem;
}

.chat-msg {
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}

.chat-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.chat-bubble {
    display: inline-block;
    padding: 0.6rem 1rem;
    border-radius: 10px;
    font-size: 0.85rem;
    line-height: 1.6;
    max-width: 90%;
}

.user-label  { color: var(--accent-glow); }
.bot-label   { color: var(--accent-2); }

.user-bubble { background: rgba(124,58,237,0.15); border: 1px solid rgba(124,58,237,0.25); align-self: flex-end; }
.bot-bubble  { background: rgba(6,182,212,0.1);  border: 1px solid rgba(6,182,212,0.2);   align-self: flex-start; }

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Transcript box ── */
.transcript-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    font-size: 0.82rem;
    line-height: 1.8;
    max-height: 300px;
    overflow-y: auto;
    color: var(--text-muted);
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── Stale Streamlit elements ── */
.stProgress > div > div > div { background: var(--accent) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
label { color: var(--text-muted) !important; font-size: 0.8rem !important; }

/* scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "processing": False,
    "pipeline_done": False,
    "pipeline_steps": {},
    "transcript": None,
    "errors": [],
    "current_job_id": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helpers ────────────────────────────────────────────────────────────────────
def step_status(steps: dict, key: str) -> str:
    s = steps.get(key, "pending")
    if s == "active":  return "dot-active"
    if s == "done":    return "dot-done"
    if s == "failed":  return "dot-failed"
    return "dot-pending"

def render_step_bar(label: str, key: str, icon: str):
    css = step_status(st.session_state.pipeline_steps, key)
    st.markdown(f"""
    <div class="status-bar">
        <div class="status-dot {css}"></div>
        <span>{icon} {label}</span>
    </div>""", unsafe_allow_html=True)

PIPELINE_STEPS = [
    ("audio",      "🔊", "Audio Processing"),
    ("transcript", "📝", "Transcription"),
    ("title",      "🏷️", "Title Generation"),
    ("summary",    "📋", "Summarisation"),
    ("extract",    "🔍", "Extraction"),
    ("rag",        "🧠", "RAG Engine"),
]

def render_pipeline_status(slot):
    """Redraw the sidebar pipeline status inside an st.empty() slot (live updates)."""
    with slot.container():
        st.markdown("---")
        st.markdown('<span class="badge badge-green">Pipeline Status</span>', unsafe_allow_html=True)
        for step, icon, label in PIPELINE_STEPS:
            render_step_bar(label, step, icon)

def show_transcript(text):
    """The ONLY place the transcript is drawn (plain text area: no HTML parsing problems)."""
    with st.expander("📝 Full Transcript", expanded=True):
        st.text_area("Transcript", value=text, height=300, label_visibility="collapsed")

def render_job_badge(slot):
    """Show the current Job ID in the sidebar (nothing when no job is selected)."""
    jid = st.session_state.current_job_id
    with slot.container():
        if jid:
            st.markdown(
                '<span class="badge badge-cyan">Job ID</span>'
                f'<div class="hero-sub" style="text-transform:none;letter-spacing:0;word-break:break-all">{jid}</div>',
                unsafe_allow_html=True,
            )

def job_label(state):
    """One-line label for the Saved Jobs dropdown."""
    title = ((state.get("results") or {}).get("title") or "Untitled").replace("\n", " ").strip()[:30]
    done = sum(1 for s in js.STAGES if state["stages"][s]["status"] == "done")
    icon = "✅" if done == len(js.STAGES) else "⚠️"
    return f"{icon} {title} · {state['job_id']} · {done}/{len(js.STAGES)}"

def reset_view():
    """Forget everything shown for the previous job so nothing leaks into another job."""
    st.session_state.result = None
    st.session_state.transcript = None
    st.session_state.errors = []
    st.session_state.chat_history = []
    st.session_state.pipeline_steps = {}

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.6rem">🎬 AI<br>Video</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Meeting Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<span class="badge badge-purple">Input</span>', unsafe_allow_html=True)
    source = st.text_input("YouTube URL or File Path", placeholder="https://youtube.com/watch?v=... or /path/to/file.mp4")

    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    run_btn = st.button("⚡  Analyse", use_container_width=True)

    job_slot = st.empty()                       # shows the current Job ID
    render_job_badge(job_slot)

    status_slot = st.empty()
    if st.session_state.pipeline_steps:
        render_pipeline_status(status_slot)

    # ── Saved jobs: open a finished job, or resume one that stopped part-way ──
    st.markdown("---")
    st.markdown('<span class="badge badge-purple">Saved Jobs</span>', unsafe_allow_html=True)
    saved_jobs = js.list_jobs()
    selected_job_id = None
    open_btn = False
    resume_btn = False
    if saved_jobs:
        _by_id = {s["job_id"]: s for s in saved_jobs}
        selected_job_id = st.selectbox(
            "Select a saved job",
            list(_by_id),
            format_func=lambda j: job_label(_by_id[j]),
            key="saved_job_select",
        )
        _c1, _c2 = st.columns(2)
        with _c1:
            open_btn = st.button("📂 Open", use_container_width=True, key="open_job_btn")
        with _c2:
            resume_btn = st.button(
                "▶ Resume", use_container_width=True, key="resume_job_btn",
                disabled=js.is_complete(_by_id[selected_job_id]),
            )
    else:
        st.caption("No saved jobs yet. Analyse a video to create one.")

# ─── Main Area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Transcribe · Summarise · Chat with your meetings</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Run / Resume / Open a job ───────────────────────────────────────────────────
def run_job(state):
    """Run or resume one job. The pipeline skips every stage that is already saved as done."""
    job_id = state["job_id"]
    st.session_state.processing = True
    st.session_state.pipeline_done = False
    reset_view()                                          # nothing from another job stays on screen
    st.session_state.current_job_id = job_id
    st.session_state.pipeline_steps = stage_summary(state)   # show the saved progress right away
    render_job_badge(job_slot)
    render_pipeline_status(status_slot)

    progress_placeholder = st.empty()
    transcript_slot = st.empty()            # live transcript while later steps still run

    def update_step(key, status):
        st.session_state.pipeline_steps[key] = status
        render_pipeline_status(status_slot)

    def on_transcript(text):
        st.session_state.transcript = text   # SAVED BEFORE ANY GEMINI CALL
        with transcript_slot.container():    # show it right now
            show_transcript(text)

    try:
        with progress_placeholder.container():
            st.info("⚙️ Pipeline running — finished stages are skipped…")
        out = run_pipeline(state, on_stage=update_step, on_transcript=on_transcript)
        st.session_state.result = out["result"]
        st.session_state.errors = list(out["errors"])
        if out["result"] is not None:
            st.session_state.transcript = out["result"]["transcript"]
    except Exception as e:
        print(traceback.format_exc())
        st.session_state.errors.append(f"Unexpected error: {e}")
    finally:
        st.session_state.processing = False

    steps = stage_summary(state)
    for k, v in list(steps.items()):
        if v == "active":                   # never show a stuck "active" stage
            steps[k] = "failed"
    st.session_state.pipeline_steps = steps
    st.session_state.pipeline_done = True
    if st.session_state.errors:
        progress_placeholder.warning("⚠️ Finished with errors — details below.")
    else:
        progress_placeholder.success("✅ Analysis complete!")
        time.sleep(0.5)
    # Redraw everything from session state (also removes the live transcript copy,
    # so the transcript is only ever shown once)
    st.rerun()


def open_job(job_id):
    """Load a saved job from disk. No pipeline stage is run."""
    state = js.load_job(job_id)
    if state is None:
        st.session_state.errors = [f"Saved job {job_id} could not be loaded (missing or corrupted state file)."]
        return
    reset_view()
    st.session_state.current_job_id = job_id
    with st.spinner("Opening saved job…"):
        result, errors = load_saved_result(state)
    errors = list(errors)
    if result is not None and result.get("rag_chain") is None:
        errors.append("Chat is unavailable: the RAG index for this job could not be loaded (see the terminal for details).")
    st.session_state.result = result
    st.session_state.transcript = result["transcript"] if result else None
    st.session_state.errors = errors
    st.session_state.pipeline_steps = stage_summary(state)
    st.session_state.pipeline_done = True
    st.rerun()


if not st.session_state.processing:
    if run_btn:
        if not source.strip():
            st.error("Please enter a YouTube URL or file path.")
        else:
            run_job(js.create_or_load_job(source.strip(), language))
    elif resume_btn and selected_job_id:
        _state = js.load_job(selected_job_id)
        if _state is None:
            st.session_state.errors = [f"Saved job {selected_job_id} could not be loaded."]
        else:
            run_job(_state)
    elif open_btn and selected_job_id:
        open_job(selected_job_id)

# ── Errors (always shown; they never hide the transcript) ───────────────────────
for _msg in st.session_state.errors:
    st.error(f"❌ {_msg}")
if st.session_state.errors and st.session_state.current_job_id:
    st.info("Some stages did not finish. Select this job under Saved Jobs and press ▶ Resume to continue from the failed stage.")

# ── Results ──────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result
    _NA = "⚠️ Not available — this step failed (see error above)."
    title_txt        = r.get("title") or "Title unavailable"
    summary_txt      = r.get("summary") or _NA
    action_items_txt = r.get("action_items") or _NA
    decisions_txt    = r.get("key_decisions") or _NA
    questions_txt    = r.get("open_questions") or _NA

    if st.session_state.current_job_id:
        st.markdown(
            f'<div class="hero-sub" style="text-transform:none;letter-spacing:0">Job ID: <code>{st.session_state.current_job_id}</code></div>',
            unsafe_allow_html=True,
        )

    # Title banner
    st.markdown(f"""
    <div class="card">
        <div class="card-title">📌 Session Title</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;color:var(--text)">
            {title_txt}
        </div>
    </div>""", unsafe_allow_html=True)

    # Top row: summary + transcript
    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">📋 Summary</div>
            <div class="card-content">{summary_txt}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        show_transcript(r["transcript"])

    # Second row: action items | decisions | questions
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">✅ Action Items</div>
            <div class="card-content">{action_items_txt}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">🔑 Key Decisions</div>
            <div class="card-content">{decisions_txt}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">❓ Open Questions</div>
            <div class="card-content">{questions_txt}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── RAG Chat ──────────────────────────────────────────────────────────────
    st.markdown('<div style="font-family:\'Syne\',sans-serif;font-size:1.2rem;font-weight:700;margin-bottom:1rem">💬 Chat with your Meeting</div>', unsafe_allow_html=True)

    # Chat history display
    # Chat history display
if st.session_state.chat_history:
    with st.container():
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-label user-label">You</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="chat-bubble user-bubble">{msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="chat-label bot-label">🤖 Assistant</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    '<div class="chat-bubble bot-bubble">',
                    unsafe_allow_html=True
                )
                st.markdown(msg["content"])
                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )
else:
    st.markdown("""
    <div class="card" style="text-align:center;padding:2rem">
        <div style="font-size:2rem;margin-bottom:0.5rem">💬</div>
        <div style="color:var(--text-muted);font-size:0.85rem">
            Ask anything about your meeting transcript
        </div>
    </div>
    """, unsafe_allow_html=True)