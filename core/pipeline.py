"""
Resumable pipeline for the AI Video Assistant.

Runs the existing functions (process_input, transcribe_all, generate_title,
summarize, extract_*, build_rag_chain) stage by stage. Before each stage the
saved job state is checked, so completed stages are never repeated, and every
successful stage is written to disk immediately.

This module has no Streamlit dependency: app.py passes small callbacks for UI updates.
"""

import os
import traceback

from core import job_store as js
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain

STAGE_LABELS = {
    "audio": "Audio processing",
    "transcript": "Transcription",
    "title": "Title generation",
    "summary": "Summarisation",
    "extract": "Extraction",
    "rag": "RAG engine setup",
}


# ─── Small helpers ──────────────────────────────────────────────────────────────
def _is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple)):
        return len(value) == 0
    return False


def _clean(value):
    """Make a result JSON-safe and reject empty answers."""
    if _is_blank(value):
        raise RuntimeError("The model returned an empty response.")
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, dict)):
        return value
    return str(value)


def _has_result(state: dict, key: str) -> bool:
    return not _is_blank(js.get_result(state, key))


def collect_errors(state: dict) -> list:
    """Human-readable error messages for every failed stage of a job."""
    errors = []
    for stage in js.STAGES:
        info = state["stages"][stage]
        if info["status"] == "failed":
            errors.append(f"{STAGE_LABELS[stage]} failed: {info.get('error') or 'unknown error'}")
    return errors


def stage_summary(state: dict) -> dict:
    """{'audio': 'done', ...} for the sidebar pipeline status."""
    return {s: state["stages"][s]["status"] for s in js.STAGES}


def build_result(state: dict, transcript: str, rag_chain) -> dict:
    """The dict shape app.py already uses in st.session_state.result."""
    r = state["results"]
    return {
        "title": r.get("title"),
        "transcript": transcript,
        "summary": r.get("summary"),
        "action_items": r.get("action_items"),
        "key_decisions": r.get("key_decisions"),
        "open_questions": r.get("open_questions"),
        "rag_chain": rag_chain,
    }


def _reusable_chunks(state: dict):
    chunks = state.get("audio_chunks") or []
    if (
        js.stage_status(state, "audio") == "done"
        and chunks
        and all(os.path.exists(c) for c in chunks)
    ):
        return chunks
    return None


# ─── Load a saved job without running anything ──────────────────────────────────
def load_saved_result(state: dict):
    """
    Rebuild the result dict for a saved job from disk. The RAG chain is rebuilt from the
    saved transcript (it is never stored). Returns (result_or_None, errors).
    """
    transcript = js.load_transcript(state["job_id"])
    errors = collect_errors(state)
    if transcript is None:
        return None, errors

    rag_chain = None
    try:
        rag_chain = build_rag_chain(transcript, state["job_id"])
    except Exception:
        print(traceback.format_exc())
    return build_result(state, transcript, rag_chain), errors


# ─── The resumable pipeline ─────────────────────────────────────────────────────
def run_pipeline(state: dict, on_stage=None, on_transcript=None) -> dict:
    """
    Run only the stages that are not done yet.

    on_stage(stage, status)   called whenever a stage changes status (for the UI)
    on_transcript(text)       called as soon as the transcript is available

    Returns {'result': dict_or_None, 'errors': [str]}.
    After the first failed Gemini stage, the remaining Gemini stages are left
    "pending" (no point hammering a rate-limited API). Saved results are never touched.
    """
    job_id = state["job_id"]
    source = state["source"]
    language = state["language"]

    def mark(stage, status, error=None, code=None):
        js.set_stage(state, stage, status, error, code)
        if on_stage:
            on_stage(stage, status)

    def fail(stage, exc):
        print(traceback.format_exc())
        code, message = js.classify_error(exc)
        if stage in ("audio", "transcript") and code in ("rate_limit", "unavailable"):
            # the Gemini-specific wording does not apply to download / Whisper / Sarvam
            code, message = "error", str(exc)[:300] or type(exc).__name__
        mark(stage, "failed", message, code)

    def finish(transcript, rag_chain):
        result = build_result(state, transcript, rag_chain) if transcript else None
        return {"result": result, "errors": collect_errors(state)}

    # ── 1-2. Audio + transcription (skipped entirely when the transcript is saved) ──
    transcript = None
    if js.stage_status(state, "transcript") == "done":
        transcript = js.load_transcript(job_id)
        if transcript is None:  # state says done but the file is gone/empty: redo it
            mark("transcript", "pending")

    if transcript is None:
        chunks = _reusable_chunks(state)
        if chunks is None:
            mark("audio", "active")
            try:
                chunks = process_input(source)
                if not chunks:
                    raise RuntimeError("No audio chunks were created.")
                js.save_audio_chunks(state, chunks)
            except Exception as e:
                fail("audio", e)
                return finish(None, None)
            mark("audio", "done")

        mark("transcript", "active")
        try:
            transcript = transcribe_all(chunks, language)
            if not transcript or not str(transcript).strip():
                raise RuntimeError("Transcription returned an empty transcript.")
            js.save_transcript(job_id, transcript)  # saved before any Gemini call
        except Exception as e:
            transcript = None
            fail("transcript", e)
            return finish(None, None)
        mark("transcript", "done")

    if on_transcript:
        on_transcript(transcript)

    # ── 3-5. Gemini stages: each success is saved immediately ──
    gemini_ok = True

    # Title
    if gemini_ok and not (js.stage_status(state, "title") == "done" and _has_result(state, "title")):
        mark("title", "active")
        try:
            js.save_result(state, "title", _clean(generate_title(transcript)))
            mark("title", "done")
        except Exception as e:
            fail("title", e)
            gemini_ok = False

    # Summary
    if gemini_ok and not (js.stage_status(state, "summary") == "done" and _has_result(state, "summary")):
        mark("summary", "active")
        try:
            js.save_result(state, "summary", _clean(summarize(transcript)))
            mark("summary", "done")
        except Exception as e:
            fail("summary", e)
            gemini_ok = False

    # Extraction (action items / decisions / questions are saved one by one)
    extractors = [
        ("action_items", extract_action_items),
        ("key_decisions", extract_key_decisions),
        ("open_questions", extract_questions),
    ]
    extract_done = js.stage_status(state, "extract") == "done" and all(
        _has_result(state, k) for k, _ in extractors
    )
    if gemini_ok and not extract_done:
        mark("extract", "active")
        extract_error = None
        for key, func in extractors:
            if _has_result(state, key):
                continue  # already saved earlier: never call Gemini again for it
            try:
                js.save_result(state, key, _clean(func(transcript)))
            except Exception as e:
                print(traceback.format_exc())
                extract_error = e
                break
        if extract_error is None:
            mark("extract", "done")
        else:
            code, message = js.classify_error(extract_error)
            mark("extract", "failed", message, code)
            gemini_ok = False

    # ── 6. RAG: no Gemini call is made while building, so it is always attempted ──
    rag_chain = None
    rag_was_done = js.stage_status(state, "rag") == "done"
    if not rag_was_done:
        mark("rag", "active")
    try:
        rag_chain = build_rag_chain(transcript, job_id)  # idempotent: no duplicate documents
        if not rag_was_done:
            mark("rag", "done")
    except Exception as e:
        fail("rag", e)

    return finish(transcript, rag_chain)
