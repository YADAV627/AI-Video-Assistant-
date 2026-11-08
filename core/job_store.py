"""
Persistent job storage for the AI Video Assistant.

Every video + language combination is one "job" stored in its own folder:

    processed_data/<job_id>/state.json     stage status, errors, generated text
    processed_data/<job_id>/transcript.txt the full transcript

Only JSON and TXT are used (no pickle) and no API keys are ever stored.
All writes are atomic (temp file + os.replace) so a crash can never leave a
half-written state file behind.
"""

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from urllib.parse import urlparse, parse_qs

# processed_data/ lives next to app.py, whatever the current working directory is.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed_data")

STAGES = ["audio", "transcript", "title", "summary", "extract", "rag"]
RESULT_KEYS = ["title", "summary", "action_items", "key_decisions", "open_questions"]

_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ─── Job ID ─────────────────────────────────────────────────────────────────────
def youtube_video_id(url: str):
    """Return the 11-character YouTube video ID, or None if the URL is not YouTube."""
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]

    video_id = None
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    elif host == "youtube.com" or host.endswith(".youtube.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
        else:
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v"):
                video_id = parts[1]

    if video_id and _VIDEO_ID.match(video_id):
        return video_id
    return None


def make_job_id(source: str, language: str) -> str:
    """
    Stable job ID = <source identifier>_<language>.
      YouTube URL -> yt_<video id>          (tracking params like &t= ?si= are ignored)
      other URL   -> url_<hash of the url>
      local file  -> file_<hash of absolute path>
    """
    source = source.strip()
    lang = language.strip().lower()
    if source.startswith(("http://", "https://")):
        video_id = youtube_video_id(source)
        ident = "yt_" + video_id if video_id else "url_" + _sha1(source)[:12]
    else:
        path = os.path.normcase(os.path.abspath(source))
        ident = "file_" + _sha1(path)[:12]
    return f"{ident}_{lang}"


# ─── Paths and atomic writes ────────────────────────────────────────────────────
def _job_dir(job_id: str) -> str:
    return os.path.join(PROCESSED_DIR, job_id)


def _state_path(job_id: str) -> str:
    return os.path.join(_job_dir(job_id), "state.json")


def _transcript_path(job_id: str) -> str:
    return os.path.join(_job_dir(job_id), "transcript.txt")


def _atomic_write(path: str, text: str) -> None:
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=folder, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# ─── State ──────────────────────────────────────────────────────────────────────
def _new_state(job_id: str, source: str, language: str) -> dict:
    return {
        "job_id": job_id,
        "source": source.strip(),
        "language": language.strip().lower(),
        "created_at": _now(),
        "updated_at": _now(),
        "stages": {s: {"status": "pending", "error": None, "error_code": None} for s in STAGES},
        "results": {k: None for k in RESULT_KEYS},
        "audio_chunks": [],
    }


def save_state(state: dict) -> None:
    state["updated_at"] = _now()
    _atomic_write(_state_path(state["job_id"]), json.dumps(state, ensure_ascii=False, indent=2))


def load_job(job_id: str):
    """Load a saved job. Returns None if it does not exist or its file is corrupted."""
    path = _state_path(job_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if not isinstance(state, dict) or "stages" not in state:
            raise ValueError("invalid state file")
    except (ValueError, OSError):
        # keep the bad file for inspection, never crash the app
        try:
            os.replace(path, path + ".corrupt")
        except OSError:
            pass
        return None

    changed = False
    # forward compatibility: make sure every expected key exists
    for s in STAGES:
        if s not in state["stages"]:
            state["stages"][s] = {"status": "pending", "error": None, "error_code": None}
            changed = True
    state.setdefault("results", {})
    for k in RESULT_KEYS:
        state["results"].setdefault(k, None)
    state.setdefault("audio_chunks", [])

    # A stage left "active" means the app was stopped while it was running.
    for info in state["stages"].values():
        if info.get("status") == "active":
            info["status"] = "failed"
            info["error"] = "Interrupted: the app stopped while this stage was running."
            info["error_code"] = "interrupted"
            changed = True
    if changed:
        save_state(state)
    return state


def create_or_load_job(source: str, language: str) -> dict:
    job_id = make_job_id(source, language)
    state = load_job(job_id)
    if state is None:
        state = _new_state(job_id, source, language)
        save_state(state)
    return state


def set_stage(state: dict, stage: str, status: str, error=None, code=None) -> None:
    state["stages"][stage]["status"] = status
    state["stages"][stage]["error"] = error
    state["stages"][stage]["error_code"] = code
    save_state(state)


def stage_status(state: dict, stage: str) -> str:
    return state["stages"][stage]["status"]


def save_result(state: dict, key: str, value) -> None:
    state["results"][key] = value
    save_state(state)


def get_result(state: dict, key: str):
    return state["results"].get(key)


def save_audio_chunks(state: dict, chunks: list) -> None:
    state["audio_chunks"] = [str(c) for c in chunks]
    save_state(state)


def save_transcript(job_id: str, text: str) -> None:
    _atomic_write(_transcript_path(job_id), text)


def load_transcript(job_id: str):
    path = _transcript_path(job_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text if text.strip() else None


def is_complete(state: dict) -> bool:
    return all(state["stages"][s]["status"] == "done" for s in STAGES)


def list_jobs() -> list:
    """All saved jobs, most recently updated first."""
    jobs = []
    if not os.path.isdir(PROCESSED_DIR):
        return jobs
    for name in os.listdir(PROCESSED_DIR):
        if os.path.isdir(_job_dir(name)):
            state = load_job(name)
            if state:
                jobs.append(state)
    jobs.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
    return jobs


# ─── Error classification (Gemini / network) ────────────────────────────────────
def classify_error(exc: Exception):
    """Return (code, friendly message). Never raises."""
    text = (type(exc).__name__ + " " + str(exc)).upper()
    if re.search(r"\b429\b", text) or "RESOURCE_EXHAUSTED" in text or "RESOURCEEXHAUSTED" in text:
        return "rate_limit", "Gemini quota or rate limit reached (429 RESOURCE_EXHAUSTED). Wait a little, then press Resume."
    if re.search(r"\b503\b", text) or "UNAVAILABLE" in text:
        return "unavailable", "Gemini is temporarily unavailable (503). Press Resume in a moment."
    if (
        isinstance(exc, TimeoutError)
        or "TIMEOUT" in text
        or "TIMED OUT" in text
        or "DEADLINE" in text
    ):
        return "timeout", "The request timed out. Check your connection, then press Resume."
    if "CONNECTION" in text or "NETWORK" in text or "GETADDRINFO" in text or "NAME RESOLUTION" in text:
        return "network", "Network error while contacting the service. Check your connection, then press Resume."
    return "error", str(exc)[:300] or type(exc).__name__
