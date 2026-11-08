"""ChromaDB helpers for legacy single-collection and isolated per-job RAG.

The original public helpers remain available for compatibility. The resumable
pipeline uses ``build_job_vector_store`` so different videos never share a
retrieval collection.
"""

from __future__ import annotations

import hashlib
import os

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CHROMA_DIR = os.path.join(PROJECT_ROOT, "vector_db")
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def _split_transcript(transcript: str) -> list[str]:
    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError("Cannot build a vector store from an empty transcript.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(transcript)
    if not chunks:
        raise ValueError("Transcript did not produce any text chunks.")
    return chunks


def build_vector_store(transcript: str, job_id: str = None) -> Chroma:
    """Build a legacy shared index, or an isolated per-job index when job_id is set.

    The optional job_id parameter preserves the original public function signature.
    New application code should always provide job_id for video-specific RAG.
    """
    if job_id is not None:
        return build_job_vector_store(transcript, job_id)

    chunks = _split_transcript(transcript)
    documents = [
        Document(page_content=chunk, metadata={"chunk_index": index})
        for index, chunk in enumerate(chunks)
    ]
    return Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )


def load_vector_store() -> Chroma:
    """Load the original shared collection (legacy compatibility only)."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def get_retriever(vector_store: Chroma, k: int = 4, job_id: str = None):
    """Create a retriever; an optional job_id keeps legacy callers compatible.

    For strict isolation prefer the per-job collection and always pass job_id.
    """
    search_kwargs = {"k": k}
    if job_id is not None:
        search_kwargs["filter"] = {"job_id": job_id}
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )


def _job_collection_name(job_id: str) -> str:
    """Return a Chroma-safe, deterministic collection name for one job."""
    digest = hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:32]
    # Chroma collection names must be short and must begin/end with alphanumerics.
    return f"job_{digest}"


def _transcript_fingerprint(transcript: str) -> str:
    return hashlib.sha256(transcript.encode("utf-8")).hexdigest()


def build_job_vector_store(transcript: str, job_id: str) -> Chroma:
    """Build/reuse one isolated collection for a job without duplicate chunks.

    The collection is unique to ``job_id``. Chunk IDs are deterministic. If a
    transcript changes for an existing job, only that job's collection is
    replaced; all other jobs' collections are untouched. A partial build is
    detected and repaired on the next call.
    """
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("A non-empty job_id is required for isolated RAG storage.")

    chunks = _split_transcript(transcript)
    fingerprint = _transcript_fingerprint(transcript)
    collection_name = _job_collection_name(job_id)

    store = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )

    # Public Chroma API only: no direct use of private _collection internals.
    existing = store.get(include=["metadatas"])
    existing_ids = list(existing.get("ids") or [])
    existing_metas = list(existing.get("metadatas") or [])
    expected_ids = [f"{fingerprint[:24]}_{index}" for index in range(len(chunks))]

    existing_is_valid = (
        len(existing_ids) == len(expected_ids)
        and set(existing_ids) == set(expected_ids)
        and len(existing_metas) == len(existing_ids)
        and all(
            (metadata or {}).get("job_id") == job_id
            and (metadata or {}).get("transcript_fingerprint") == fingerprint
            for metadata in existing_metas
        )
    )
    if existing_is_valid:
        print(f"RAG: reusing {len(existing_ids)} indexed chunks for job {job_id}.")
        return store

    # The collection belongs exclusively to this job, so deleting its old
    # contents cannot affect another video's index.
    if existing_ids:
        store.delete(ids=existing_ids)

    documents = [
        Document(
            page_content=chunk,
            metadata={
                "job_id": job_id,
                "transcript_fingerprint": fingerprint,
                "chunk_index": index,
            },
        )
        for index, chunk in enumerate(chunks)
    ]
    store.add_documents(documents=documents, ids=expected_ids)

    # Verify the write before allowing a RAG chain to use this collection.
    saved = store.get(ids=expected_ids, include=["metadatas"])
    saved_ids = list(saved.get("ids") or [])
    saved_metas = list(saved.get("metadatas") or [])
    if set(saved_ids) != set(expected_ids) or len(saved_metas) != len(expected_ids):
        raise RuntimeError("ChromaDB job index verification failed; RAG is unavailable.")
    if not all(
        (metadata or {}).get("job_id") == job_id
        and (metadata or {}).get("transcript_fingerprint") == fingerprint
        for metadata in saved_metas
    ):
        raise RuntimeError("ChromaDB returned mismatched job metadata; refusing unfiltered retrieval.")

    print(f"RAG: indexed {len(expected_ids)} chunks for job {job_id}.")
    return store



def load_job_vector_store(job_id: str) -> Chroma:
    """Load and validate an existing isolated job collection without re-indexing."""
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("A non-empty job_id is required for isolated RAG storage.")

    store = Chroma(
        collection_name=_job_collection_name(job_id),
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )
    data = store.get(include=["metadatas"])
    ids = list(data.get("ids") or [])
    metadatas = list(data.get("metadatas") or [])
    if not ids or len(ids) != len(metadatas):
        raise RuntimeError(f"No valid ChromaDB index exists for job {job_id}.")
    if not all((metadata or {}).get("job_id") == job_id for metadata in metadatas):
        raise RuntimeError(f"ChromaDB index validation failed for job {job_id}.")
    return store
