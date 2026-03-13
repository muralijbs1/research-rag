from __future__ import annotations

from pathlib import Path
import hashlib

from src.ingestion.chunker import load_and_chunk_pdf
from src.ingestion.embedder import embed_texts
from src.retrieval.vector_store import VectorStore


def _resolve_pdf_path(pdf_path: str | Path) -> Path:
    """
    Resolve a PDF path in a notebook/Streamlit-friendly way.

    Why this exists:
    - Notebooks often run with cwd = ``notebooks/``.
    - Streamlit often runs with cwd = project root.
    - Users will pass relative paths like ``uploaded_pdfs/foo.pdf``.

    Strategy:
    - If the given path exists as-is, use it.
    - Otherwise, search up a few parent directories and see if the relative
      path exists from there.
    """
    path = Path(pdf_path)
    if path.exists():
        return path

    if path.is_absolute():
        raise FileNotFoundError(f"PDF not found at absolute path: {path}")

    cwd = Path.cwd()
    for base in [cwd, *cwd.parents][:5]:
        candidate = base / path
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"PDF not found: {path} (cwd={cwd}). "
        "Pass an absolute path or a path relative to your project root."
    )


def run_ingestion(
    pdf_path: str | Path,
    paper_name: str | None = None,
) -> dict[str, int]:
    """
    End‑to‑end ingestion orchestrator for a single PDF.

    This is the only function the Streamlit upload page needs to call.
    It takes a PDF on disk plus a ``paper_name`` label and then:

    1. Parses + chunks the PDF into text segments.
    2. Creates embeddings for each chunk.
    3. Stores chunks + embeddings in the Chroma vector store
       under IDs that start with ``paper_name``.

    Parameters
    ----------
    pdf_path:
        Path to the uploaded PDF on disk.
    paper_name:
        Optional identifier for this paper (used as an ID prefix in Chroma).
        If omitted, the file name (without ``.pdf``) is used automatically.

    Returns
    -------
    dict[str, int]
        Simple statistics that the UI can display, e.g. number of chunks.
    """
    store = VectorStore()

    path: Path = _resolve_pdf_path(pdf_path)

    # If the caller does not provide a name, derive one from the file name.
    resolved_paper_name: str = paper_name or path.stem

    # Compute a stable hash of the raw PDF bytes so we can detect
    # exact-duplicate uploads even if the file name changes.
    pdf_bytes: bytes = path.read_bytes()
    pdf_hash: str = hashlib.sha256(pdf_bytes).hexdigest()

    # 5) Check whether this content was already ingested.
    if store.paper_exists_by_hash(pdf_hash):
        # Always delete previous chunks for this exact PDF content so that
        # a new ingest acts as a fresh overwrite.
        store.delete_by_hash(pdf_hash)

    # 1) Parse + chunk PDF
    chunks = load_and_chunk_pdf(path)

    # 2) Create embeddings
    embeddings = embed_texts(chunks)

    # 3 & 6) Store in vector store
    store.add(
        chunks=chunks,
        vectors=embeddings,
        paper_name=resolved_paper_name,
        paper_hash=pdf_hash,
    )

    return {"num_chunks": len(chunks)}

