from __future__ import annotations

from pathlib import Path
import hashlib

from src.ingestion.chunker import load_and_chunk_pdf
from src.ingestion.embedder import embed_texts
from src.retrieval.vector_store import VectorStore


class SystemPaperError(Exception):
    """Raised when a user tries to ingest a PDF that matches a system paper."""


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
    source: str = "user",
) -> dict[str, int]:
    """
    End-to-end ingestion orchestrator for a single PDF.

    Parameters
    ----------
    pdf_path:
        Path to the uploaded PDF on disk.
    paper_name:
        Optional identifier for this paper. Defaults to the file stem.
    source:
        "user" for user-uploaded papers, "system" for pre-ingested library
        papers. Defaults to "user".

    Returns
    -------
    dict with "num_chunks".

    Raises
    ------
    SystemPaperError
        If the PDF content hash matches a system paper already in the store.
        User-uploaded duplicates are silently overwritten as before.
    """
    store = VectorStore()

    path: Path = _resolve_pdf_path(pdf_path)
    resolved_paper_name: str = paper_name or path.stem

    pdf_bytes: bytes = path.read_bytes()
    pdf_hash: str = hashlib.sha256(pdf_bytes).hexdigest()

    existing_source = store.get_paper_source_by_hash(pdf_hash)

    if existing_source == "system":
        raise SystemPaperError(
            "This paper is already in our library. Ask your questions directly."
        )

    if existing_source == "user":
        # Overwrite previous user upload with the same content.
        store.delete_by_hash(pdf_hash)

    chunks = load_and_chunk_pdf(path)
    embeddings = embed_texts(chunks)

    store.add(
        chunks=chunks,
        vectors=embeddings,
        paper_name=resolved_paper_name,
        paper_hash=pdf_hash,
        source=source,
    )

    return {"num_chunks": len(chunks)}
