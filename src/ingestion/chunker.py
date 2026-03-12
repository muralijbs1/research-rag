from __future__ import annotations

from pathlib import Path

from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.ingestion.pdf_parser import parse_pdf


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split cleaned text into overlapping character-based chunks.

    This is a simple sliding-window chunker: each chunk has length up to
    ``chunk_size`` characters, and consecutive chunks overlap by
    ``overlap`` characters. The values default to ``CHUNK_SIZE`` and
    ``CHUNK_OVERLAP`` from ``config.py`` so you can tune them in a
    notebook and then update the config centrally.

    Parameters
    ----------
    text:
        Cleaned input text (e.g. from a PDF page or concatenated pages).
    chunk_size:
        Target number of characters per chunk.
    overlap:
        Number of characters two consecutive chunks should share.

    Returns
    -------
    list[str]
        A list of text chunks in order.
    """
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start: int = 0
    text_length: int = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        if end == text_length:
            break
        start = end - overlap

    return chunks



def load_and_chunk_pdf(
    pdf_path: str | Path,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """
    Parse a PDF into cleaned text and split it into overlapping chunks.

    This ties together ``parse_pdf`` and ``chunk_text`` so you can go
    from a file on disk to ready-to-embed chunks in one call.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file on disk.

    chunk_size and overlap:
        Optional overrides for the default character-based chunk size
        and overlap. If left as ``None``, values from ``config.py`` are
        used instead.

    Returns
    -------
    list[str]
        List of text chunks ready for embeddings / indexing.
    """
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if overlap is None:
        overlap = CHUNK_OVERLAP

    pages = parse_pdf(str(pdf_path))
    full_text = " ".join(page["text"] for page in pages)
    return chunk_text(full_text, chunk_size, overlap)
