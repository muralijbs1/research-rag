"""
reingest_all.py

Wipes ChromaDB and re-ingests all system papers from uploaded_pdfs/.

Usage
-----
    source .venv/bin/activate
    python reingest_all.py

Run this before every chunk size experiment after updating CHUNK_SIZE in .env.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from src.ingestion.ingest_pipeline import run_ingestion
from src.retrieval.vector_store import VectorStore

PAPERS_DIR = Path("/Users/muralikrishnagurijala/Desktop/My_RAG/uploaded_pdfs")


def main() -> None:
    # --- 1. Wipe ChromaDB --------------------------------------------------
    print("Resetting ChromaDB collection...")
    store = VectorStore()
    store.reset_collection()
    print("Collection wiped.\n")

    # --- 2. Find all PDFs --------------------------------------------------
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {PAPERS_DIR}")
        return

    print(f"Found {len(pdfs)} papers. Starting ingestion...\n")

    # --- 3. Ingest each paper ----------------------------------------------
    total_chunks = 0
    failed = []

    for i, pdf_path in enumerate(pdfs, start=1):
        paper_name = pdf_path.stem
        try:
            result = run_ingestion(pdf_path, paper_name=paper_name, source="system")
            chunks = result["num_chunks"]
            total_chunks += chunks
            print(f"  [{i:02d}/{len(pdfs)}] {paper_name} — {chunks} chunks")
        except Exception as e:
            failed.append(paper_name)
            print(f"  [{i:02d}/{len(pdfs)}] FAILED: {paper_name} — {e}")

    # --- 4. Summary --------------------------------------------------------
    print(f"\n=== Ingestion Complete ===")
    print(f"  Papers ingested : {len(pdfs) - len(failed)}/{len(pdfs)}")
    print(f"  Total chunks    : {total_chunks}")
    if failed:
        print(f"  Failed          : {failed}")


if __name__ == "__main__":
    main()