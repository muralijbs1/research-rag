"""
reingest_all.py

Ingest papers into ChromaDB or Pinecone.

Usage
-----
# Wipe and re-ingest all system papers (default):
    python reingest_all.py

# Ingest new papers without wiping existing data:
    python reingest_all.py --folder uploaded_pdfs/new_papers --no-wipe

# Ingest any folder without wiping:
    python reingest_all.py --folder /path/to/papers --no-wipe
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from src.ingestion.ingest_pipeline import run_ingestion
from src.retrieval.vector_store import VectorStore

DEFAULT_PAPERS_DIR = Path("/Users/muralikrishnagurijala/Desktop/My_RAG/uploaded_pdfs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest papers into vector store.")
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Folder containing PDFs to ingest. Defaults to uploaded_pdfs/.",
    )
    parser.add_argument(
        "--no-wipe",
        action="store_true",
        help="Skip wiping the collection before ingesting.",
    )
    args = parser.parse_args()

    papers_dir = Path(args.folder) if args.folder else DEFAULT_PAPERS_DIR
    if not papers_dir.is_absolute():
        papers_dir = DEFAULT_PAPERS_DIR.parent / papers_dir

    if not papers_dir.exists():
        print(f"Folder not found: {papers_dir}")
        return

    store = VectorStore()

    # --- Wipe unless --no-wipe is set ---
    if not args.no_wipe:
        print("Resetting collection...")
        store.reset_collection()
        print("Collection wiped.\n")
    else:
        print("Skipping wipe — adding to existing data.\n")

    # --- Find all PDFs ---
    pdfs = sorted(papers_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {papers_dir}")
        return

    print(f"Found {len(pdfs)} papers in {papers_dir}. Starting ingestion...\n")

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

    print(f"\n=== Ingestion Complete ===")
    print(f"  Papers ingested : {len(pdfs) - len(failed)}/{len(pdfs)}")
    print(f"  Total chunks    : {total_chunks}")
    if failed:
        print(f"  Failed          : {failed}")


if __name__ == "__main__":
    main()