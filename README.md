# Research Paper RAG Assistant

Ask questions about research papers in plain English and get cited, grounded answers powered by AI.

> **Status:** Phase 1 complete — ingestion pipeline fully working. Phase 2 in progress.

---

## What It Does

Upload a research paper PDF. Ask a question. The system finds the most relevant sections of the paper, passes them to an AI model, and returns a cited answer — along with a score showing how grounded and reliable that answer is.

---

## How It Works

```
PDF → extract text → chunk → embed → store in ChromaDB
Question → embed → search ChromaDB → rerank → generate answer → RAGAS score
```

1. **Parse** — PyMuPDF extracts clean text from each page
2. **Chunk** — text is split into overlapping 500-character segments
3. **Embed** — each chunk is converted to a 1536-dimension vector via OpenAI `text-embedding-3-small`
4. **Store** — vectors and chunks saved to ChromaDB locally on disk
5. **Retrieve** — user question is embedded and top-20 similar chunks are fetched
6. **Rerank** — Cohere Rerank selects the top-5 most relevant chunks *(Phase 2)*
7. **Generate** — Claude Sonnet or GPT-4o answers using only the retrieved chunks *(Phase 2)*
8. **Evaluate** — RAGAS scores the answer for faithfulness and relevancy *(Phase 2)*

---

## Tech Stack

| Layer | Tool |
|---|---|
| PDF Parsing | PyMuPDF |
| Chunking | Custom sliding-window chunker |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | ChromaDB (local persistent) |
| Reranking | Cohere Rerank API + SBERT (Phase 2) |
| LLMs | Claude Sonnet + GPT-4o via unified router (Phase 2) |
| Evaluation | RAGAS — faithfulness, answer relevancy, context recall, context precision (Phase 2) |
| Experiment Tracking | MLflow (Phase 4) |
| Frontend | Streamlit multi-page app (Phase 3) |
| Deployment | Streamlit Community Cloud (Phase 5) |

---

## Project Structure

```
research-rag/
  src/
    config.py                   # all settings in one place
    ingestion/
      pdf_parser.py             # extract text from PDF pages
      chunker.py                # split text into overlapping chunks
      embedder.py               # convert chunks to vectors via OpenAI
      ingest_pipeline.py        # orchestrates full ingestion in one call
    retrieval/
      vector_store.py           # ChromaDB wrapper — save, search, delete
      retriever.py              # embed question, fetch top-k chunks (Phase 2)
      reranker.py               # Cohere / SBERT reranker (Phase 2)
    generation/
      prompt_builder.py         # assemble LLM prompt (Phase 2)
      llm_router.py             # single entry point for all LLM calls (Phase 2)
      generator.py              # call router, return answer + sources (Phase 2)
    evaluation/
      evaluator.py              # run RAGAS on a test set (Phase 2)
  notebooks/
    scratch_pdf_exploration.ipynb
    report_chunking_experiments.ipynb
    scratch_embedding_exploration.ipynb
    scratch_retrieval_check.ipynb
  app.py                        # Streamlit entry point (Phase 3)
  pages/                        # Streamlit pages (Phase 3)
```

---

## Phase 1 — What Is Built

- [x] PDF text extraction with PyMuPDF
- [x] Overlapping character-based chunker with configurable size and overlap
- [x] OpenAI embedding with batching (100 chunks per API call)
- [x] ChromaDB persistent vector store with full CRUD
- [x] Content-hash deduplication — re-ingesting the same paper overwrites cleanly
- [x] End-to-end ingestion pipeline tested on *Attention Is All You Need*
- [x] 4 notebooks covering exploration, chunking experiments, embedding verification, retrieval check

---

## Getting Started

### Prerequisites

- Python 3.11+
- OpenAI API key
- Anthropic API key *(Phase 2)*
- Cohere API key *(Phase 2)*

### Setup

```bash
# clone the repo
git clone https://github.com/muralijbs1/research-rag.git
cd research-rag

# create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# install project as a package (needed for notebook imports)
pip install -e .

# create your .env file
cp .env.example .env
# then open .env and add your API keys
```

### Ingest a Paper

```python
from src.ingestion.ingest_pipeline import run_ingestion

result = run_ingestion("uploaded_pdfs/your_paper.pdf")
print(result)  # {'num_chunks': 88}
```

### Search

```python
from src.retrieval.vector_store import VectorStore
from src.ingestion.embedder import embed_texts

vs = VectorStore()
query_vector = embed_texts(["What is the attention mechanism?"])[0]
results = vs.search(query_vector, top_k=5)

for r in results:
    print(r["text"][:200])
```

---

## Roadmap

- [x] Phase 1 — Ingestion pipeline
- [ ] Phase 2 — Reranker, multi-LLM router, RAGAS baseline
- [ ] Phase 3 — Streamlit multi-page app
- [ ] Phase 4 — LangGraph + 10 experiments + MLflow
- [ ] Phase 5 — Deploy to Streamlit Cloud

---

## Configuration

All settings live in `.env` and are read by `src/config.py`:

```
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
COHERE_API_KEY=your-key
CHROMA_PERSIST_DIR=./chroma_db
DEFAULT_LLM=gpt4o
EMBEDDING_MODEL=text-embedding-3-small
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RETRIEVAL=20
RERANK_TOP_N=5
```

---

*Built as a portfolio project to learn production RAG engineering from the ground up.*