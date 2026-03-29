# Research Paper RAG Assistant

A full-stack Retrieval-Augmented Generation (RAG) system for ML research papers. Upload PDFs, ask questions, compare LLMs side-by-side, and evaluate pipeline quality — all from a Streamlit UI backed by a LangGraph agentic pipeline.

---

## Features

- **PDF Ingestion** — parse, chunk, embed, and store papers in ChromaDB with SHA-256 deduplication
- **Agentic RAG Pipeline** — LangGraph graph with intent checking, query rewriting, and quality-gated retrieval
- **Multi-Query Retrieval** — Groq generates 2 query variants per question for broader chunk coverage
- **Reranking** — SBERT cross-encoder (local, MPS-accelerated) or Cohere Rerank API
- **Multi-LLM Support** — GPT-4o-mini, Claude Haiku, and Groq (Llama 3.3 70B) via LiteLLM
- **Conversational Chat** — Groq-powered gateway that routes research questions to RAG and handles small talk naturally
- **Model Comparison** — side-by-side GPT-4o-mini vs Claude Haiku with latency and token counts
- **Evaluation** — RAGAS metrics over a 23-question benchmark across 11 papers, logged to MLflow
- **Paper Library** — browse all ingested papers and chunk counts

---

## How It Works

```
PDF → extract text → chunk → embed → store in ChromaDB

User message
    │
    ▼
Groq gateway        ← is this AI/ML research or small talk?
    ├─ chat          → friendly reply
    └─ rag
        ▼
Multi-query retrieval  ← original + 2 Groq-generated variants → merge + dedup
        ▼
SBERT reranker         ← cross-encoder scores, keep top-5
        ▼
Quality check          ← avg score > 0? if not → rewrite query (max 2 retries)
        ▼
GPT-4o-mini / Claude   ← grounded answer from context chunks
```

---

## Setup

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd My_RAG

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .

# 4. Configure environment variables
cp .env.example .env   # fill in your keys
```

### Required API Keys (`.env`)

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GROQ_API_KEY=...
COHERE_API_KEY=...    # optional — only if RERANKER=cohere
```

---

## Running the App

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501`.

---

## Pages

| Page | Description |
|---|---|
| **Upload** | Upload a PDF and ingest it into ChromaDB |
| **Chat** | Conversational RAG with history and auto-generated titles |
| **Compare** | Side-by-side GPT-4o-mini vs Claude Haiku on the same question |
| **Evaluation** | Run the eval suite and view RAGAS scores |
| **Library** | Browse all ingested papers |

---

## Project Structure

```
src/
├── config.py                    # Env-driven config (chunk sizes, model names, paths)
├── ingestion/
│   ├── pdf_parser.py            # PyMuPDF: PDF → pages
│   ├── chunker.py               # Sliding-window chunker (750 chars, 75 overlap)
│   ├── embedder.py              # OpenAI text-embedding-3-small (batched)
│   └── ingest_pipeline.py       # Orchestrator with SHA-256 dedup
├── retrieval/
│   ├── vector_store.py          # ChromaDB PersistentClient wrapper
│   ├── retriever.py             # Embed → top-k ChromaDB search
│   ├── reranker.py              # SBERT cross-encoder or Cohere Rerank
│   └── multi_query_retriever.py # Groq query expansion + dedup
├── generation/
│   ├── llm_router.py            # Central LLM router (LiteLLM). All calls go here.
│   ├── prompts_writer.py        # All prompts in one place
│   ├── prompt_builder.py        # Assembles final prompt string
│   └── generator.py             # High-level RAG generation function
├── graph/
│   ├── nodes.py                 # LangGraph nodes + RAGState
│   ├── rag_graph.py             # StateGraph wiring
│   └── intent_response.py       # Groq chat gateway + title generation
└── evaluation/
    ├── eval_dataset.py          # Loads eval_data/test_questions.json
    ├── evaluator.py             # RAGAS scoring
    └── experiment_log.py        # MLflow logging helpers

pages/                           # Streamlit multi-page app
├── 01_Upload.py
├── 02_Chat.py
├── 03_Compare.py
├── 04_Evaluation.py
└── 05_Library.py
```

---

## Programmatic Usage

```python
# Ingest a paper
from src.ingestion.ingest_pipeline import run_ingestion
run_ingestion("uploaded_pdfs/Attention_is_all_you_need.pdf")

# LangGraph pipeline
from src.graph.rag_graph import rag_pipeline
result = rag_pipeline.invoke({
    "question": "What is multi-head attention?",
    "original_question": "What is multi-head attention?",
    "retries": 0,
})
print(result["answer"]["answer"])

# Individual steps
from src.retrieval.retriever import retrieve
from src.retrieval.reranker import rerank
from src.generation.generator import generate_answer

chunks = retrieve("What is multi-head attention?", top_k=20)
reranked = rerank("What is multi-head attention?", chunks, top_n=5)
result = generate_answer("What is multi-head attention?", reranked)
print(result["answer"])
```

---

## Testing

```bash
pytest
pytest --cov=src
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| PDF parsing | PyMuPDF |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | ChromaDB (persistent) |
| Reranking | SBERT `cross-encoder/ms-marco-MiniLM-L-6-v2` / Cohere Rerank |
| LLM routing | LiteLLM |
| LLMs | GPT-4o-mini, Claude Haiku, Groq Llama 3.3 70B |
| Agentic pipeline | LangGraph |
| UI | Streamlit |
| Evaluation | RAGAS + MLflow |

---

## Configuration

All settings in `.env`, read by `src/config.py`:

| Variable | Default | Purpose |
|---|---|---|
| `CHUNK_SIZE` | 750 | Characters per chunk |
| `CHUNK_OVERLAP` | 75 | Overlap between chunks |
| `TOP_K_RETRIEVAL` | 20 | Chunks fetched from ChromaDB |
| `RERANK_TOP_N` | 5 | Chunks kept after reranking |
| `RERANKER` | `sbert` | `sbert` or `cohere` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `DEFAULT_LITELLM` | `openai/gpt-4o-mini` | Default generation model |
| `GROQ_LITELLM_MODEL` | `groq/llama-3.3-70b-versatile` | Groq model |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage path |

---

*Built as a portfolio project to learn production RAG engineering from the ground up.*
