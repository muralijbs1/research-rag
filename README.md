# Research Paper RAG Assistant

A production-oriented RAG (Retrieval-Augmented Generation) app for querying research papers. Built with Python 3.11, LangChain, LangGraph, and ChromaDB.

## Stack

- **Orchestration:** LangChain, LangGraph  
- **Vector store:** ChromaDB (persistent)  
- **Embeddings:** OpenAI  
- **LLMs:** Claude (Anthropic), GPT-4o (OpenAI)  
- **Reranker:** Cohere  
- **Evaluation:** RAGAS  
- **UI:** Streamlit  

## Requirements

- Python 3.11+
- API keys: OpenAI, Anthropic, Cohere (for full features)

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Environment variables

Create a `.env` in the project root (see `.env.example` if added). Example:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
COHERe_API_KEY=...
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=...
```

- `OPENAI_API_KEY` — embeddings and GPT-4o  
- `ANTHROPIC_API_KEY` — Claude  
- `COHERe_API_KEY` — reranker  
- `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` — optional LangSmith tracing  

## Run

```bash
# Streamlit UI
streamlit run app.py

# Run tests
pytest
```

*(Update `app.py` path if your entrypoint is different.)*

## Project structure (intended)

```
My_RAG/
├── src/
│   ├── generation/
│   │   └── llm_router.py    # All LLM calls go through here
│   ├── ingestion/           # Document loaders, chunking, indexing
│   ├── retrieval/           # Retrieval, reranking
│   └── ...
├── chroma_db/               # Persistent ChromaDB (gitignored)
├── uploaded_pdfs/           # Uploaded documents (gitignored)
├── app.py                   # Streamlit entrypoint
├── requirements.txt
└── .env                     # Secrets (gitignored)
```

## Conventions

- All LLM calls go through `src/generation/llm_router.py`.  
- ChromaDB uses `PersistentClient` with a path (e.g. `chroma_db/`), never in-memory.  
- Streamlit state that must survive reruns uses `st.session_state`.  
- Type hints and docstrings on every function.  

## Optional

- **MPS:** On Apple Silicon (e.g. M3), use `device="mps"` for any local PyTorch models.  
- **LangSmith:** Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` to trace runs.  

---

*This README needs to be updated project grows.*
