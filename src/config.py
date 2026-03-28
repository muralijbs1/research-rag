from dotenv import load_dotenv
import os

load_dotenv(override=True)

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 750))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 75))

# Retrieval
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", 20))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", 5))

# Models
DEFAULT_LLM = os.getenv("DEFAULT_LLM", "gpt-4o-mini")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
#RERANKER = os.getenv("RERANKER", "cohere")  # switch back to cohere when needed
RERANKER = os.getenv("RERANKER", "sbert")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Models for Litellm
DEFAULT_LITELLM = os.getenv("DEFAULT_LITELLM", "openai/gpt-4o-mini")
OPENAI_LITELLM_MODEL = os.getenv("OPENAI_LITELLM_MODEL", "openai/gpt-4o-mini")
ANTHROPIC_LITELLM_MODEL = os.getenv("ANTHROPIC_LITELLM_MODEL", "anthropic/claude-haiku-4-5-20251001")


# Paths
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")