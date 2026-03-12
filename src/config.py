from dotenv import load_dotenv
import os

load_dotenv()

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# Retrieval
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", 20))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", 5))

# Models
DEFAULT_LLM = os.getenv("DEFAULT_LLM", "gpt4o")
RERANKER = os.getenv("RERANKER", "cohere")

# Paths
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")