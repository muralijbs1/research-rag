import chromadb
from src.config import CHROMA_PERSIST_DIR

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(name="papers")

    def add(self, chunks: list[str], vectors: list[list[float]], paper_name: str) -> None:
        if not chunks:
            raise ValueError("No chunks to add")
        if len(chunks) != len(vectors):
            raise ValueError("Chunks and vectors must be the same length")
        
        ids = [f"{paper_name}_chunk_{i}" for i in range(len(chunks))]
        
        self.collection.add(
            documents=chunks,
            embeddings=vectors,
            ids=ids
        )

    def search(self, query_vector: list[float], top_k: int = 20) -> list[dict]:
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        return [
            {"text": doc, "score": score}
            for doc, score in zip(
                results["documents"][0],
                results["distances"][0]
            )
        ]

    def delete(self, paper_name: str) -> None:
        existing = self.collection.get()
        ids_to_delete = [
            id for id in existing["ids"]
            if id.startswith(paper_name)
        ]
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)