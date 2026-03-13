import chromadb
from src.config import CHROMA_PERSIST_DIR

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(name="papers")

    def add(self, chunks: list[str], vectors: list[list[float]], paper_name: str, paper_hash: str | None = None) -> None:
        if not chunks:
            raise ValueError("No chunks to add")
        if len(chunks) != len(vectors):
            raise ValueError("Chunks and vectors must be the same length")
        
        ids = [f"{paper_name}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {"paper_name": paper_name, "paper_hash": paper_hash}
            for _ in chunks
        ]

        self.collection.add(
            documents=chunks,
            embeddings=vectors,
            ids=ids,
            metadatas=metadatas,
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

    def list_papers(self) -> list[str]:
        """
        Return the distinct paper names currently stored in this collection.

        Paper names are inferred from IDs of the form ``{paper_name}_chunk_{i}``.
        """
        existing = self.collection.get()
        paper_names = set()
        for _id in existing["ids"]:
            if "_chunk_" in _id:
                prefix = _id.split("_chunk_", 1)[0]
                paper_names.add(prefix)
        return sorted(paper_names)

    def count_papers(self) -> int:
        """
        Return how many distinct papers are stored.

        This simply counts the unique prefixes found by ``list_papers()``.
        """
        return len(self.list_papers())

    def paper_exists_by_hash(self, paper_hash: str) -> bool:
        """
        Check whether a paper with the given content hash is already stored.

        This uses the ``paper_hash`` metadata field written in ``add``.
        """
        existing = self.collection.get(where={"paper_hash": paper_hash}, limit=1)
        return bool(existing.get("ids"))

    def delete_by_hash(self, paper_hash: str) -> None:
        """
        Delete all chunks whose metadata ``paper_hash`` matches the given value.

        Useful for "delete and re‑ingest" flows when a user confirms they
        really want to overwrite an existing paper with identical content.
        """
        existing = self.collection.get(where={"paper_hash": paper_hash})
        ids_to_delete = existing.get("ids") or []
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)