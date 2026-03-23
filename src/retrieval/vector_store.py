import chromadb
from src.config import CHROMA_PERSIST_DIR

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(name="papers")

    def add(
        self,
        chunks: list[str],
        vectors: list[list[float]],
        paper_name: str,
        paper_hash: str | None = None,
        source: str = "user",
    ) -> None:
        if not chunks:
            raise ValueError("No chunks to add")
        if len(chunks) != len(vectors):
            raise ValueError("Chunks and vectors must be the same length")

        ids = [f"{paper_name}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "paper_name": paper_name,
                "paper_hash": paper_hash or " ",
                "source": source,
            }
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
        """Return distinct paper names currently stored."""
        existing = self.collection.get()
        paper_names = set()
        for _id in existing["ids"]:
            if "_chunk_" in _id:
                prefix = _id.split("_chunk_", 1)[0]
                paper_names.add(prefix)
        return sorted(paper_names)

    def list_papers_with_info(self) -> list[dict]:
        """
        Return one dict per distinct paper with name, chunk count, and source.
        Example: [{"paper_name": "foo", "chunk_count": 42, "source": "user"}]
        """
        existing = self.collection.get(include=["metadatas"])
        papers: dict[str, dict] = {}
        for _id, meta in zip(existing["ids"], existing["metadatas"]):
            if "_chunk_" not in _id:
                continue
            name = meta.get("paper_name") or _id.split("_chunk_", 1)[0]
            if name not in papers:
                papers[name] = {
                    "paper_name": name,
                    "chunk_count": 0,
                    "source": meta.get("source", "user"),
                }
            papers[name]["chunk_count"] += 1
        return sorted(papers.values(), key=lambda p: p["paper_name"])

    def chunk_count_by_name(self, paper_name: str) -> int:
        """Return how many chunks are stored for the given paper name."""
        existing = self.collection.get(
            where={"paper_name": paper_name}, include=["metadatas"]
        )
        return len(existing.get("ids") or [])

    def count_papers(self) -> int:
        return len(self.list_papers())

    def paper_exists_by_hash(self, paper_hash: str) -> bool:
        existing = self.collection.get(where={"paper_hash": paper_hash}, limit=1)
        return bool(existing.get("ids"))

    def get_paper_source_by_hash(self, paper_hash: str) -> str | None:
        """
        Return the source ("system" or "user") of a paper identified by its
        content hash, or None if no paper with that hash exists.
        """
        existing = self.collection.get(
            where={"paper_hash": paper_hash},
            limit=1,
            include=["metadatas"],
        )
        ids = existing.get("ids") or []
        if not ids:
            return None
        return (existing["metadatas"][0] or {}).get("source", "user")

    def delete_by_hash(self, paper_hash: str) -> None:
        existing = self.collection.get(where={"paper_hash": paper_hash})
        ids_to_delete = existing.get("ids") or []
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
