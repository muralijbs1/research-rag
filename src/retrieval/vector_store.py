import os
import chromadb
from src.config import CHROMA_PERSIST_DIR


class VectorStore:
    def __init__(self):
        pinecone_key = os.getenv("PINECONE_API_KEY")

        if pinecone_key:
            from pinecone import Pinecone
            pc = Pinecone(api_key=pinecone_key)
            self._index = pc.Index(host=os.getenv("PINECONE_HOST"))
            self._mode = "pinecone"
        else:
            self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            self.collection = self.client.get_or_create_collection(name="papers")
            self._mode = "chromadb"

    # ------------------------------------------------------------------
    # ADD
    # ------------------------------------------------------------------
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

        if self._mode == "pinecone":
            records = [
                {
                    "id": id_,
                    "values": vector,
                    "metadata": {
                        "text": chunk,
                        "paper_name": paper_name,
                        "paper_hash": paper_hash or "",
                        "source": source,
                    }
                }
                for id_, chunk, vector in zip(ids, chunks, vectors)
            ]
            batch_size = 100
            for i in range(0, len(records), batch_size):
                self._index.upsert(vectors=records[i:i + batch_size])
        else:
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

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------
    def search(self, query_vector: list[float], top_k: int = 20) -> list[dict]:
        if self._mode == "pinecone":
            results = self._index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
            )
            return [
                {"text": match["metadata"]["text"], "score": match["score"]}
                for match in results["matches"]
            ]
        else:
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

    # ------------------------------------------------------------------
    # DELETE by paper name
    # ------------------------------------------------------------------
    def delete(self, paper_name: str) -> None:
        if self._mode == "pinecone":
            results = self._index.query(
                vector=[0.0] * 1536,
                top_k=10000,
                include_metadata=True,
                filter={"paper_name": {"$eq": paper_name}}
            )
            ids = [m["id"] for m in results["matches"]]
            batch_size = 1000
            for i in range(0, len(ids), batch_size):
                self._index.delete(ids=ids[i:i + batch_size])
        else:
            existing = self.collection.get()
            ids_to_delete = [
                id for id in existing["ids"]
                if id.startswith(paper_name)
            ]
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)

    # ------------------------------------------------------------------
    # LIST PAPERS
    # ------------------------------------------------------------------
    def list_papers(self) -> list[str]:
        if self._mode == "pinecone":
            papers = self.list_papers_with_info()
            return [p["paper_name"] for p in papers]
        else:
            existing = self.collection.get()
            paper_names = set()
            for _id in existing["ids"]:
                if "_chunk_" in _id:
                    prefix = _id.split("_chunk_", 1)[0]
                    paper_names.add(prefix)
            return sorted(paper_names)

    def list_papers_with_info(self) -> list[dict]:
        if self._mode == "pinecone":
            results = self._index.query(
                vector=[0.0] * 1536,
                top_k=10000,
                include_metadata=True,
            )
            papers: dict[str, dict] = {}
            for match in results["matches"]:
                meta = match.get("metadata", {})
                name = meta.get("paper_name", "")
                if not name:
                    continue
                if name not in papers:
                    papers[name] = {
                        "paper_name": name,
                        "chunk_count": 0,
                        "source": meta.get("source", "user"),
                    }
                papers[name]["chunk_count"] += 1
            return sorted(papers.values(), key=lambda p: p["paper_name"])
        else:
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

    # ------------------------------------------------------------------
    # CHUNK COUNT
    # ------------------------------------------------------------------
    def chunk_count_by_name(self, paper_name: str) -> int:
        if self._mode == "pinecone":
            results = self._index.query(
                vector=[0.0] * 1536,
                top_k=10000,
                include_metadata=True,
                filter={"paper_name": {"$eq": paper_name}}
            )
            return len(results["matches"])
        else:
            existing = self.collection.get(
                where={"paper_name": paper_name}, include=["metadatas"]
            )
            return len(existing.get("ids") or [])

    def count_papers(self) -> int:
        return len(self.list_papers())

    # ------------------------------------------------------------------
    # HASH OPERATIONS
    # ------------------------------------------------------------------
    def paper_exists_by_hash(self, paper_hash: str) -> bool:
        if self._mode == "pinecone":
            results = self._index.query(
                vector=[0.0] * 1536,
                top_k=1,
                include_metadata=True,
                filter={"paper_hash": {"$eq": paper_hash}}
            )
            return len(results["matches"]) > 0
        else:
            existing = self.collection.get(where={"paper_hash": paper_hash}, limit=1)
            return bool(existing.get("ids"))

    def get_paper_source_by_hash(self, paper_hash: str) -> str | None:
        if self._mode == "pinecone":
            results = self._index.query(
                vector=[0.0] * 1536,
                top_k=1,
                include_metadata=True,
                filter={"paper_hash": {"$eq": paper_hash}}
            )
            if not results["matches"]:
                return None
            return results["matches"][0]["metadata"].get("source", "user")
        else:
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
        if self._mode == "pinecone":
            results = self._index.query(
                vector=[0.0] * 1536,
                top_k=10000,
                include_metadata=True,
                filter={"paper_hash": {"$eq": paper_hash}}
            )
            ids = [m["id"] for m in results["matches"]]
            batch_size = 1000
            for i in range(0, len(ids), batch_size):
                self._index.delete(ids=ids[i:i + batch_size])
        else:
            existing = self.collection.get(where={"paper_hash": paper_hash})
            ids_to_delete = existing.get("ids") or []
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------
    def reset_collection(self) -> None:
        if self._mode == "pinecone":
            results = self._index.query(
                vector=[0.0] * 1536,
                top_k=10000,
                include_metadata=False,
            )
            ids = [m["id"] for m in results["matches"]]
            batch_size = 1000
            for i in range(0, len(ids), batch_size):
                self._index.delete(ids=ids[i:i + batch_size])
        else:
            self.client.delete_collection(name="papers")
            self.collection = self.client.get_or_create_collection(name="papers")