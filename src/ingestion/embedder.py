from openai import OpenAI
from src.config import EMBEDDING_MODEL


def embed_texts(
    texts: list[str],
    batch_size: int = 100,
    model: str | None = None,
) -> list[list[float]]:
    if not texts:
        return []

    model_used: str = model or EMBEDDING_MODEL
    client = OpenAI()  

    embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(
            model=model_used,     
            input=batch,             
        )
        embeddings.extend([row.embedding for row in response.data])

    return embeddings