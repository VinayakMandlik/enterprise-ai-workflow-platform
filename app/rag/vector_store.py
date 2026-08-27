"""
Qdrant Cloud vector store connection.

Creates the collection on first run if it doesn't exist yet, with the
correct vector dimension for our embedding model (384 for
all-MiniLM-L6-v2 — this MUST match get_embeddings()'s output size,
or every insert will fail).
"""
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import get_settings
from app.providers import get_embeddings

EMBEDDING_DIM = 384  # matches sentence-transformers/all-MiniLM-L6-v2

_singleton_store = None


def get_vector_store():
    global _singleton_store
    if _singleton_store is not None:
        return _singleton_store

    settings = get_settings()
    embeddings = get_embeddings()

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )

    if not client.collection_exists(settings.qdrant_collection):
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

    _singleton_store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
    )
    return _singleton_store