"""
Qdrant Cloud vector store connection.

Creates the collection on first run if it doesn't exist yet, with the
correct vector dimension for our embedding model (384 for
bge-small-en-v1.5 via fastembed — this MUST match get_embeddings()'s
output size, or every insert will fail).
"""
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

from app.config import get_settings
from app.providers import get_embeddings

EMBEDDING_DIM = 384  # matches BAAI/bge-small-en-v1.5

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

    # Create a payload index on metadata.source so we can filter/delete
    # by source filename. Safe to call every time — Qdrant just skips
    # it if the index already exists.
    client.create_payload_index(
        collection_name=settings.qdrant_collection,
        field_name="metadata.source",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    _singleton_store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
    )
    return _singleton_store


def delete_by_source(source_name: str):
    """
    Deletes all chunks in Qdrant whose metadata 'source' matches this
    filename. Necessary because one uploaded file becomes MANY chunks —
    deleting the file means deleting every chunk that came from it.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    settings = get_settings()
    store = get_vector_store()

    store.client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[FieldCondition(key="metadata.source", match=MatchValue(value=source_name))]
        ),
    )


def list_all_sources() -> list[str]:
    """
    Returns every distinct 'source' filename currently indexed in
    Qdrant — the true list of what's actually searchable, regardless
    of how it got there. Used by the document list endpoint and by
    the digest agent to self-heal against stale memory entries.
    """
    settings = get_settings()
    store = get_vector_store()

    all_sources = set()
    offset = None
    while True:
        points, offset = store.client.scroll(
            collection_name=settings.qdrant_collection,
            with_payload=True,
            limit=100,
            offset=offset,
        )
        for point in points:
            source = point.payload.get("metadata", {}).get("source")
            if source:
                all_sources.add(source)
        if offset is None:
            break

    return sorted(all_sources)