"""
Retriever: wraps the Qdrant vector store as a LangChain retriever
interface — the piece that actually answers "find me relevant docs
for this question" (the Retrieval half of RAG).
"""
from app.rag.vector_store import get_vector_store


def get_retriever(k: int = 4):
    """
    k = how many top-matching chunks to return per query.
    """
    store = get_vector_store()
    return store.as_retriever(search_kwargs={"k": k})