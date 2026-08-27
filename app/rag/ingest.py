"""
Document ingestion: turns raw enterprise documents into chunked,
embedded records inside Qdrant.
"""
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.rag.vector_store import get_vector_store


def ingest_directory(directory: str = "./data/documents") -> int:
    """
    Walk a directory of enterprise docs, chunk them, embed them, and
    upsert into Qdrant. Returns number of chunks ingested.
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      # max characters per chunk
        chunk_overlap=120,   # overlap between consecutive chunks
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    docs: list[Document] = []
    for file_path in dir_path.glob("**/*"):
        if file_path.suffix.lower() not in (".txt", ".md"):
            continue

        text = file_path.read_text(encoding="utf-8", errors="ignore")
        chunks = splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"source": file_path.name, "chunk": i},
                )
            )

    if not docs:
        return 0

    store = get_vector_store()
    store.add_documents(docs)
    return len(docs)
def ingest_text(text: str, source_name: str) -> int:
    """
    Chunk and embed a raw text string (e.g. from an uploaded file),
    without needing it to exist as a file on local disk first.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)

    docs = [
        Document(page_content=chunk, metadata={"source": source_name, "chunk": i})
        for i, chunk in enumerate(chunks)
    ]

    if not docs:
        return 0

    store = get_vector_store()
    store.add_documents(docs)
    return len(docs)


if __name__ == "__main__":
    count = ingest_directory()
    print(f"Ingested {count} chunks.")