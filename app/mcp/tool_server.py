"""
MCP (Model Context Protocol) server.

Tools are defined ONCE here and exposed over the real MCP protocol.
Any MCP-compatible client (our LangGraph agent, or Claude Desktop, or
anything else) can discover and call these identically — that's the
"standardized tool registration" and interoperability claim made real.

Run standalone with:  python -m app.mcp.tool_server
"""
import os
import logging
import warnings

# CRITICAL: MCP's stdio transport requires stdout to contain ONLY
# JSON-RPC protocol messages. Any stray print/log statement from a
# library (HuggingFace, httpx, etc.) corrupts the stream and causes
# the client to hang forever waiting for a well-formed response.
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from mcp.server.fastmcp import FastMCP
from app.rag.retriever import get_retriever

mcp = FastMCP("enterprise-workflow-tools")

# Pre-warm the embedding model and vector store connection at server
# startup, not on first tool call. A slow first response during MCP's
# stdio round-trip can cause the client to lose the response on some
# setups — pre-warming avoids ever hitting that slow path live.
from app.rag.vector_store import get_vector_store
get_vector_store()


@mcp.tool()
def search_knowledge_base(query: str, top_k: int = 4) -> str:
    """Semantic search over ingested enterprise documents (RAG retrieval)."""
    retriever = get_retriever(k=top_k)
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found."
    return "\n\n---\n\n".join(
        f"[source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )


@mcp.tool()
def summarize_text(text: str) -> str:
    """Summarize an arbitrary chunk of enterprise document text."""
    from app.providers import get_llm
    llm = get_llm()
    resp = llm.invoke(f"Summarize the following in 3 bullet points:\n\n{text}")
    return resp.content


if __name__ == "__main__":
    mcp.run(transport="stdio")