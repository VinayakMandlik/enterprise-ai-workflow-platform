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


@mcp.tool()
def draft_document(purpose: str, key_points: str) -> str:
    """
    Task/Action Agent tool: drafts a real deliverable (email, memo,
    summary document) based on a stated purpose and key points to
    include. Use this when the user wants something WRITTEN for them,
    not just information retrieved.
    """
    from app.providers import get_llm
    llm = get_llm()
    prompt = f"""Draft a professional, well-formatted document for this purpose:
"{purpose}"

Key points to include:
{key_points}

Write the complete draft, ready to send/use as-is."""
    resp = llm.invoke(prompt)
    return resp.content


@mcp.tool()
def deep_research(topic: str, max_searches: int = 3) -> str:
    """
    Research Agent tool: performs MULTIPLE searches with different
    phrasings across the knowledge base, then synthesizes a combined
    report — for questions that need cross-referencing multiple
    documents rather than a single lookup. Use this for broad or
    exploratory questions like "what has changed" or "summarize
    everything about X across our documents".
    """
    from app.providers import get_llm

    llm = get_llm()
    retriever = get_retriever(k=4)

    # Step 1: generate several different search phrasings for the topic
    query_gen_prompt = f"""Generate {max_searches} different, specific search
queries to thoroughly research this topic: "{topic}"
Respond with ONLY the queries, one per line, no numbering."""
    queries_raw = llm.invoke(query_gen_prompt).content
    queries = [q.strip() for q in queries_raw.split("\n") if q.strip()][:max_searches]

    # Step 2: run each search and collect results
    all_findings = []
    for q in queries:
        docs = retriever.invoke(q)
        for d in docs:
            all_findings.append(f"[source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}")

    if not all_findings:
        return "No relevant information found across any searches."

    combined = "\n\n---\n\n".join(all_findings)

    # Step 3: synthesize a single report from all findings
    synthesis_prompt = f"""Based on the following retrieved information,
write a clear, well-organized research summary about: "{topic}"

Retrieved information:
{combined[:6000]}

Write a synthesized report, citing sources where relevant."""
    report = llm.invoke(synthesis_prompt).content
    return report


if __name__ == "__main__":
    mcp.run(transport="stdio")