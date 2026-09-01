"""
The agentic workflow graph — the centerpiece: task routing, state
management, tool execution, and workflow transitions, all via LangGraph.
"""
from __future__ import annotations
from typing import Annotated, TypedDict
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage

from app.providers import get_llm
from app.memory.long_term import get_long_term_memory


class WorkflowState(TypedDict):
    # Short-term / working memory: accumulates across the graph run,
    # and (via the checkpointer) across turns within the same thread.
    messages: Annotated[list[AnyMessage], operator.add]
    user_id: str
    route: str  # decided by the router node


ROUTER_SYSTEM_PROMPT = """You are a routing controller for an enterprise
AI workflow platform. Classify the user's request into exactly one of:
- "knowledge_query"  : needs document/knowledge-base lookup (RAG)
- "system_lookup"    : needs a report/status lookup from internal systems
- "general"          : can be answered directly, no tools needed

Respond with only the single label."""


def build_router_node():
    llm = get_llm()

    def router(state: WorkflowState) -> WorkflowState:
        last_user_msg = state["messages"][-1].content
        decision = llm.invoke(
            [SystemMessage(content=ROUTER_SYSTEM_PROMPT), HumanMessage(content=last_user_msg)]
        )
        route = decision.content.strip().lower()
        if route not in ("knowledge_query", "system_lookup", "general"):
            route = "general"
        return {"messages": [], "route": route, "user_id": state["user_id"]}

    return router


def build_agent_node(tools):
    """LLM node bound to MCP-discovered tools; decides which tool to
    call and with what arguments, or answers directly."""
    llm = get_llm().bind_tools(tools)

    def agent(state: WorkflowState) -> WorkflowState:
        system = SystemMessage(
            content=(
                "You are an enterprise workflow assistant. Use the available "
                "tools when they would give a more accurate, grounded answer. "
                "Always cite the source document when you use search results. "
                "Use deep_research for broad/exploratory questions needing "
                "multiple documents. Use draft_document when the user wants "
                "something written, like an email or memo. "
                "For general questions unrelated to enterprise documents or "
                "systems, just answer directly and helpfully."
            )
        )
        response = llm.invoke([system] + state["messages"])
        return {"messages": [response]}

    return agent


def build_critic_node():
    """
    Second, independent LLM pass that reviews the main agent's answer
    for relevance/quality before it's finalized. This is a real
    multi-agent pattern: one agent produces, another verifies.
    """
    llm = get_llm()

    def critic(state: WorkflowState) -> WorkflowState:
        last_ai_msg = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None
        )
        if not last_ai_msg or not last_ai_msg.content:
            return {}

        user_question = next(
            (m for m in state["messages"] if isinstance(m, HumanMessage)), None
        )
        if not user_question:
            return {}

        review_prompt = f"""You are a quality reviewer. A user asked:
"{user_question.content}"

An AI assistant answered:
"{last_ai_msg.content}"

Does this answer actually address the question, directly and clearly?
Respond with only "APPROVED" or "NEEDS_REVISION: <brief reason>"."""

        verdict = llm.invoke([HumanMessage(content=review_prompt)]).content.strip()

        # Store the verdict as metadata for now — later this could trigger
        # a real retry loop back to the agent node.
        return {"messages": [], "route": state["route"] + f" | critic: {verdict}"}

    return critic


def build_memory_writer_node():
    def memory_writer(state: WorkflowState) -> WorkflowState:
        memory = get_long_term_memory()
        last_ai_msg = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None
        )
        if last_ai_msg:
            memory.remember(
                user_id=state["user_id"],
                key="last_interaction",
                value={"summary": last_ai_msg.content[:500]},
            )
        return {}

    return memory_writer


def build_graph(tools):
    """
    Assembles the full StateGraph. `tools` are the LangChain-wrapped
    MCP tools discovered at startup (see app/mcp/tool_client.py).

    Flow: router (classifies, for observability) -> agent (always
    generates a response, calling tools if it decides to) -> tools
    (loop back to agent if a tool was called) -> critic (reviews the
    final answer) -> memory_writer -> END.
    """
    from langgraph.prebuilt import ToolNode, tools_condition

    graph = StateGraph(WorkflowState)

    graph.add_node("router", build_router_node())
    graph.add_node("agent", build_agent_node(tools))
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("critic", build_critic_node())
    graph.add_node("memory_writer", build_memory_writer_node())

    graph.add_edge(START, "router")
    graph.add_edge("router", "agent")  # agent ALWAYS runs; route is just metadata now
    graph.add_conditional_edges(
        "agent", tools_condition, {"tools": "tools", END: "critic"}
    )
    graph.add_edge("tools", "agent")  # loop back so the LLM can use tool results
    graph.add_edge("critic", "memory_writer")
    graph.add_edge("memory_writer", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def run_intake_agent(text: str, filename: str, user_id: str) -> dict:
    """
    Document Intake Agent: runs automatically when a document is
    uploaded (not part of the chat graph). Summarizes the document and
    extracts key facts, saving both to long-term memory — genuine
    automation triggered by an event, not a user question.
    """
    llm = get_llm()

    intake_prompt = f"""You just received a new document titled "{filename}".
Analyze it and respond in this exact format:

SUMMARY: <2-3 sentence summary>
KEY_FACTS: <bullet list of important facts, numbers, or dates, one per line>

Document content:
{text[:4000]}"""

    response = llm.invoke([HumanMessage(content=intake_prompt)])

    memory = get_long_term_memory()
    memory.remember(
        user_id=user_id,
        key=f"document_intake:{filename}",
        value={"filename": filename, "analysis": response.content},
    )

    return {"filename": filename, "analysis": response.content}


def run_scheduled_digest_agent(user_id: str = "demo-user") -> dict:
    """
    Scheduled Agent: designed to run on a timer (see app/scheduler.py),
    independent of any user request. Reviews document intake summaries,
    but ONLY for documents that still actually exist in the knowledge
    base right now — self-healing against stale memory entries left
    behind by documents that were deleted, rather than requiring
    manual cleanup every time.

    NOTE: on free-tier hosting that sleeps when idle, this only fires
    while the app happens to be awake — a known limitation of running
    background schedulers on free hosting tiers.
    """
    from app.rag.vector_store import list_all_sources

    llm = get_llm()
    memory = get_long_term_memory()

    # Source of truth: what's actually still indexed right now.
    currently_existing = set(list_all_sources())

    all_memories = memory.recall_all(user_id)
    intake_summaries = []
    stale_keys = []

    for key, value in all_memories.items():
        if not key.startswith("document_intake:"):
            continue
        filename = key.split("document_intake:", 1)[1]
        if filename in currently_existing:
            intake_summaries.append(value["analysis"])
        else:
            stale_keys.append(key)  # document was deleted, but memory wasn't cleaned up

    # Self-heal: clean up any stale entries we just found, so future
    # calls don't need to re-check them.
    for key in stale_keys:
        memory.forget(user_id=user_id, key=key)

    if not intake_summaries:
        digest = "No documents have been processed yet."
    else:
        combined = "\n\n---\n\n".join(intake_summaries)
        digest_prompt = f"""Here are summaries of all documents currently in
the knowledge base:

{combined[:6000]}

Write a brief consolidated daily digest highlighting anything notable."""
        digest = llm.invoke(digest_prompt).content

    memory.remember(user_id=user_id, key="daily_digest", value={"digest": digest})
    return {"digest": digest}