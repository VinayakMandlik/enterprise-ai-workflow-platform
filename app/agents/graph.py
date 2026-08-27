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
                "For general questions unrelated to enterprise documents or "
                "systems, just answer directly and helpfully."
            )
        )
        response = llm.invoke([system] + state["messages"])
        return {"messages": [response]}

    return agent


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
    (loop back to agent if a tool was called) -> memory_writer -> END.
    """
    from langgraph.prebuilt import ToolNode, tools_condition

    graph = StateGraph(WorkflowState)

    graph.add_node("router", build_router_node())
    graph.add_node("agent", build_agent_node(tools))
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("memory_writer", build_memory_writer_node())

    graph.add_edge(START, "router")
    graph.add_edge("router", "agent")  # agent ALWAYS runs; route is just metadata now
    graph.add_conditional_edges(
        "agent", tools_condition, {"tools": "tools", END: "memory_writer"}
    )
    graph.add_edge("tools", "agent")  # loop back so the LLM can use tool results
    graph.add_edge("memory_writer", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)