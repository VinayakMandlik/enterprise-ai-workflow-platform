"""
MCP client: connects to tool_server.py as a subprocess over stdio,
discovers whatever tools it exposes, and wraps them as LangChain tools
that a LangGraph agent can call.
"""
import asyncio
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.tools import StructuredTool


class MCPToolRegistry:
    def __init__(self, server_module: str = "app.mcp.tool_server"):
        self._server_params = StdioServerParameters(
            command="python", args=["-m", server_module]
        )
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def connect(self):
        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(
            stdio_client(self._server_params)
        )
        self._session = await self._stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()

    async def close(self):
        if self._stack:
            await self._stack.aclose()

    async def get_langchain_tools(self) -> list[StructuredTool]:
        assert self._session is not None, "call connect() first"
        listing = await self._session.list_tools()
        return [self._wrap_tool(t.name, t.description, t.inputSchema) for t in listing.tools]

    def _wrap_tool(self, name: str, description: str, input_schema: dict) -> StructuredTool:
        session = self._session
        args_model = self._build_args_model(name, input_schema)

        async def _call(**kwargs):
            result = await session.call_tool(name, arguments=kwargs)
            return "\n".join(c.text for c in result.content if hasattr(c, "text"))

        return StructuredTool.from_function(
            coroutine=_call,
            name=name,
            description=description or name,
            args_schema=args_model,
        )

    @staticmethod
    def _build_args_model(tool_name: str, input_schema: dict):
        """
        Converts an MCP tool's JSON schema into a real Pydantic model,
        so StructuredTool exposes the ACTUAL parameter names (query,
        top_k, etc.) to the LLM instead of a generic catch-all.
        """
        from pydantic import create_model

        type_map = {"string": str, "integer": int, "number": float, "boolean": bool}
        properties = input_schema.get("properties", {})
        required = set(input_schema.get("required", []))

        fields = {}
        for field_name, field_spec in properties.items():
            py_type = type_map.get(field_spec.get("type"), str)
            if field_name in required:
                fields[field_name] = (py_type, ...)
            else:
                fields[field_name] = (py_type, field_spec.get("default", None))

        return create_model(f"{tool_name}_Args", **fields)


async def load_mcp_tools() -> tuple[MCPToolRegistry, list[StructuredTool]]:
    """Convenience entrypoint the LangGraph agent will use."""
    registry = MCPToolRegistry()
    await registry.connect()
    tools = await registry.get_langchain_tools()
    return registry, tools


if __name__ == "__main__":
    async def _demo():
        registry, tools = await load_mcp_tools()
        print("Discovered MCP tools:")
        for t in tools:
            print(f" - {t.name}: {t.description}")
        await registry.close()

    asyncio.run(_demo())