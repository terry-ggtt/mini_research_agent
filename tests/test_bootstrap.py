import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from mini_research_agent.bootstrap import create_application
from mini_research_agent.tools.mcp import MCPRuntime


class RecordingModel:
    def __init__(self):
        self.bound_tool_names = []

    def bind_tools(self, tools):
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    async def ainvoke(self, _messages):
        return AIMessage(content="unused")


@tool
def web_lookup(query: str) -> str:
    """Search a stub public source."""

    return query


@tool("search_knowledge_base")
def knowledge_lookup(query: str) -> str:
    """Search a stub knowledge base."""

    return query


@tool("list_directory")
def list_directory(path: str) -> str:
    """List a stub directory."""

    return path


@tool("write_file")
def unsafe_write(path: str, content: str) -> str:
    """Represent an unsafe MCP write tool."""

    return path + content


@tool("read_file_range")
def bounded_read(path: str, start_line: int, end_line: int) -> str:
    """Read a stub bounded range."""

    return f"{path}:{start_line}-{end_line}"


@pytest.mark.asyncio
async def test_application_composes_web_rag_and_safe_mcp_tools():
    research_model = RecordingModel()
    runtime = MCPRuntime(
        client=object(),
        tools=[list_directory, unsafe_write],
    )

    async def scope_graph(state):
        return state

    application = await create_application(
        web_tools=[web_lookup],
        knowledge_tools=[knowledge_lookup],
        mcp_runtime=runtime,
        bounded_file_tools=[bounded_read],
        research_model=research_model,
        compression_model=RecordingModel(),
        supervisor_model=RecordingModel(),
        writer_model=RecordingModel(),
        scope_graph=scope_graph,
    )

    assert application.mcp is runtime
    assert research_model.bound_tool_names == [
        "web_lookup",
        "search_knowledge_base",
        "list_directory",
        "read_file_range",
    ]
    assert "write_file" not in research_model.bound_tool_names


@pytest.mark.asyncio
async def test_application_rejects_duplicate_tool_names():
    runtime = MCPRuntime(client=object(), tools=[])

    with pytest.raises(ValueError, match="Duplicate research tool name"):
        await create_application(
            web_tools=[web_lookup, web_lookup],
            knowledge_tools=[],
            mcp_runtime=runtime,
            bounded_file_tools=[],
            research_model=RecordingModel(),
            compression_model=RecordingModel(),
            supervisor_model=RecordingModel(),
            writer_model=RecordingModel(),
            scope_graph=lambda state: state,
        )
