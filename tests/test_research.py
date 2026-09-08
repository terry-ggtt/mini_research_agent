import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from mini_research_agent.graphs.research import create_research_graph


class StubRunnable:
    def __init__(self, responses):
        self.responses = list(responses)
        self.invocations = []

    async def ainvoke(self, messages):
        self.invocations.append(messages)
        return self.responses.pop(0)


class StubResearchModel:
    def __init__(self, *, bound_responses, unbound_responses=None):
        self.bound_model = StubRunnable(bound_responses)
        self.unbound_model = StubRunnable(unbound_responses or [])
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self.bound_model

    async def ainvoke(self, messages):
        return await self.unbound_model.ainvoke(messages)


class StubCompressionModel:
    def __init__(self, content="compressed research"):
        self.content = content
        self.invocations = []

    async def ainvoke(self, messages):
        self.invocations.append(messages)
        return AIMessage(content=self.content)


@pytest.mark.asyncio
async def test_research_graph_compresses_when_model_requests_no_tools():
    model = StubResearchModel(
        bound_responses=[AIMessage(content="research complete")],
    )
    compression_model = StubCompressionModel("compressed result")
    graph = create_research_graph(
        model=model,
        compress_model=compression_model,
        tools=[],
    )

    result = await graph.ainvoke({"research_topic": "study coffee quality"})

    first_call = model.bound_model.invocations[0]
    assert isinstance(first_call[0], SystemMessage)
    assert isinstance(first_call[1], HumanMessage)
    assert first_call[1].content == "study coffee quality"
    assert result["compressed_research"] == "compressed result"
    assert result["raw_notes"] == ["research complete"]


@pytest.mark.asyncio
async def test_research_graph_executes_tool_and_returns_tool_message_to_model():
    calls = []

    @tool
    def lookup(query: str) -> str:
        """Look up evidence for one research query."""

        calls.append(query)
        return "tool evidence"

    model = StubResearchModel(
        bound_responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup",
                        "args": {"query": "coffee"},
                        "id": "lookup-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="research complete"),
        ]
    )
    graph = create_research_graph(
        model=model,
        compress_model=StubCompressionModel(),
        tools=[lookup],
    )

    result = await graph.ainvoke({"research_topic": "study coffee"})

    assert calls == ["coffee"]
    second_call = model.bound_model.invocations[1]
    tool_message = next(
        message for message in second_call if isinstance(message, ToolMessage)
    )
    assert tool_message.content == "tool evidence"
    assert tool_message.name == "lookup"
    assert tool_message.tool_call_id == "lookup-1"
    assert result["compressed_research"] == "compressed research"


@pytest.mark.asyncio
async def test_research_graph_returns_tool_failure_to_model():
    @tool
    def broken_lookup(query: str) -> str:
        """Raise an error while looking up evidence."""

        raise RuntimeError(f"cannot search {query}")

    model = StubResearchModel(
        bound_responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "broken_lookup",
                        "args": {"query": "coffee"},
                        "id": "broken-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="finished after error"),
        ]
    )
    graph = create_research_graph(
        model=model,
        compress_model=StubCompressionModel(),
        tools=[broken_lookup],
    )

    await graph.ainvoke({"research_topic": "study coffee"})

    second_call = model.bound_model.invocations[1]
    error_message = next(
        message for message in second_call if isinstance(message, ToolMessage)
    )
    assert "RuntimeError" in error_message.content
    assert "cannot search coffee" in error_message.content
    assert error_message.tool_call_id == "broken-1"


@pytest.mark.asyncio
async def test_research_graph_uses_unbound_model_when_tool_budget_is_zero():
    model = StubResearchModel(
        bound_responses=[],
        unbound_responses=[AIMessage(content="finished without tools")],
    )
    graph = create_research_graph(
        model=model,
        compress_model=StubCompressionModel(),
        tools=[],
        max_tool_iterations=0,
    )

    result = await graph.ainvoke({"research_topic": "no tools allowed"})

    assert model.bound_model.invocations == []
    assert len(model.unbound_model.invocations) == 1
    assert result["compressed_research"] == "compressed research"


@pytest.mark.asyncio
async def test_research_graph_requires_research_topic():
    graph = create_research_graph(
        model=StubResearchModel(
            bound_responses=[AIMessage(content="unused")],
        ),
        compress_model=StubCompressionModel(),
        tools=[],
    )

    with pytest.raises(KeyError, match="research_topic"):
        await graph.ainvoke({})


@pytest.mark.asyncio
async def test_research_graph_enforces_file_read_operation_budget():
    calls = []

    @tool("read_file_range")
    def read_file_range(path: str, start_line: int, end_line: int) -> str:
        """Read a bounded file range."""

        calls.append((path, start_line, end_line))
        return "bounded evidence"

    model = StubResearchModel(
        bound_responses=[
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "read_file_range",
                    "args": {"path": "notes.txt", "start_line": 1, "end_line": 2},
                    "id": "read-1",
                    "type": "tool_call",
                }],
            ),
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "read_file_range",
                    "args": {"path": "notes.txt", "start_line": 3, "end_line": 4},
                    "id": "read-2",
                    "type": "tool_call",
                }],
            ),
            AIMessage(content="finished"),
        ]
    )
    graph = create_research_graph(
        model=model,
        compress_model=StubCompressionModel(),
        tools=[read_file_range],
        max_file_read_operations=1,
    )

    await graph.ainvoke({"research_topic": "read notes"})

    assert calls == [("notes.txt", 1, 2)]
    final_model_call = model.bound_model.invocations[-1]
    rejected = next(
        message
        for message in final_model_call
        if isinstance(message, ToolMessage) and message.tool_call_id == "read-2"
    )
    assert "budget has been exhausted" in rejected.content


@pytest.mark.asyncio
async def test_research_graph_rejects_already_covered_file_range():
    calls = []

    @tool("read_file_range")
    def read_file_range(path: str, start_line: int, end_line: int) -> str:
        """Read a bounded file range."""

        calls.append((path, start_line, end_line))
        return "bounded evidence"

    model = StubResearchModel(
        bound_responses=[
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "read_file_range",
                    "args": {"path": "notes.txt", "start_line": 1, "end_line": 10},
                    "id": "read-1",
                    "type": "tool_call",
                }],
            ),
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "read_file_range",
                    "args": {"path": "NOTES.txt", "start_line": 3, "end_line": 5},
                    "id": "read-2",
                    "type": "tool_call",
                }],
            ),
            AIMessage(content="finished"),
        ]
    )
    graph = create_research_graph(
        model=model,
        compress_model=StubCompressionModel(),
        tools=[read_file_range],
    )

    await graph.ainvoke({"research_topic": "read notes"})

    assert calls == [("notes.txt", 1, 10)]
    final_model_call = model.bound_model.invocations[-1]
    rejected = next(
        message
        for message in final_model_call
        if isinstance(message, ToolMessage) and message.tool_call_id == "read-2"
    )
    assert "already covered" in rejected.content


@pytest.mark.asyncio
async def test_research_graph_enforces_cumulative_file_character_budget():
    calls = []

    @tool("read_file_range")
    def read_file_range(path: str, start_line: int, end_line: int) -> str:
        """Read a bounded file range."""

        calls.append((path, start_line, end_line))
        return "x" * 50

    model = StubResearchModel(
        bound_responses=[
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "read_file_range",
                    "args": {"path": "notes.txt", "start_line": 1, "end_line": 2},
                    "id": "read-1",
                    "type": "tool_call",
                }],
            ),
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "read_file_range",
                    "args": {"path": "notes.txt", "start_line": 3, "end_line": 4},
                    "id": "read-2",
                    "type": "tool_call",
                }],
            ),
            AIMessage(content="finished"),
        ]
    )
    graph = create_research_graph(
        model=model,
        compress_model=StubCompressionModel(),
        tools=[read_file_range],
        max_file_characters=10,
    )

    await graph.ainvoke({"research_topic": "read notes"})

    assert calls == [("notes.txt", 1, 2)]
    second_model_call = model.bound_model.invocations[1]
    truncated = next(
        message
        for message in second_model_call
        if isinstance(message, ToolMessage) and message.tool_call_id == "read-1"
    )
    assert "truncated by cumulative" in truncated.content
