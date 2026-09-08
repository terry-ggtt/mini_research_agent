import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

import mini_research_agent.graphs.mcp_research as mcp_module


class StubMCPClient:
    def __init__(self, tools):
        self.tools = tools
        self.get_tools_calls = 0

    async def get_tools(self):
        self.get_tools_calls += 1
        return self.tools


class StubBoundModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.invocations = []

    async def ainvoke(self, messages):
        self.invocations.append(messages)
        return self.responses.pop(0)


class StubResearchModel:
    def __init__(self, *, bound_responses, unbound_responses=None):
        self.bound_model = StubBoundModel(bound_responses)
        self.unbound_responses = list(unbound_responses or [])
        self.bound_tools = []
        self.unbound_invocations = []

    def bind_tools(self, tools=None, **kwargs):
        bound_tools = tools if tools is not None else kwargs["tools"]
        self.bound_tools.append(bound_tools)
        return self.bound_model

    async def ainvoke(self, messages):
        self.unbound_invocations.append(messages)
        return self.unbound_responses.pop(0)


class StubCompressionModel:
    def __init__(self, content="compressed research"):
        self.content = content
        self.invocations = []

    async def ainvoke(self, messages):
        self.invocations.append(messages)
        return AIMessage(content=self.content)


@pytest.fixture(autouse=True)
def reset_module_runtime(monkeypatch):
    """Keep the current module-level cache from leaking between tests."""

    for name in ("_client", "_tools", "_model_with_tools"):
        if hasattr(mcp_module, name):
            monkeypatch.setattr(mcp_module, name, None)


def build_graph(
    monkeypatch,
    *,
    client,
    model,
    compression_model,
    max_tool_iterations=5,
):
    """Build a graph without allowing a real MCP subprocess to start."""

    monkeypatch.setattr(
        mcp_module,
        "MultiServerMCPClient",
        lambda _config: client,
    )
    return mcp_module.create_mcp_research_graph(
        model=model,
        compress_model=compression_model,
        mcp_client=client,
        max_tool_iterations=max_tool_iterations,
    )


@pytest.mark.asyncio
async def test_graph_initializes_messages_and_compresses_without_tool_calls(
    monkeypatch,
):
    client = StubMCPClient([])
    model = StubResearchModel(
        bound_responses=[AIMessage(content="research finished")]
    )
    compression_model = StubCompressionModel("compressed result")
    graph = build_graph(
        monkeypatch,
        client=client,
        model=model,
        compression_model=compression_model,
    )

    result = await graph.ainvoke({"research_topic": "study local documents"})

    model_messages = model.bound_model.invocations[0]
    assert isinstance(model_messages[0], SystemMessage)
    assert isinstance(model_messages[1], HumanMessage)
    assert model_messages[1].content == "study local documents"
    assert result["compressed_research"] == "compressed result"
    assert "research finished" in result["raw_notes"][0]
    assert client.get_tools_calls == 1
    assert len(model.bound_tools) == 1


@pytest.mark.asyncio
async def test_graph_executes_mcp_tool_and_returns_its_result_to_model(
    monkeypatch,
):
    calls = []

    @tool
    async def read_file(path: str) -> str:
        """Read one local research file."""

        calls.append(path)
        return "document evidence"

    model = StubResearchModel(
        bound_responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"path": "notes.md"},
                        "id": "call-read-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="enough evidence"),
        ]
    )
    client = StubMCPClient([read_file])
    compression_model = StubCompressionModel()
    graph = build_graph(
        monkeypatch,
        client=client,
        model=model,
        compression_model=compression_model,
    )

    result = await graph.ainvoke({"research_topic": "read the notes"})

    assert calls == ["notes.md"]
    second_model_call = model.bound_model.invocations[1]
    tool_messages = [message for message in second_model_call if message.type == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "document evidence"
    assert tool_messages[0].tool_call_id == "call-read-1"
    assert "document evidence" in result["raw_notes"][0]
    assert client.get_tools_calls == 1
    assert len(model.bound_tools) == 1


@pytest.mark.asyncio
async def test_tool_failure_is_returned_to_the_model(monkeypatch):
    @tool
    async def broken_tool(value: str) -> str:
        """Raise an error while reading research data."""

        raise RuntimeError(f"cannot read {value}")

    model = StubResearchModel(
        bound_responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "broken_tool",
                        "args": {"value": "document"},
                        "id": "call-broken-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="finished after tool error"),
        ]
    )
    graph = build_graph(
        monkeypatch,
        client=StubMCPClient([broken_tool]),
        model=model,
        compression_model=StubCompressionModel(),
    )

    result = await graph.ainvoke({"research_topic": "read a document"})

    second_model_call = model.bound_model.invocations[1]
    error_message = next(
        message for message in second_model_call if message.type == "tool"
    )
    assert "cannot read document" in error_message.content
    assert "cannot read document" in result["raw_notes"][0]


@pytest.mark.asyncio
async def test_injected_mcp_client_is_used(monkeypatch):
    client = StubMCPClient([])

    def fail_if_default_client_is_created(_config):
        raise AssertionError("the injected MCP client was ignored")

    monkeypatch.setattr(
        mcp_module,
        "MultiServerMCPClient",
        fail_if_default_client_is_created,
    )
    graph = mcp_module.create_mcp_research_graph(
        model=StubResearchModel(
            bound_responses=[AIMessage(content="finished")]
        ),
        compress_model=StubCompressionModel(),
        mcp_client=client,
    )

    await graph.ainvoke({"research_topic": "test injection"})

    assert client.get_tools_calls == 1


@pytest.mark.asyncio
async def test_zero_tool_budget_uses_the_unbound_model(monkeypatch):
    client = StubMCPClient([])
    model = StubResearchModel(
        bound_responses=[],
        unbound_responses=[AIMessage(content="finished without tools")],
    )
    graph = build_graph(
        monkeypatch,
        client=client,
        model=model,
        compression_model=StubCompressionModel(),
        max_tool_iterations=0,
    )

    result = await graph.ainvoke({"research_topic": "no tools allowed"})

    assert len(model.unbound_invocations) == 1
    assert model.bound_model.invocations == []
    assert result["compressed_research"] == "compressed research"


@pytest.mark.asyncio
async def test_two_graphs_do_not_share_cached_models_or_clients(monkeypatch):
    first_client = StubMCPClient([])
    first_model = StubResearchModel(
        bound_responses=[AIMessage(content="first result")]
    )
    first_graph = build_graph(
        monkeypatch,
        client=first_client,
        model=first_model,
        compression_model=StubCompressionModel("first compressed"),
    )
    await first_graph.ainvoke({"research_topic": "first topic"})

    second_client = StubMCPClient([])
    second_model = StubResearchModel(
        bound_responses=[AIMessage(content="second result")]
    )
    second_graph = build_graph(
        monkeypatch,
        client=second_client,
        model=second_model,
        compression_model=StubCompressionModel("second compressed"),
    )
    result = await second_graph.ainvoke({"research_topic": "second topic"})

    assert second_client.get_tools_calls == 1
    assert len(second_model.bound_model.invocations) == 1
    assert result["compressed_research"] == "second compressed"
