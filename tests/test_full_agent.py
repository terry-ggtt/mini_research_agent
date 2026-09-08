import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from mini_research_agent.graphs.full_agent import create_full_agent


class StubWriter:
    def __init__(self, content="final report"):
        self.content = content
        self.invocations = []

    async def ainvoke(self, messages):
        self.invocations.append(messages)
        return AIMessage(content=self.content)


def runnable(function):
    return RunnableLambda(function)


@pytest.mark.asyncio
async def test_full_agent_stops_for_clarification():
    calls = []

    async def scope_node(state):
        calls.append("scope")
        return {
            "messages": [AIMessage(content="Which city should be researched?")],
        }

    async def supervisor_node(state):
        calls.append("supervisor")
        return {"notes": ["must not run"]}

    writer = StubWriter()
    graph = create_full_agent(
        writer_model=writer,
        scope_graph=runnable(scope_node),
        supervisor_graph=runnable(supervisor_node),
    )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="find the best coffee shop")]}
    )

    assert calls == ["scope"]
    assert writer.invocations == []
    assert "final_report" not in result
    assert result["messages"][-1].content == "Which city should be researched?"


@pytest.mark.asyncio
async def test_full_agent_runs_scope_supervisor_and_writer():
    calls = []

    async def scope_node(state):
        calls.append("scope")
        return {
            "research_brief": "compare coffee quality",
            "supervisor_messages": [
                HumanMessage(content="compare coffee quality")
            ],
        }

    async def supervisor_node(state):
        calls.append("supervisor")
        assert state["research_brief"] == "compare coffee quality"
        assert state["supervisor_messages"][-1].content == (
            "compare coffee quality"
        )
        return {
            "notes": ["finding one", "finding two"],
            "raw_notes": ["raw evidence"],
        }

    writer = StubWriter("generated final report")
    graph = create_full_agent(
        writer_model=writer,
        scope_graph=runnable(scope_node),
        supervisor_graph=runnable(supervisor_node),
    )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="research coffee shops")]}
    )

    assert calls == ["scope", "supervisor"]
    assert result["research_brief"] == "compare coffee quality"
    assert result["notes"] == ["finding one", "finding two"]
    assert result["raw_notes"] == ["raw evidence"]
    assert result["final_report"] == "generated final report"
    assert result["messages"][-1].content == "generated final report"

    writer_prompt = writer.invocations[0][0].content
    assert "compare coffee quality" in writer_prompt
    assert "finding one" in writer_prompt
    assert "finding two" in writer_prompt


@pytest.mark.asyncio
async def test_full_agent_uses_injected_subgraphs():
    calls = []

    async def scope_node(state):
        calls.append("injected scope")
        return {
            "research_brief": "brief",
            "supervisor_messages": [HumanMessage(content="brief")],
        }

    async def supervisor_node(state):
        calls.append("injected supervisor")
        return {"notes": ["finding"], "raw_notes": []}

    graph = create_full_agent(
        writer_model=StubWriter(),
        scope_graph=runnable(scope_node),
        supervisor_graph=runnable(supervisor_node),
        checkpointer=None,
    )

    await graph.ainvoke({"messages": [HumanMessage(content="question")]})

    assert calls == ["injected scope", "injected supervisor"]


@pytest.mark.asyncio
async def test_full_agent_can_be_created_without_checkpointer():
    async def scope_node(state):
        return {
            "research_brief": "brief",
            "supervisor_messages": [HumanMessage(content="brief")],
        }

    async def supervisor_node(state):
        return {"notes": ["finding"], "raw_notes": []}

    graph = create_full_agent(
        writer_model=StubWriter(),
        scope_graph=runnable(scope_node),
        supervisor_graph=runnable(supervisor_node),
    )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="question")]}
    )

    assert result["final_report"] == "final report"
