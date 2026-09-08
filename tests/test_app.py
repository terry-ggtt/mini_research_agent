import pytest
from langchain_core.messages import AIMessage, HumanMessage

from mini_research_agent.graphs.app import create_app_graph


class StubGraph:
    """Return a fixed result and record all graph invocations."""

    def __init__(self, result):
        self.result = result
        self.inputs = []

    async def ainvoke(self, graph_input):
        self.inputs.append(graph_input)
        return self.result


@pytest.mark.asyncio
async def test_clear_request_runs_scope_then_research():
    user_message = HumanMessage(content="研究旧金山咖啡店")
    verification = AIMessage(content="信息充分，开始研究。")
    brief = "研究旧金山咖啡店，并比较咖啡品质。"

    scope = StubGraph(
        {
            "messages": [user_message, verification],
            "research_brief": brief,
        }
    )
    researcher = StubGraph(
        {
            "compressed_research": "压缩后的研究结果",
            "raw_notes": ["来源一", "来源二"],
        }
    )

    graph = create_app_graph(scope_graph=scope, research_graph=researcher)
    result = await graph.ainvoke({"messages": [user_message]})

    assert scope.inputs == [{"messages": [user_message]}]
    assert researcher.inputs == [{"research_topic": brief}]
    assert [message.content for message in result["messages"]] == [
        "研究旧金山咖啡店",
        "信息充分，开始研究。",
    ]
    assert result["research_brief"] == brief
    assert result["compressed_research"] == "压缩后的研究结果"
    assert result["raw_notes"] == ["来源一", "来源二"]


@pytest.mark.asyncio
async def test_clarification_stops_before_research():
    user_message = HumanMessage(content="研究最好的咖啡店")
    question = AIMessage(content="你希望研究哪个城市？")

    scope = StubGraph(
        {
            "messages": [user_message, question],
        }
    )
    researcher = StubGraph(
        {
            "compressed_research": "不应生成",
            "raw_notes": ["不应生成"],
        }
    )

    graph = create_app_graph(scope_graph=scope, research_graph=researcher)
    result = await graph.ainvoke({"messages": [user_message]})

    assert researcher.inputs == []
    assert [message.content for message in result["messages"]] == [
        "研究最好的咖啡店",
        "你希望研究哪个城市？",
    ]
    assert result["research_brief"] is None
    assert "compressed_research" not in result
    assert result["raw_notes"] == []


@pytest.mark.asyncio
async def test_research_graph_receives_only_the_generated_brief():
    user_message = HumanMessage(content="研究 AI Agent")
    brief = "比较 LangGraph Agent 的状态、节点和路由设计。"
    scope = StubGraph(
        {
            "messages": [user_message],
            "research_brief": brief,
        }
    )
    researcher = StubGraph(
        {
            "compressed_research": "研究结果",
        }
    )

    result = await create_app_graph(
        scope_graph=scope,
        research_graph=researcher,
    ).ainvoke({"messages": [user_message]})

    assert researcher.inputs == [{"research_topic": brief}]
    assert result["raw_notes"] == []
