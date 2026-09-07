import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mini_research_agent.multi_agent_supervisor_graph import (
    create_supervisor_graph,
    get_notes_from_tool_calls,
)
from mini_research_agent.schemas.supervisor import ResearchGap, ResearchReview


class StubBoundModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.invocations = []

    async def ainvoke(self, messages):
        self.invocations.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class StubSupervisorModel:
    def __init__(self, responses):
        self.bound_model = StubBoundModel(responses)
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self.bound_model


class StubReviewModel:
    def __init__(self, responses):
        self.structured_model = StubBoundModel(responses)
        self.schema = None
        self.method = None

    def with_structured_output(self, schema, *, method):
        self.schema = schema
        self.method = method
        return self.structured_model


class StubResearchGraph:
    def __init__(self, *, fail=False):
        self.inputs = []
        self.fail = fail

    async def ainvoke(self, graph_input):
        self.inputs.append(graph_input)
        if self.fail:
            raise RuntimeError("research failed")

        topic = graph_input["research_topic"]
        return {
            "compressed_research": f"finding:{topic}",
            "raw_notes": [f"raw:{topic}"],
        }


def tool_call(name, call_id, **args):
    return {
        "name": name,
        "args": args,
        "id": call_id,
        "type": "tool_call",
    }


def sufficient_review():
    return ResearchReview(sufficient=True)


@pytest.mark.asyncio
async def test_supervisor_stops_when_model_requests_no_tools():
    model = StubSupervisorModel([AIMessage(content="research is sufficient")])
    researcher = StubResearchGraph()
    graph = create_supervisor_graph(
        model=model,
        research_graph=researcher,
        review_model=StubReviewModel([sufficient_review()]),
    )

    result = await graph.ainvoke(
        {
            "supervisor_messages": [HumanMessage(content="research brief")],
            "research_brief": "research brief",
        }
    )

    assert researcher.inputs == []
    assert result["notes"] == []
    assert result["research_iterations"] == 1


@pytest.mark.asyncio
async def test_supervisor_runs_parallel_research_and_collects_results():
    model = StubSupervisorModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    tool_call("ConductResearch", "research-a", research_topic="A"),
                    tool_call("ConductResearch", "research-b", research_topic="B"),
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[tool_call("ResearchComplete", "complete")],
            ),
        ]
    )
    researcher = StubResearchGraph()
    graph = create_supervisor_graph(
        model=model,
        research_graph=researcher,
        review_model=StubReviewModel([sufficient_review()]),
    )

    result = await graph.ainvoke(
        {
            "supervisor_messages": [HumanMessage(content="research brief")],
            "research_brief": "research brief",
        }
    )

    assert researcher.inputs == [
        {"research_topic": "A"},
        {"research_topic": "B"},
    ]
    assert result["notes"] == ["finding:A", "finding:B"]
    assert result["raw_notes"] == ["raw:A", "raw:B"]
    assert result["research_iterations"] == 2

    second_model_call = model.bound_model.invocations[1]
    research_messages = [
        message
        for message in second_model_call
        if isinstance(message, ToolMessage)
        and message.name == "ConductResearch"
    ]
    assert [message.tool_call_id for message in research_messages] == [
        "research-a",
        "research-b",
    ]


@pytest.mark.asyncio
async def test_supervisor_think_tool_is_not_in_final_notes():
    model = StubSupervisorModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    tool_call(
                        "think_tool",
                        "think-1",
                        reflection="plan the research",
                    )
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[tool_call("ResearchComplete", "complete")],
            ),
        ]
    )
    graph = create_supervisor_graph(
        model=model,
        research_graph=StubResearchGraph(),
        review_model=StubReviewModel([sufficient_review()]),
    )

    result = await graph.ainvoke(
        {
            "supervisor_messages": [HumanMessage(content="research brief")],
            "research_brief": "research brief",
        }
    )

    assert result["notes"] == []
    assert any(
        isinstance(message, ToolMessage) and message.name == "think_tool"
        for message in result["supervisor_messages"]
    )


@pytest.mark.asyncio
async def test_supervisor_respects_iteration_limit_before_running_tools():
    model = StubSupervisorModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    tool_call(
                        "ConductResearch",
                        "research-a",
                        research_topic="A",
                    )
                ],
            )
        ]
    )
    researcher = StubResearchGraph()
    graph = create_supervisor_graph(
        model=model,
        research_graph=researcher,
        review_model=StubReviewModel([sufficient_review()]),
        max_researcher_iterations=1,
    )

    result = await graph.ainvoke(
        {
            "supervisor_messages": [HumanMessage(content="research brief")],
            "research_brief": "research brief",
        }
    )

    assert researcher.inputs == []
    assert result["research_iterations"] == 1
    assert result["notes"] == []


@pytest.mark.asyncio
async def test_supervisor_propagates_research_failure():
    model = StubSupervisorModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    tool_call(
                        "ConductResearch",
                        "research-a",
                        research_topic="A",
                    )
                ],
            )
        ]
    )
    graph = create_supervisor_graph(
        model=model,
        research_graph=StubResearchGraph(fail=True),
        review_model=StubReviewModel([sufficient_review()]),
    )

    with pytest.raises(RuntimeError, match="research failed"):
        await graph.ainvoke(
            {
                "supervisor_messages": [
                    HumanMessage(content="research brief")
                ],
                "research_brief": "research brief",
            }
        )


def test_get_notes_only_returns_conduct_research_messages():
    messages = [
        ToolMessage(
            content="internal thought",
            name="think_tool",
            tool_call_id="think-1",
        ),
        ToolMessage(
            content="research finding",
            name="ConductResearch",
            tool_call_id="research-1",
        ),
    ]

    assert get_notes_from_tool_calls(messages) == ["research finding"]


@pytest.mark.asyncio
async def test_review_research_returns_to_supervisor_for_actionable_gap():
    model = StubSupervisorModel(
        [
            AIMessage(
                content="",
                tool_calls=[tool_call("ResearchComplete", "complete-1")],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    tool_call(
                        "ConductResearch",
                        "research-1",
                        research_topic="Find the acceptance record and metric.",
                    )
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[tool_call("ResearchComplete", "complete-2")],
            ),
        ]
    )
    reviewer = StubReviewModel(
        [
            ResearchReview(
                sufficient=False,
                gaps=[
                    ResearchGap(
                        question="Was the target metric accepted?",
                        reason="No acceptance record was provided.",
                        priority="critical",
                        next_research_task=(
                            "Find the acceptance record and metric."
                        ),
                    )
                ],
            ),
            sufficient_review(),
        ]
    )
    researcher = StubResearchGraph()
    graph = create_supervisor_graph(
        model=model,
        research_graph=researcher,
        review_model=reviewer,
    )
    assert "review_research" in graph.get_graph().nodes

    result = await graph.ainvoke(
        {
            "supervisor_messages": [HumanMessage(content="research brief")],
            "research_brief": "research brief",
        }
    )

    assert researcher.inputs == [
        {"research_topic": "Find the acceptance record and metric."}
    ]
    assert result["review_count"] == 2
    assert result["termination_reason"] == "sufficient"
    assert result["notes"] == [
        "finding:Find the acceptance record and metric."
    ]
    assert "acceptance record" in model.bound_model.invocations[1][0].content
    assert any(
        isinstance(message, ToolMessage)
        and message.name == "ResearchComplete"
        and message.tool_call_id == "complete-1"
        for message in model.bound_model.invocations[1]
    )


@pytest.mark.asyncio
async def test_review_research_keeps_limitations_when_budget_is_exhausted():
    model = StubSupervisorModel(
        [
            AIMessage(
                content="",
                tool_calls=[tool_call("ResearchComplete", "complete")],
            )
        ]
    )
    reviewer = StubReviewModel(
        [
            ResearchReview(
                sufficient=False,
                gaps=[
                    ResearchGap(
                        question="Missing measurement",
                        reason="No primary source was found.",
                        priority="critical",
                        next_research_task="Find the primary measurement.",
                    )
                ],
                limitations=["Available evidence is secondary."],
            )
        ]
    )
    graph = create_supervisor_graph(
        model=model,
        research_graph=StubResearchGraph(),
        review_model=reviewer,
        max_researcher_iterations=1,
    )

    result = await graph.ainvoke(
        {
            "supervisor_messages": [HumanMessage(content="research brief")],
            "research_brief": "research brief",
        }
    )

    assert result["termination_reason"] == "budget_exhausted"
    assert "Available evidence is secondary." in result["research_limitations"]
    assert any("Missing measurement" in item for item in result["research_limitations"])


@pytest.mark.asyncio
async def test_review_research_propagates_review_failure():
    model = StubSupervisorModel(
        [
            AIMessage(
                content="",
                tool_calls=[tool_call("ResearchComplete", "complete")],
            )
        ]
    )
    graph = create_supervisor_graph(
        model=model,
        research_graph=StubResearchGraph(),
        review_model=StubReviewModel([RuntimeError("review failed")]),
    )

    with pytest.raises(RuntimeError, match="review failed"):
        await graph.ainvoke(
            {
                "supervisor_messages": [HumanMessage(content="research brief")],
                "research_brief": "research brief",
            }
        )


@pytest.mark.asyncio
async def test_same_turn_research_is_completed_before_review():
    model = StubSupervisorModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    tool_call("ConductResearch", "research", research_topic="A"),
                    tool_call("ResearchComplete", "complete"),
                ],
            )
        ]
    )
    reviewer = StubReviewModel([sufficient_review()])
    researcher = StubResearchGraph()
    graph = create_supervisor_graph(
        model=model,
        research_graph=researcher,
        review_model=reviewer,
    )

    result = await graph.ainvoke(
        {
            "supervisor_messages": [HumanMessage(content="research brief")],
            "research_brief": "research brief",
        }
    )

    assert researcher.inputs == [{"research_topic": "A"}]
    assert result["notes"] == ["finding:A"]
    review_input = reviewer.structured_model.invocations[0][1].content
    assert "finding:A" in review_input
