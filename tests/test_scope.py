import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from mini_research_agent.schemas.scope_models import (
    ClarificationDecision,
    ResearchBriefOutput,
)
from mini_research_agent.graphs.scope import create_scope_graph


class StubModel:
    """Return queued structured outputs and record every model invocation."""

    def __init__(self, responses):
        self.responses = {
            schema: list(schema_responses)
            for schema, schema_responses in responses.items()
        }
        self.invocations = []

    def with_structured_output(self, schema, **kwargs):
        parent = self

        class StructuredRunnable:
            def invoke(self, messages):
                parent.invocations.append(
                    {
                        "schema": schema,
                        "messages": messages,
                        "method": kwargs.get("method"),
                    }
                )
                return parent.responses[schema].pop(0)

        return StructuredRunnable()


def clarification(*, needed, question="", verification=""):
    return ClarificationDecision(
        need_clarification=needed,
        question=question,
        verification=verification,
    )


def test_clarification_schema_requires_all_fields():
    with pytest.raises(ValidationError):
        ClarificationDecision(
            need_clarification=True,
            question="你希望研究哪个地区？",
        )


def test_research_brief_schema_requires_research_brief():
    with pytest.raises(ValidationError):
        ResearchBriefOutput()


def test_unclear_request_returns_question_and_stops_before_brief_node():
    model = StubModel(
        {
            ClarificationDecision: [
                clarification(
                    needed=True,
                    question="你希望研究哪个城市的咖啡店？",
                )
            ],
            ResearchBriefOutput: [
                ResearchBriefOutput(research_brief="不应被使用")
            ],
        }
    )
    graph = create_scope_graph(model)

    result = graph.invoke(
        {"messages": [HumanMessage(content="研究最好的咖啡店")]}
    )

    assert result["messages"][-1].content == "你希望研究哪个城市的咖啡店？"
    assert result.get("research_brief") is None
    assert result["supervisor_messages"] == []
    assert [call["schema"] for call in model.invocations] == [
        ClarificationDecision
    ]


def test_clear_request_generates_brief_and_supervisor_message():
    expected_brief = "研究 2025 年旧金山咖啡店，并按咖啡品质进行比较。"
    model = StubModel(
        {
            ClarificationDecision: [
                clarification(
                    needed=False,
                    verification="信息充分，开始生成研究简报。",
                )
            ],
            ResearchBriefOutput: [
                ResearchBriefOutput(research_brief=expected_brief)
            ],
        }
    )
    graph = create_scope_graph(model)

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="研究 2025 年旧金山最好的咖啡店，以咖啡品质为标准。"
                )
            ]
        }
    )

    assert result["research_brief"] == expected_brief
    assert result["messages"][-1].content == "信息充分，开始生成研究简报。"
    assert result["supervisor_messages"][-1].content == f"{expected_brief}."
    assert [call["schema"] for call in model.invocations] == [
        ClarificationDecision,
        ResearchBriefOutput,
    ]
    assert all(
        call["method"] == "function_calling" for call in model.invocations
    )


def test_second_user_answer_can_complete_the_scope():
    expected_brief = "研究 2025 年旧金山精品咖啡店，以咖啡品质为主要标准。"
    model = StubModel(
        {
            ClarificationDecision: [
                clarification(
                    needed=True,
                    question="你希望研究哪个城市？",
                ),
                clarification(
                    needed=False,
                    verification="信息已经充分。",
                ),
            ],
            ResearchBriefOutput: [
                ResearchBriefOutput(research_brief=expected_brief)
            ],
        }
    )
    graph = create_scope_graph(model)

    first_result = graph.invoke(
        {"messages": [HumanMessage(content="研究精品咖啡店")]}
    )
    second_messages = [
        *first_result["messages"],
        HumanMessage(content="旧金山，研究 2025 年，以咖啡品质为标准。"),
    ]
    second_result = graph.invoke({"messages": second_messages})

    assert first_result.get("research_brief") is None
    assert second_result["research_brief"] == expected_brief
    assert [call["schema"] for call in model.invocations] == [
        ClarificationDecision,
        ClarificationDecision,
        ResearchBriefOutput,
    ]


def test_each_node_receives_the_conversation_inside_its_prompt():
    user_request = "研究 2025 年旧金山咖啡店"
    model = StubModel(
        {
            ClarificationDecision: [
                clarification(needed=False, verification="开始研究。")
            ],
            ResearchBriefOutput: [
                ResearchBriefOutput(research_brief="旧金山咖啡店研究简报")
            ],
        }
    )

    create_scope_graph(model).invoke(
        {"messages": [HumanMessage(content=user_request)]}
    )

    clarification_prompt = model.invocations[0]["messages"][0].content
    brief_prompt = model.invocations[1]["messages"][0].content
    assert user_request in clarification_prompt
    assert user_request in brief_prompt
    assert "Assess whether" in clarification_prompt
    assert "detailed and concrete research question" in brief_prompt
