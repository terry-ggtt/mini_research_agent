import asyncio
from langgraph.config import get_stream_writer
from typing_extensions import Literal
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command
from langgraph.graph import StateGraph , START , END
from mini_research_agent.schemas.supervisor import (
    ConductResearch,
    ResearchComplete,
    ResearchReview,
    SupervisorState,
)
from mini_research_agent.prompts.supervisor import (
    lead_researcher_prompt,
    research_review_prompt,
)
from mini_research_agent.tools.thinking import think_tool
from mini_research_agent.utils.dates import get_today_str
from mini_research_agent.config import create_chat_model
from mini_research_agent.graphs.research import create_research_graph
def get_notes_from_tool_calls(
    messages: list[BaseMessage],
) -> list[str]:
    return [
        str(message.content)
        for message in messages
        if isinstance(message, ToolMessage)
        and message.name == "ConductResearch"
    ]


def get_completed_research_tasks(messages: list[BaseMessage]) -> list[str]:
    tasks = []
    for message in messages:
        for tool_call in getattr(message, "tool_calls", []):
            if tool_call["name"] == "ConductResearch":
                tasks.append(str(tool_call.get("args", {}).get("research_topic", "")))
    return [task for task in tasks if task]


def build_review_messages(
    *,
    research_brief: str,
    findings: list[str],
    completed_tasks: list[str],
    previous_feedback: str,
    remaining_iterations: int,
) -> list[BaseMessage]:
    findings_text = "\n\n".join(findings) or "No research findings were returned."
    tasks_text = "\n".join(f"- {task}" for task in completed_tasks) or "- None"
    content = f"""<ResearchBrief>
{research_brief}
</ResearchBrief>

<ResearchFindings>
{findings_text}
</ResearchFindings>

<CompletedResearchTasks>
{tasks_text}
</CompletedResearchTasks>

<PreviousReviewFeedback>
{previous_feedback or "None"}
</PreviousReviewFeedback>

<RemainingSupervisorIterations>
{remaining_iterations}
</RemainingSupervisorIterations>"""
    return [
        SystemMessage(content=research_review_prompt),
        HumanMessage(content=content),
    ]


def format_review_feedback(review: ResearchReview) -> str:
    actionable_gaps = [
        gap for gap in review.gaps
        if gap.priority == "critical" and gap.next_research_task.strip()
    ]
    lines = [
        "The evidence review found critical gaps. Use the remaining budget only "
        "for these targeted follow-up tasks:"
    ]
    for index, gap in enumerate(actionable_gaps, start=1):
        lines.append(
            f"{index}. Gap: {gap.question}\n"
            f"   Reason: {gap.reason}\n"
            f"   Task: {gap.next_research_task.strip()}"
        )
    lines.append(
        "Do not repeat questions already covered. New findings must directly "
        "answer the gaps above and include locatable sources."
    )
    return "\n".join(lines)


def build_limitations(review: ResearchReview) -> list[str]:
    limitations = list(review.limitations)
    limitations.extend(f"Unresolved conflict: {item}" for item in review.conflicts)
    limitations.extend(
        f"{gap.question}: {gap.reason}"
        for gap in review.gaps
    )
    return list(dict.fromkeys(item.strip() for item in limitations if item.strip()))


# Ensure async compatibility for Jupyter environments
try:
    import nest_asyncio
    # Only apply if running in Jupyter/IPython environment
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            nest_asyncio.apply()
    except ImportError:
        pass  # Not in Jupyter, no need for nest_asyncio
except ImportError:
    pass  # nest_asyncio not available, proceed without it

def create_supervisor_graph(
    *,
    model=None,
    research_graph=None,
    review_model=None,
    max_review_rounds=2,
    max_researcher_iterations=6,
    max_concurrent_researchers=3,
):
    supervisor_tool =  [ConductResearch , ResearchComplete , think_tool]
    supervisor_model = (
        model
        if model is not None
        else create_chat_model()
    )
    supervisor_with_tools_model = supervisor_model.bind_tools(supervisor_tool)
    review_base_model = (
        review_model
        if review_model is not None
        else create_chat_model()
    )
    reviewer = review_base_model.with_structured_output(
        ResearchReview,
        method="function_calling",
    )
    researcher = (
        research_graph
        if research_graph is not None
        else create_research_graph()
    )

    async def supervisor(state: SupervisorState)->Command[Literal["supervisor_tools"]]:

        supervisor_messages = state.get("supervisor_messages", [])
        system_message = lead_researcher_prompt.format(
        date=get_today_str(),               
            max_researcher_iterations=max_researcher_iterations,
            max_concurrent_research_units=max_concurrent_researchers,
            )
        feedback = state.get("review_feedback", "")
        if feedback:
            system_message += f"""

<ResearchReviewFeedback>
{feedback}
</ResearchReviewFeedback>"""
        messages = [SystemMessage(content=system_message)] + supervisor_messages

        response = await supervisor_with_tools_model.ainvoke(messages)

        return Command(
            goto="supervisor_tools",
            update={
                "supervisor_messages": [response],
                "research_iterations": state.get("research_iterations", 0) + 1
            }
        )

    async def supervisor_tools(state: SupervisorState)->Command[Literal["supervisor", "review_research"]]:
        """Execute tools requested by state and return the next graph Command.

        Emits task progress. Research failures propagate after cancelling and
        awaiting sibling tasks, so unfinished work cannot leak into later turns.
        """
        emit = get_stream_writer()

        async def run_research_task(tool_call):
            """Run one ConductResearch tool_call and return its research result.

            Emits start/completion/failure events keyed by the tool-call ID.
            Exceptions and cancellation propagate to the supervising batch.
            """
            task_id = str(tool_call["id"])
            topic = str(tool_call["args"]["research_topic"])
            topic_preview = " ".join(topic.split())[:100]

            emit({
                "type": "progress",
                "task_id": task_id,
                "phase": "正在研究资料",
                "message": f"开始研究：{topic_preview}",
            })

            try:
                result = await researcher.ainvoke({
                    "research_topic": topic,
                })
            except asyncio.CancelledError:
                emit({
                    "type": "progress",
                    "task_id": task_id,
                    "message": "研究已取消",
                })
                raise
            except Exception as exc:
                emit({
                    "type": "progress",
                    "task_id": task_id,
                    "message": f"研究失败：{type(exc).__name__}",
                })
                raise

            emit({
                "type": "progress",
                "task_id": task_id,
                "message": "研究完成",
            })
            return result
        supervisor_messages = state.get("supervisor_messages", [])
        research_iterations = state.get("research_iterations", 0)
        most_recent_message = supervisor_messages[-1]

        tool_messages = []
        all_raw_notes = []

        exceeded_iterations = research_iterations >= max_researcher_iterations
        no_tool_calls = not most_recent_message.tool_calls
        research_complete = any(
            tool_call["name"] == "ResearchComplete" 
            for tool_call in most_recent_message.tool_calls
        )
        research_complete_calls = [
            tool_call for tool_call in most_recent_message.tool_calls
            if tool_call["name"] == "ResearchComplete"
        ]

        if not exceeded_iterations:
            # Execute ALL tool calls before deciding next step
            try:
                # Separate think_tool calls from ConductResearch calls
                think_tool_calls = [
                    tool_call for tool_call in most_recent_message.tool_calls 
                    if tool_call["name"] == "think_tool"
                ]

                conduct_research_calls = [
                    tool_call for tool_call in most_recent_message.tool_calls 
                    if tool_call["name"] == "ConductResearch"
                ]

                for tool_call in think_tool_calls:
                    observation = think_tool.invoke(tool_call["args"])
                    tool_messages.append(
                        ToolMessage(
                            content=observation,
                            name =tool_call["name"],
                            tool_call_id=tool_call["id"],

                        )
                    )

 
                if conduct_research_calls:
                    tasks = [
                        asyncio.create_task(run_research_task(tool_call))
                        for tool_call in conduct_research_calls
                    ]
                    try:
                        responses = await asyncio.gather(*tasks)
                    finally:
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)

                    research_tool_messages=[
                        ToolMessage(
                            content = result["compressed_research"],
                            name = tool_call["name"],
                            tool_call_id = tool_call["id"]
                        )for result ,tool_call in zip(responses, conduct_research_calls)
                    ]

                    tool_messages.extend(research_tool_messages)

                    all_raw_notes = [
                        "\n".join(result.get("raw_notes", []))
                        for result in responses
                    ]

            except Exception:
                # A failed researcher must fail the supervisor run. Ending the
                # graph here would let the writer generate a report from empty
                # or incomplete notes and conceal the real failure.
                raise

        tool_messages.extend(
            ToolMessage(
                content=(
                    "Research completion requested. The evidence reviewer "
                    "will decide whether research can finish."
                ),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
            for tool_call in research_complete_calls
        )

        should_review = exceeded_iterations or no_tool_calls or research_complete
        return Command(
            goto="review_research" if should_review else "supervisor",
            update={
                "supervisor_messages": tool_messages,
                "raw_notes": all_raw_notes,
            },
        )

    async def review_research(
        state: SupervisorState,
    ) -> Command[Literal["supervisor", "__end__"]]:
        """Review findings in state and return a follow-up or termination Command.

        Emits review/follow-up progress and preserves reviewer exceptions.
        """
        emit = get_stream_writer()
        emit({
            "type": "progress",
            "phase": "正在检查研究证据",
            "message": "开始检查证据是否充分",
        })
        supervisor_messages = list(state.get("supervisor_messages", []))
        findings = get_notes_from_tool_calls(supervisor_messages)
        research_iterations = state.get("research_iterations", 0)
        remaining_iterations = max(0, max_researcher_iterations - research_iterations)

        review = await reviewer.ainvoke(
            build_review_messages(
                research_brief=state.get("research_brief", ""),
                findings=findings,
                completed_tasks=get_completed_research_tasks(supervisor_messages),
                previous_feedback=state.get("review_feedback", ""),
                remaining_iterations=remaining_iterations,
            )
        )
        review_count = state.get("review_count", 0) + 1
        critical_gaps = [
            gap for gap in review.gaps if gap.priority == "critical"
        ]
        actionable_gaps = [
            gap for gap in critical_gaps if gap.next_research_task.strip()
        ]
        can_continue = (
            bool(actionable_gaps)
            and review_count < max_review_rounds
            and research_iterations < max_researcher_iterations
        )

        if can_continue:
            emit({
                "type": "progress",
                "phase": "正在补充研究",
                "message": "证据存在关键缺口，开始补充研究",
            })
            return Command(
                goto="supervisor",
                update={
                    "review_count": review_count,
                    "review_feedback": format_review_feedback(review),
                },
            )

        if review.sufficient and not critical_gaps:
            termination_reason = "sufficient"
        elif research_iterations >= max_researcher_iterations:
            termination_reason = "budget_exhausted"
        elif review_count >= max_review_rounds and actionable_gaps:
            termination_reason = "review_limit_reached"
        else:
            termination_reason = "unable_to_continue"

        return Command(
            goto=END,
            update={
                "notes": findings,
                "review_count": review_count,
                "research_limitations": build_limitations(review),
                "termination_reason": termination_reason,
            },
        )

    supervisor_bulider = StateGraph(SupervisorState)
    supervisor_bulider.add_node("supervisor", supervisor)
    supervisor_bulider.add_node("supervisor_tools", supervisor_tools)
    supervisor_bulider.add_node("review_research", review_research)
    supervisor_bulider.add_edge(START, "supervisor")
    supervisor_agent = supervisor_bulider.compile()
    return supervisor_agent



