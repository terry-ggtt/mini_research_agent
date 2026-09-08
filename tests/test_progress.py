import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from mini_research_agent import cli
from mini_research_agent.graphs.full_agent import create_full_agent
from mini_research_agent.graphs.research import create_research_graph
from mini_research_agent.graphs.supervisor import create_supervisor_graph
from mini_research_agent.schemas.full_agent import FullAgentState
from mini_research_agent.schemas.research_state import ResearcherState
from test_research import StubCompressionModel, StubResearchModel
from test_supervisor import StubReviewModel, StubSupervisorModel, sufficient_review, tool_call


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_fails", [False, True])
async def test_nested_research_progress_reaches_cli(capsys, tool_fails):
    """Use capsys/tool_fails to verify nested events and tool-error recovery."""
    @tool
    async def lookup(query: str) -> str:
        """Return local evidence for query without external services."""
        if tool_fails:
            raise RuntimeError("private tool failure detail")
        return "private raw evidence"

    researcher = create_research_graph(
        model=StubResearchModel(bound_responses=[
            AIMessage(content="", tool_calls=[tool_call("lookup", "lookup-1", query="topic")]),
            AIMessage(content="done"),
        ]),
        compress_model=StubCompressionModel("research notes"),
        tools=[lookup],
    )
    supervisor = create_supervisor_graph(
        model=StubSupervisorModel([
            AIMessage(content="", tool_calls=[tool_call("ConductResearch", "task-a", research_topic="topic")]),
            AIMessage(content="done"),
        ]),
        research_graph=researcher,
        review_model=StubReviewModel([sufficient_review()]),
    )

    async def scope(state):
        """Return a fixed scope update for the supplied full-agent state."""
        return {"research_brief": "topic", "supervisor_messages": [HumanMessage(content="topic")]}

    scope_builder = StateGraph(FullAgentState)
    scope_builder.add_node("scope", scope)
    scope_builder.add_edge(START, "scope")
    scope_builder.add_edge("scope", END)
    graph = create_full_agent(
        scope_graph=scope_builder.compile(), supervisor_graph=supervisor,
        writer_model=StubCompressionModel("final answer"), checkpointer=InMemorySaver(),
    )
    result = await cli.run_with_progress(
        graph, {"messages": [HumanMessage(content="topic")]},
        {"configurable": {"thread_id": "progress-test"}},
    )
    output = capsys.readouterr().out
    assert "任务1：开始研究" in output
    assert "正在调用工具：lookup" in output
    if tool_fails:
        assert "工具调用异常：lookup（RuntimeError）" in output
        assert "工具已返回：lookup" not in output
        assert "private tool failure detail" not in output
    else:
        assert "工具已返回：lookup" in output
    assert "任务1：研究完成" in output
    assert "开始检查证据是否充分" in output
    assert "正在生成最终报告" in output
    assert "private raw evidence" not in output
    assert result["final_report"] == "final answer"


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
async def test_heartbeat_stops_on_every_exit(outcome, capsys):
    """Use outcome and capsys to verify waiting feedback and cleanup on all exits."""
    gate = asyncio.Event()

    async def slow_node(state):
        """Wait on the test gate, then return state or raise the requested failure."""
        await gate.wait()
        if outcome == "failure":
            raise RuntimeError("test failure")
        return state

    builder = StateGraph(FullAgentState)
    builder.add_node("slow", slow_node)
    builder.add_edge(START, "slow")
    builder.add_edge("slow", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    running = asyncio.create_task(cli.run_with_progress(
        graph, {"messages": [HumanMessage(content="topic")]},
        {"configurable": {"thread_id": outcome}}, heartbeat_interval=0.01,
    ))
    try:
        async with asyncio.timeout(2):
            while "没有新进度" not in capsys.readouterr().out:
                if running.done():
                    await running
                await asyncio.sleep(0.01)
        if outcome == "cancel":
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
        else:
            gate.set()
            if outcome == "failure":
                with pytest.raises(RuntimeError, match="test failure"):
                    await running
            else:
                await running
        output = capsys.readouterr().out
        if outcome == "failure":
            assert "执行失败" in output
        if outcome == "cancel":
            assert "任务已取消" in output
        await asyncio.sleep(0.04)
        assert capsys.readouterr().out == ""
    finally:
        gate.set()
        if not running.done():
            running.cancel()
        await asyncio.gather(running, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_first", [False, True])
async def test_parallel_tasks_report_identity_and_clean_up(fail_first):
    """Verify parallel task IDs and sibling cleanup when fail_first is true."""
    both_started = asyncio.Event()
    slow_release = asyncio.Event()
    started = set()
    finished = set()

    async def research(state):
        """Return findings for state; coordinate overlap and record cleanup."""
        topic = state["research_topic"]
        started.add(topic)
        if len(started) == 2:
            both_started.set()
        try:
            await both_started.wait()
            if topic == "A":
                if fail_first:
                    raise RuntimeError("research failed")
                slow_release.set()
            else:
                await slow_release.wait()
            return {"compressed_research": topic, "raw_notes": [topic]}
        finally:
            finished.add(topic)

    builder = StateGraph(ResearcherState)
    builder.add_node("research", research)
    builder.add_edge(START, "research")
    builder.add_edge("research", END)
    graph = create_supervisor_graph(
        model=StubSupervisorModel([
            AIMessage(content="", tool_calls=[
                tool_call("ConductResearch", "task-a", research_topic="A"),
                tool_call("ConductResearch", "task-b", research_topic="B"),
            ]),
            AIMessage(content="done"),
        ]),
        research_graph=builder.compile(),
        review_model=StubReviewModel([sufficient_review()]),
    )
    events = []

    async def consume():
        """Collect actual supervisor custom events; return None or propagate errors."""
        async for event in graph.astream(
            {"research_brief": "topic", "supervisor_messages": [HumanMessage(content="topic")]},
            stream_mode="custom",
        ):
            events.append(event)

    async with asyncio.timeout(3):
        if fail_first:
            with pytest.raises(RuntimeError, match="research failed"):
                await consume()
        else:
            await consume()
    assert started == finished == {"A", "B"}
    starts = [event["task_id"] for event in events if event["message"].startswith("开始研究")]
    assert set(starts) == {"task-a", "task-b"}
    if not fail_first:
        ends = [event["task_id"] for event in events if event["message"] == "研究完成"]
        assert set(ends) == {"task-a", "task-b"}


@pytest.mark.asyncio
async def test_heartbeat_deadline_follows_latest_event(monkeypatch, capsys):
    """Use monkeypatch/capsys and a virtual clock to test the exact idle deadline."""
    clock = [0.0]
    sleeps = asyncio.Queue()
    events = asyncio.Queue()

    async def controlled_sleep(delay):
        """Record requested delay and await a test-controlled wake-up; return None."""
        wake = asyncio.get_running_loop().create_future()
        await sleeps.put((delay, wake))
        await wake

    class EventGraph:
        async def astream(self, inputs, **kwargs):
            """Yield queued custom events for inputs; kwargs hold stream settings."""
            while True:
                event = await events.get()
                yield (), "custom", event

    monkeypatch.setattr(cli, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(cli, "asyncio", SimpleNamespace(
        sleep=controlled_sleep, create_task=asyncio.create_task,
        CancelledError=asyncio.CancelledError,
    ))
    running = asyncio.create_task(cli.run_with_progress(EventGraph(), {}, {}))
    try:
        async with asyncio.timeout(2):
            delay, wake = await sleeps.get()
            assert delay == 10
            clock[0] = 1
            await events.put({"type": "progress", "message": "new progress"})
            while "new progress" not in capsys.readouterr().out:
                await asyncio.sleep(0)
            clock[0] = 10
            wake.set_result(None)
            delay, wake = await sleeps.get()
            assert delay == 1  # A t=1 event must produce an idle notice at t=11.
            clock[0] = 11
            wake.set_result(None)
            await sleeps.get()
            assert "最近 10 秒没有新进度" in capsys.readouterr().out
    finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
