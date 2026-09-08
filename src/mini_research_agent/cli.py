import asyncio
import time
from contextlib import suppress

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from mini_research_agent.bootstrap import create_application


async def run_with_progress(graph, inputs, config, *, heartbeat_interval=10.0):
    """Run a checkpointed graph and print progress, returning its final state.

    Args:
        graph: Compiled graph supporting astream and aget_state.
        inputs: Input state for this conversation turn.
        config: Runnable configuration including the checkpoint thread ID.
        heartbeat_interval: Positive seconds between idle progress checks.

    Raises:
        ValueError: If heartbeat_interval is not positive.
        Exception: Graph failures are reported and re-raised. Cancellation
            also propagates; the background heartbeat is always stopped.
    """
    if heartbeat_interval <= 0:
        raise ValueError("heartbeat_interval must be positive")
    started = time.monotonic()
    last_event = started
    phase = "正在分析研究需求"
    task_labels = {}

    def log(message):
        """Print message with elapsed time and reset the idle timer; return None."""
        nonlocal last_event
        last_event = time.monotonic()
        elapsed = int(last_event - started)
        print(f"[{elapsed:>3}秒] {message}", flush=True)

    async def heartbeat():
        """Print idle waiting status until cancelled; return no value."""
        last_notice = started
        while True:
            deadline = max(last_event, last_notice) + heartbeat_interval
            await asyncio.sleep(max(0.0, deadline - time.monotonic()))
            now = time.monotonic()

            # 最近有新事件时，不额外打印等待提示。
            if now - max(last_event, last_notice) >= heartbeat_interval:
                elapsed = int(now - started)
                quiet = int(now - last_event)
                print(
                    f"[{elapsed:>3}秒] {phase}"
                    f"；最近 {quiet} 秒没有新进度",
                    flush=True,
                )
                last_notice = now

    log(phase)
    heartbeat_task = asyncio.create_task(heartbeat())

    try:
        async for namespace, mode, data in graph.astream(
            inputs,
            config=config,
            stream_mode=["updates", "custom"],
            subgraphs=True,
            version="v1",
        ):
            if mode == "custom":
                if not isinstance(data, dict):
                    continue
                if data.get("type") != "progress":
                    continue

                message = str(data.get("message", ""))
                task_id = data.get("task_id")

                if task_id:
                    label = task_labels.setdefault(
                        task_id, f"任务{len(task_labels) + 1}"
                    )
                    message = f"{label}：{message}"

                if data.get("phase"):
                    phase = str(data["phase"])

                log(message)

            elif mode == "updates" and not namespace:
                # 只用顶层节点更新整体阶段，避免子图重复输出。
                if "scope_graph" in data:
                    scope_result = data["scope_graph"] or {}
                    if scope_result.get("research_brief"):
                        phase = "正在研究资料"
                        log("研究范围已确定，开始研究")

                if "supervisor_subgraph" in data:
                    phase = "正在生成最终报告"
                    log(phase)

        # astream 返回事件；通过现有 checkpointer 读取最终状态。
        snapshot = await graph.aget_state(config)
        return snapshot.values

    except asyncio.CancelledError:
        log("任务已取消")
        raise
    except Exception as exc:
        log(f"执行失败：{type(exc).__name__}: {exc}")
        raise
    finally:
        heartbeat_task.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat_task


async def run_cli() -> None:
    """Initialize the runtime and print progress/results for each input turn.

    Returns None on empty input or a final report. Reuses the checkpoint
    thread for clarification replies and propagates execution failures.
    """
    print("正在初始化研究环境……", flush=True)

    runtime = await create_application(
        checkpointer=InMemorySaver(),
    )
    full_agent = runtime.graph

    config = {
        "configurable": {
            "thread_id": "research-session-1",
        }
    }

    while True:
        user_text = input("研究需求或补充信息：").strip()

        if not user_text:
            print("未提供有效内容，程序结束。")
            return

        result = await run_with_progress(
            full_agent,
            {"messages": [HumanMessage(content=user_text)]},
            config,
        )

        final_report = result.get("final_report")

        if final_report:
            print("\n最终报告：")
            print(final_report)
            return

        messages = result.get("messages", [])

        if not messages:
            raise RuntimeError(
                "Full Agent did not return a clarification "
                "message or final report."
            )

        print("\n需要补充信息：")
        print(messages[-1].content)


def main() -> None:
    """Run the console session; report Ctrl+C and return None."""
    try:
        asyncio.run(run_cli())
    except KeyboardInterrupt:
        print("\n已停止运行。")


if __name__ == "__main__":
    main()
