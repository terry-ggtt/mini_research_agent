import asyncio

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from mini_research_agent.bootstrap import create_application

async def run_cli() -> None:
    checkpointer = InMemorySaver()
    runtime = await create_application(
        checkpointer=checkpointer,
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

        result = await full_agent.ainvoke(
            {
                "messages": [
                    HumanMessage(content=user_text),
                ]
            },
            config=config,
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
    """Run the asynchronous CLI from a synchronous console entry point."""

    asyncio.run(run_cli())


if __name__ == "__main__":
    main()
