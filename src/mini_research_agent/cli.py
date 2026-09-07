from langchain_core.messages import HumanMessage

from mini_research_agent.scope_graph import create_scope_graph


def main() -> None:
    """Run the Scope Graph in the terminal."""

    graph = create_scope_graph()
    messages = []

    while True:
        if not messages:
            user_text = input("研究需求：").strip()
        else:
            user_text = input("补充信息：").strip()

        if not user_text:
            print("未提供有效内容，程序结束。")
            return

        messages.append(HumanMessage(content=user_text))

        result = graph.invoke({
            "messages": messages,
        })

        research_brief = result.get("research_brief")

        if research_brief:
            print("\n研究简报：")
            print(research_brief)
            return

        messages = result["messages"]
        clarification_question = messages[-1].content

        print("\n需要澄清：")
        print(clarification_question)


if __name__ == "__main__":
    main()