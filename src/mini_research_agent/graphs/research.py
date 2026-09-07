from mini_research_agent.config import create_chat_model
from mini_research_agent.tools.web_search import create_research_tools
from mini_research_agent.utils.dates import get_today_str
from mini_research_agent.schemas.research_state import ResearcherOutputState, ResearcherState
from mini_research_agent.prompts.research import research_agent_prompt, compress_research_system_prompt, compress_research_human_message
from langchain_core.messages import HumanMessage , SystemMessage , ToolMessage , filter_messages
from typing_extensions import Literal
from langgraph.graph import StateGraph ,START , END
def create_research_graph(
    *,
    model=None,
    compress_model=None,
    tools=None,
    max_tool_iterations: int = 5,
    max_file_read_operations: int = 5,
    max_file_characters: int = 60_000,
):
    if max_file_read_operations < 0:
        raise ValueError("max_file_read_operations cannot be negative.")
    if max_file_characters < 0:
        raise ValueError("max_file_characters cannot be negative.")

    def _build_tools_by_name(
        tools,
    ) -> dict:
        tools_by_name = {}

        for tool in tools:
            if tool.name in tools_by_name:
                raise ValueError(
                    f"Duplicate research tool name: "
                    f"{tool.name}"
                )

            tools_by_name[tool.name] = tool

        return tools_by_name
    chat_model = model or create_chat_model()

    compression_model = (
        compress_model
        or create_chat_model(max_tokens=32000)
    )

    research_tools = (
        tools
        if tools is not None
        else create_research_tools()
    )

    tools_by_name = _build_tools_by_name(
        research_tools
    )
    model_with_tools = chat_model.bind_tools(
        research_tools
    )
    

    def initialize_research(
        state: ResearcherState,
        ) -> dict:
        return {
            "researcher_messages": [
                HumanMessage(
                    content=state["research_topic"]
                )
            ],
            "tool_call_iterations": 0,
            "file_read_operations": 0,
            "file_characters_read": 0,
            "file_read_ranges": [],
        }


    async def llm_call(state: ResearcherState) -> dict:
        system_message = research_agent_prompt.format(
            date=get_today_str(),
        )

        iteration_count = state.get(
            "tool_call_iterations",
            0,
        )

        active_model = (
            model_with_tools
            if iteration_count < max_tool_iterations
            else chat_model
        )

        response = await active_model.ainvoke(
            [
                SystemMessage(content=system_message),
                *state["researcher_messages"],
            ]
        )

        return {
            "researcher_messages": [response],
        }


    async def tool_node(
    state: ResearcherState,
) -> dict:
        tool_calls = state[
            "researcher_messages"
        ][-1].tool_calls

        tool_outputs = []
        file_read_operations = state.get("file_read_operations", 0)
        file_characters_read = state.get("file_characters_read", 0)
        file_read_ranges = list(state.get("file_read_ranges", []))

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            is_file_range_read = tool_name == "read_file_range"
            normalized_path = ""
            requested_start = 0
            requested_end = 0

            if is_file_range_read:
                normalized_path = str(tool_args.get("path", "")).replace(
                    "\\", "/"
                ).casefold()
                requested_start = int(tool_args.get("start_line", 0))
                requested_end = int(tool_args.get("end_line", 0))
                already_read = any(
                    str(item["path"]) == normalized_path
                    and int(item["start_line"]) <= requested_start
                    and int(item["end_line"]) >= requested_end
                    for item in file_read_ranges
                )

                if file_read_operations >= max_file_read_operations:
                    observation = (
                        "File read rejected: the per-research read-operation "
                        "budget has been exhausted."
                    )
                    tool_outputs.append(
                        ToolMessage(
                            content=observation,
                            name=tool_name,
                            tool_call_id=tool_call["id"],
                        )
                    )
                    continue
                if file_characters_read >= max_file_characters:
                    observation = (
                        "File read rejected: the per-research character "
                        "budget has been exhausted."
                    )
                    tool_outputs.append(
                        ToolMessage(
                            content=observation,
                            name=tool_name,
                            tool_call_id=tool_call["id"],
                        )
                    )
                    continue
                if already_read:
                    observation = (
                        "File read rejected: this range is already covered "
                        "by previously collected evidence."
                    )
                    tool_outputs.append(
                        ToolMessage(
                            content=observation,
                            name=tool_name,
                            tool_call_id=tool_call["id"],
                        )
                    )
                    continue

            try:
                tool = tools_by_name[tool_name]

                observation = await tool.ainvoke(
                    tool_args
                )

                if is_file_range_read:
                    remaining = max_file_characters - file_characters_read
                    observation = str(observation)
                    if len(observation) > remaining:
                        observation = (
                            observation[:remaining]
                            + "\n[Output truncated by cumulative file-read budget.]"
                        )
                    file_read_operations += 1
                    file_characters_read += min(len(observation), remaining)
                    file_read_ranges.append(
                        {
                            "path": normalized_path,
                            "start_line": requested_start,
                            "end_line": requested_end,
                        }
                    )

            except Exception as exc:
                observation = (
                    f"Tool execution failed: "
                    f"{type(exc).__name__}: {exc}"
                )

            tool_outputs.append(
                ToolMessage(
                    content=str(observation),
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )

        return {
            "researcher_messages": tool_outputs,
            "tool_call_iterations": (
                state.get("tool_call_iterations", 0) + 1
            ),
            "file_read_operations": file_read_operations,
            "file_characters_read": file_characters_read,
            "file_read_ranges": file_read_ranges,
        }
    async def compress_research(
    state: ResearcherState,
) -> dict:
        system_message = (
            compress_research_system_prompt.format(
                date=get_today_str(),
            )
        )

        human_message = (
            compress_research_human_message.format(
                research_topic=state["research_topic"],
            )
        )

        messages = [
            SystemMessage(content=system_message),
            *state["researcher_messages"],
            HumanMessage(content=human_message),
        ]

        response = await compression_model.ainvoke(
            messages
        )

        raw_notes = [
            str(message.content)
            for message in filter_messages(
                state["researcher_messages"],
                include_types=["tool", "ai"],
            )
        ]

        return {
            "compressed_research": str(
                response.content
            ),
            "raw_notes": [
                "\n".join(raw_notes)
            ],
        }
    def should_continue(state:ResearcherState)->Literal["tool_node", "compress_research"]:
            messages = state["researcher_messages"]
            last_message = messages[-1]
            if last_message.tool_calls:
                 return "tool_node"
            return "compress_research"


    agent_builder = StateGraph(
        ResearcherState,
        output_schema=ResearcherOutputState,
    )

    agent_builder.add_node("initialize_research", initialize_research)
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)
    agent_builder.add_node("compress_research", compress_research)

    agent_builder.add_edge(START, "initialize_research")
    agent_builder.add_edge("initialize_research", "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        {
            "tool_node": "tool_node",
            "compress_research": "compress_research"
        }
    )

    agent_builder.add_edge("tool_node", "llm_call")
    agent_builder.add_edge("compress_research", END)
    return agent_builder.compile()


