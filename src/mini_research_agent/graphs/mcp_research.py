"""Legacy chapter-three MCP research graph kept for learning compatibility.

Production composition uses ``bootstrap.create_application`` and injects MCP
tools into the unified research graph.
"""

from mini_research_agent.config import create_chat_model
from mini_research_agent.tools.thinking import think_tool
from mini_research_agent.utils.dates import get_today_str
from mini_research_agent.tools.mcp import create_filesystem_mcp_config
from mini_research_agent.prompts.research import research_agent_prompt_with_mcp, compress_research_system_prompt, compress_research_human_message
from mini_research_agent.schemas.research_state import ResearcherOutputState, ResearcherState
import asyncio
from typing_extensions import Literal
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, filter_messages
from langgraph.graph import StateGraph , START,END
mcp_config = create_filesystem_mcp_config()



def create_mcp_research_graph(
    *,
    model=None,
    compress_model=None,
    mcp_client=None,
    max_tool_iterations: int = 5,
):
    _client = mcp_client
    _tools = None
    _model_with_tools = None
    _runtime_lock = asyncio.Lock()
    chat_model = (
        model
        if model is not None
        else create_chat_model()
    )

    compression_model = (
        compress_model
        if compress_model is not None
        else create_chat_model(max_tokens=32000)
    )
    async def get_mcp_runtime():
        nonlocal _client
        nonlocal _tools
        nonlocal _model_with_tools

        if _tools is not None and _model_with_tools is not None:
            return _tools, _model_with_tools
        async with _runtime_lock:
            if _client is None:
                _client = MultiServerMCPClient(mcp_config)
            if _tools is None:
                mcp_tools = await _client.get_tools()
                _tools = [*mcp_tools, think_tool]
            if _model_with_tools is None:
                _model_with_tools = chat_model.bind_tools(tools=_tools)
        
        return _tools, _model_with_tools



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
        }
    async def llm_call(state: ResearcherState) -> dict:
        _,model_with_tools = await get_mcp_runtime()

        messages = [
            SystemMessage(content = research_agent_prompt_with_mcp.format(date = get_today_str())),
            *state["researcher_messages"]
        ]
        iteration_count = state.get(
        "tool_call_iterations",
        0,
    )

        active_model = (
            model_with_tools
            if iteration_count < max_tool_iterations
            else chat_model
        )
        response =await active_model.ainvoke(messages)

        return {
            "researcher_messages": [ response]
        }


    async def tool_node(state: ResearcherState)-> dict:


        tool_calls = state["researcher_messages"][-1].tool_calls

        _tools , _ = await get_mcp_runtime()

        tool_output = []

        tools_by_name = {tool.name: tool for tool in _tools}
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            try:
                tool = tools_by_name[tool_name]
                observation = await tool.ainvoke(tool_call["args"])

            except Exception as e:
                observation = f"Error: {str(e)}"

            tool_output.append(
                ToolMessage(
                    content=str(observation),
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )
        return {
            "researcher_messages": tool_output,
            "tool_call_iterations": (
                state.get("tool_call_iterations", 0) + 1
            ),
        }

    async def compress_research(state:ResearcherState):
        messages=[
            SystemMessage(content = compress_research_system_prompt.format(date = get_today_str())),
            *state["researcher_messages"],
            HumanMessage(content = compress_research_human_message.format(research_topic = state["research_topic"]))
        ]
        response =await compression_model.ainvoke(messages)

        raw_notes = [
            str(m.content) for m in filter_messages(
                state["researcher_messages"], 
                include_types=["tool", "ai"]
            )
        ]

        return {
            "compressed_research": str(response.content),
            "raw_notes": ["\n".join(raw_notes)]
        }

    def should_continue(state:ResearcherState)->Literal["tool_node", "compress_research"]:
        tool_calls = state["researcher_messages"][-1].tool_calls
        if tool_calls:
            return "tool_node"
        return "compress_research"

    agent_mcp_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)
    agent_mcp_builder.add_node("initialize_research", initialize_research)
    agent_mcp_builder.add_node("llm_call", llm_call)
    agent_mcp_builder.add_node("tool_node", tool_node)
    agent_mcp_builder.add_node("compress_research", compress_research)
    agent_mcp_builder.add_edge(START, "initialize_research")
    agent_mcp_builder.add_edge("initialize_research", "llm_call")
    agent_mcp_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        {
            "tool_node": "tool_node",
            "compress_research": "compress_research",
        },
    )
    agent_mcp_builder.add_edge("tool_node", "llm_call")
    agent_mcp_builder.add_edge("compress_research", END)

    agent_mcp = agent_mcp_builder.compile()
    return agent_mcp






