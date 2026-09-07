"""Legacy scope-plus-MCP graph kept for chapter compatibility.

Production execution uses ``bootstrap.create_application``.
"""

from langgraph.graph import END, START, StateGraph
from typing_extensions import Literal

from mini_research_agent.graphs.mcp_research import create_mcp_research_graph
from mini_research_agent.graphs.scope import create_scope_graph
from mini_research_agent.schemas.app import AppInputState, AppOutputState, AppState


def create_app_graph(*, scope_graph=None, research_graph=None):
    """Compose the scope and research graphs into one workflow.

    Injecting pre-built subgraphs keeps this orchestration layer testable
    without creating external model or search clients.
    """

    scope = scope_graph if scope_graph is not None else create_scope_graph()
    researcher = (
        research_graph
        if research_graph is not None
        else create_mcp_research_graph()
    )

    async def run_scope(state: AppState) -> dict:
        result = await scope.ainvoke({"messages": state["messages"]})
        return {
            "messages": result["messages"],
            "research_brief": result.get("research_brief"),
        }

    def route_after_scope(
        state: AppState,
    ) -> Literal["run_research", "__end__"]:
        if state.get("research_brief"):
            return "run_research"
        return END

    async def run_research(state: AppState) -> dict:
        research_brief = state.get("research_brief")
        if not research_brief:
            raise ValueError("Research cannot start without a research brief.")

        result = await researcher.ainvoke({"research_topic": research_brief})
        return {
            "compressed_research": result["compressed_research"],
            "raw_notes": result.get("raw_notes", []),
        }

    builder = StateGraph(
        AppState,
        input_schema=AppInputState,
        output_schema=AppOutputState,
    )
    builder.add_node("run_scope", run_scope)
    builder.add_node("run_research", run_research)

    builder.add_edge(START, "run_scope")
    builder.add_conditional_edges(
        "run_scope",
        route_after_scope,
        {
            "run_research": "run_research",
            END: END,
        },
    )
    builder.add_edge("run_research", END)

    return builder.compile()
