from mini_research_agent.schemas.full_agent import FullAgentInputState, FullAgentState, FullAgentOutputState
from mini_research_agent.graphs.scope import create_scope_graph
from mini_research_agent.graphs.supervisor import create_supervisor_graph
from mini_research_agent.prompts.writer import final_report_generation_prompt
from mini_research_agent.utils.dates import get_today_str
from mini_research_agent.config import create_chat_model
from langchain_core.messages import  HumanMessage,AIMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import Literal



def create_full_agent(
        *,
        writer_model = None , 
        scope_graph = None , 
        supervisor_graph = None ,
        checkpointer = None ,
):
    
    writer = (
        writer_model
        if writer_model is not None
        else create_chat_model(max_tokens= 32000)
    )

    scope = (
        scope_graph
        if scope_graph is not None
        else create_scope_graph()
    )

    supervisor = (
        supervisor_graph
        if supervisor_graph is not None
        else create_supervisor_graph()
    )

    async def final_report_generation(state:FullAgentState)->dict:
        research_brief = state.get("research_brief" , "")
        notes = state.get("notes" , [])

        if not research_brief:
            raise ValueError(
                "Cannot generate final report without research_brief."
            )
        findings = "\n\n".join(
            str(note)
            for note in notes
        )

        final_report_prompt = final_report_generation_prompt.format(
            research_brief=research_brief,
            findings=findings,
            research_limitations="\n".join(state.get("research_limitations", [])) or "None recorded",
            termination_reason=state.get("termination_reason", "unspecified"),
            date=get_today_str()
        )

        response = await writer.ainvoke([HumanMessage(content=final_report_prompt)])
        final_report = str(response.content)
        return {
            "final_report": final_report, 
            "messages": [
                AIMessage(content = final_report)
            ]
        }
    def route_after_scope(
            state:FullAgentState
    )->Literal["supervisor_subgraph", "__end__"]:
        if state.get("research_brief"):
            return "supervisor_subgraph"
        return END

    builder = StateGraph(
        FullAgentState , 
        input_schema = FullAgentInputState,
        output_schema = FullAgentOutputState,
    )
    builder.add_node("scope_graph", scope)
    builder.add_node("supervisor_subgraph" , supervisor)
    builder.add_node("final_report_generation",final_report_generation)
    builder.add_edge(START, "scope_graph")
    builder.add_conditional_edges(
        "scope_graph",
        route_after_scope,
        {
            "supervisor_subgraph": "supervisor_subgraph",
            END: END,
        },
    )
    builder.add_edge("supervisor_subgraph", "final_report_generation")
    builder.add_edge("final_report_generation", END)
    return builder.compile(
        checkpointer=checkpointer
    )



