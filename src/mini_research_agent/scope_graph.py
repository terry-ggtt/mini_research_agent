from datetime import datetime   
from typing_extensions import Literal
from langchain_core.messages import HumanMessage , AIMessage , get_buffer_string
from langgraph.graph import StateGraph  , START , END
from langgraph.types import Command
from mini_research_agent.config import create_chat_model
from mini_research_agent.prompts import clarify_with_user_instructions,transform_messages_into_research_topic_prompt
from mini_research_agent.src.mini_research_agent.scope_state import ScopeInputState , ScopeState
from mini_research_agent.src.mini_research_agent.scope_schemas import ClarificationDecision , ResearchBriefOutput

#=======UTILITY  FUNCTIONS ================

def get_today_str()->str:
    today = datetime.now()
    return f"{today:%a %b} {today.day},{today:%Y}"
def create_scope_graph(model=None):
    chat_model = model or create_chat_model()

    clarification_model = chat_model.with_structured_output(
    ClarificationDecision,
    method="function_calling",
    )

    brief_model = chat_model.with_structured_output(
    ResearchBriefOutput,
    method="function_calling",
    )
    def clarify_with_user(state:ScopeState) ->Command[Literal["write_research_brief" , "__end__"]] :
        

        response = clarification_model.invoke([
            HumanMessage(content = clarify_with_user_instructions.format(
                messages = get_buffer_string(messages = state["messages"]),
                date = get_today_str()
            ))
        ])
        if response.need_clarification:
            return Command(
                goto = END,
                update={"messages":[AIMessage(content = response.question)]}
            )
        else:
            return Command(
                goto="write_research_brief",
                update={
                        "messages":[AIMessage(content = response.verification)]
                }

            )

    def write_research_brief(state:ScopeState):
        
        response = brief_model.invoke([
            HumanMessage(content= transform_messages_into_research_topic_prompt.format(
                messages=get_buffer_string(state.get("messages",[])), 
                date=get_today_str()
        
            )
        )]

        )
        return {
                "research_brief": response.research_brief,
                "supervisor_messages": [HumanMessage(content=f"{response.research_brief}.")]
            }
    deep_researcher_builder = StateGraph(ScopeState , input_schema=ScopeInputState)

    deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
    deep_researcher_builder.add_node("write_research_brief", write_research_brief)

    # Add workflow edges
    deep_researcher_builder.add_edge(START, "clarify_with_user")
    deep_researcher_builder.add_edge("write_research_brief", END)

    # Compile the workflow
    return deep_researcher_builder.compile()

