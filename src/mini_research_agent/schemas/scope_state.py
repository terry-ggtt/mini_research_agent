import operator
from typing_extensions import Annotated , TypedDict , Optional,Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages


from mini_research_agent.schemas.scope_models import WebSource

class ScopeInputState(MessagesState):
    """User external input"""
    pass

class ScopeState(MessagesState):
    """
    Main state for the full multi-agent research system.
    
    Extends MessagesState with additional fields for research coordination.
    Note: Some fields are duplicated across different state classes for proper
    state management between subgraphs and the main workflow.
    """

    # Research brief generated from user conversation history
    research_brief: Optional[str]
    # Messages exchanged with the supervisor agent for coordination
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    # Raw unprocessed research notes collected during the research phase
    raw_notes: Annotated[list[str], operator.add] = []
    # Processed and structured notes ready for report generation
    notes: Annotated[list[str], operator.add] = []
    # Final formatted research report
    final_report: str

# class ResearchOutputState(TypedDict , total = False):
#     """Research Graph data returned from the outside world"""
#     compressed_research :str

#     raw_notes: list[str]

#     sources: list[WebSource]
