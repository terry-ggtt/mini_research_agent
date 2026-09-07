import operator 
from typing_extensions import TypedDict,Annotated,List,Sequence
from langchain_core.messages import BaseMessage 
from langgraph.graph import add_messages


class ResearcherState(TypedDict):
    researcher_messages:Annotated[Sequence[BaseMessage],add_messages]
    tool_call_iterations:int
    research_topic:str
    raw_notes:Annotated[List[str],operator.add]
    compressed_research:str

class ResearcherOutputState(TypedDict):
    commpressed_research:str
    raw_notes:Annotated[List[str],operator.add]
    researcher_messages:Annotated[Sequence[BaseMessage],add_messages]
    
 