import operator 
from typing_extensions import TypedDict,Annotated,List,Sequence
from langchain_core.messages import BaseMessage 
from langgraph.graph import add_messages


class ResearcherState(TypedDict,total=False):
    researcher_messages:Annotated[Sequence[BaseMessage],add_messages]
    tool_call_iterations:int
    research_topic:str
    raw_notes:Annotated[List[str],operator.add]
    compressed_research:str
    file_read_operations:int
    file_characters_read:int
    file_read_ranges:list[dict[str, str | int]]

class ResearcherOutputState(TypedDict):
    compressed_research:str
    raw_notes:Annotated[List[str],operator.add]

