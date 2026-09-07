import operator
from typing_extensions import TypedDict ,Literal ,Sequence, Annotated
from langchain_core.messages import BaseMessage 
from langgraph.graph import add_messages
from langchain_core.tools import tool
from pydantic import BaseModel, Field
class SupervisorState(TypedDict):
    supervisor_messages:Annotated[Sequence[BaseMessage] , add_messages]

    research_brief: str

    notes: Annotated[list[str], operator.add] 

    raw_notes: Annotated[list[str], operator.add]

    research_iterations: int = 0

    review_count: int
    review_feedback: str
    research_limitations: list[str]
    termination_reason: str


class ResearchGap(BaseModel):
    question: str = Field(description="尚未解决的具体研究问题。")
    reason: str = Field(description="现有证据不足的原因。")
    priority: Literal["critical", "optional"]
    next_research_task: str = Field(description="可执行的补充任务；无法补充时为空。")


class ResearchReview(BaseModel):
    sufficient: bool = Field(description="关键研究问题是否有足够证据支持。")
    covered_questions: list[str] = Field(default_factory=list)
    gaps: list[ResearchGap] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


@tool
class ConductResearch(BaseModel):
    """Tool for delegating a research task to a specialized sub-agent."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )

@tool
class ResearchComplete(BaseModel):
    """Request an evidence review to decide whether research can finish."""
    pass
