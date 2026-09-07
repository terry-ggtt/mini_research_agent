"""State contracts for the legacy scope-plus-research workflow."""

import operator
from collections.abc import Sequence
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AppInputState(TypedDict):
    """External input accepted by the legacy application graph."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


class AppState(TypedDict, total=False):
    """State shared by the scope and research subgraphs."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    research_brief: str | None
    compressed_research: str
    raw_notes: Annotated[list[str], operator.add]


class AppOutputState(TypedDict, total=False):
    """Values returned after clarification or completed research."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    research_brief: str | None
    compressed_research: str
    raw_notes: list[str]
