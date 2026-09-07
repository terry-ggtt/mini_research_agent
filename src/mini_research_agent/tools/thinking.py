"""Strategic reflection tool shared by researcher and supervisor agents."""

from langchain_core.tools import tool


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Record a strategic reflection about research progress.

    Use this after gathering evidence to assess findings, remaining gaps,
    evidence quality, and the next research action.

    Args:
        reflection: Analysis of findings, gaps, quality, and next steps.

    Returns:
        Confirmation that the reflection was recorded.
    """

    return f"Reflection recorded: {reflection}"
