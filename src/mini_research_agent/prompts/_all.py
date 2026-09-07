"""Compatibility exports for the former single prompt module."""

from .evaluation import BRIEF_CRITERIA_PROMPT, BRIEF_HALLUCINATION_PROMPT
from .research import (
    compress_research_human_message,
    compress_research_system_prompt,
    research_agent_prompt,
    research_agent_prompt_with_mcp,
    summarize_webpage_prompt,
)
from .scope import clarify_with_user_instructions, transform_messages_into_research_topic_prompt
from .supervisor import lead_researcher_prompt
from .writer import final_report_generation_prompt

__all__ = [
    "BRIEF_CRITERIA_PROMPT",
    "BRIEF_HALLUCINATION_PROMPT",
    "clarify_with_user_instructions",
    "compress_research_human_message",
    "compress_research_system_prompt",
    "final_report_generation_prompt",
    "lead_researcher_prompt",
    "research_agent_prompt",
    "research_agent_prompt_with_mcp",
    "summarize_webpage_prompt",
    "transform_messages_into_research_topic_prompt",
]
