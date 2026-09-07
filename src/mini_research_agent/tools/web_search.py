"""Tavily web-search tool and result-processing helpers."""

import os
from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg, tool
from tavily import TavilyClient
from typing_extensions import Literal

from mini_research_agent.config import create_chat_model
from mini_research_agent.prompts.research import summarize_webpage_prompt
from mini_research_agent.schemas.research_models import Summary
from mini_research_agent.tools.thinking import think_tool
from mini_research_agent.utils.dates import get_today_str


def tavily_search_multiple(
    search_queries: list[str],
    *,
    tavily_client,
    max_results: int = 3,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
) -> list[dict]:
    """Run each query through an injected Tavily client."""

    return [
        tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
        for query in search_queries
    ]


def summarize_webpage_content(webpage_content: str, *, structured_model) -> str:
    """Summarize webpage content, falling back to a bounded raw excerpt."""

    try:
        summary = structured_model.invoke(
            [
                HumanMessage(
                    content=summarize_webpage_prompt.format(
                        webpage_content=webpage_content,
                        date=get_today_str(),
                    )
                )
            ]
        )
        return (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )
    except Exception:
        return (
            webpage_content[:1000] + "..."
            if len(webpage_content) > 1000
            else webpage_content
        )


def deduplicate_search_results(search_results: list[dict]) -> dict[str, dict]:
    """Deduplicate Tavily results by URL while preserving first occurrence."""

    unique_results: dict[str, dict] = {}
    for response in search_results:
        for result in response.get("results", []):
            url = result.get("url")
            if url and url not in unique_results:
                unique_results[url] = result
    return unique_results


def process_search_results(
    unique_results: dict[str, dict],
    *,
    structured_model,
) -> dict[str, dict]:
    """Convert raw Tavily results into compact source records."""

    summarized_results: dict[str, dict] = {}
    for url, result in unique_results.items():
        raw_content = result.get("raw_content")
        content = (
            summarize_webpage_content(
                raw_content,
                structured_model=structured_model,
            )
            if raw_content
            else result.get("content", "")
        )
        summarized_results[url] = {
            "title": result.get("title", "Untitled source"),
            "content": content,
        }
    return summarized_results


def format_search_output(summarized_results: dict[str, dict]) -> str:
    """Format source records for consumption by the research model."""

    if not summarized_results:
        return "No valid search results found. Please try different search queries."

    sections = ["Search results:"]
    for index, (url, result) in enumerate(summarized_results.items(), 1):
        sections.append(
            f"--- SOURCE {index}: {result['title']} ---\n"
            f"URL: {url}\n\n"
            f"SUMMARY:\n{result['content']}\n\n"
            + "-" * 80
        )
    return "\n\n".join(sections)


def create_research_tools(*, tavily_client=None, summarization_model=None):
    """Create the default web-search and reflection tools."""

    search_client = tavily_client or TavilyClient(
        api_key=os.environ["TAVILY_API_KEY"]
    )
    summary_model = summarization_model or create_chat_model(max_tokens=8000)
    structured_model = summary_model.with_structured_output(
        Summary,
        method="function_calling",
    )

    @tool(parse_docstring=True)
    def tavily_search(
        query: str,
        max_results: Annotated[int, InjectedToolArg] = 3,
        topic: Annotated[
            Literal["general", "news", "finance"],
            InjectedToolArg,
        ] = "general",
    ) -> str:
        """Fetch Tavily results and summarize webpage content.

        Args:
            query: A single search query to execute.
            max_results: Maximum number of results to return.
            topic: Tavily search category.
        """

        search_results = tavily_search_multiple(
            [query],
            tavily_client=search_client,
            max_results=max_results,
            topic=topic,
            include_raw_content=True,
        )
        unique_results = deduplicate_search_results(search_results)
        summarized_results = process_search_results(
            unique_results,
            structured_model=structured_model,
        )
        return format_search_output(summarized_results)

    return [tavily_search, think_tool]
