"""Expose knowledge-base retrieval as an agent tool."""

from collections.abc import Sequence

from langchain_core.tools import BaseTool, tool

from mini_research_agent.rag.retriever import (
    KnowledgeBaseRetriever,
)
from mini_research_agent.rag.schemas import (
    RetrievalResult,
)


NO_RESULTS_MESSAGE = (
    "No relevant evidence was found in the local "
    "knowledge base for this query."
)


def _format_score(score: float | None) -> str:
    """Format an optional relevance score."""

    if score is None:
        return "unknown"

    return f"{score:.4f}"


def _format_retrieval_result(
    result: RetrievalResult,
    *,
    source_number: int,
) -> str:
    """Format one retrieval result for the research agent."""

    metadata = result.metadata

    source_lines = [
        f"--- KNOWLEDGE BASE SOURCE {source_number} ---",
        f"Citation ID: KB:{metadata.chunk_id}",
        f"Title: {metadata.title}",
        f"Source: {metadata.source}",
        f"Document ID: {metadata.document_id}",
        f"Chunk ID: {metadata.chunk_id}",
        f"Chunk Index: {metadata.chunk_index}",
        f"Relevance Score: {_format_score(result.score)}",
    ]

    if metadata.section:
        source_lines.append(
            f"Section: {metadata.section}"
        )

    if metadata.page is not None:
        source_lines.append(
            f"Page: {metadata.page}"
        )

    source_lines.extend(
        [
            "",
            "CONTENT:",
            result.content,
        ]
    )

    return "\n".join(source_lines)


def format_retrieval_results(
    results: Sequence[RetrievalResult],
) -> str:
    """Format retrieval results into one tool response."""

    if not results:
        return NO_RESULTS_MESSAGE

    formatted_sources = [
        _format_retrieval_result(
            result,
            source_number=index,
        )
        for index, result in enumerate(
            results,
            start=1,
        )
    ]

    return (
        "Knowledge base search results:\n\n"
        + "\n\n".join(formatted_sources)
    )


def create_knowledge_base_tool(
    *,
    retriever: KnowledgeBaseRetriever,
) -> BaseTool:
    """Create a retrieval tool bound to one knowledge base."""

    @tool(
        "search_knowledge_base",
        parse_docstring=True,
    )
    def search_knowledge_base(
        query: str,
    ) -> str:
        """Search the local knowledge base for relevant evidence.

        Use this tool when the research topic may be answered by
        internal documents, local files, project documentation,
        policies, reports, notes, or other indexed knowledge.
        Use web search instead when current public internet
        information is required.

        Args:
            query: A focused semantic search query describing the
                evidence needed from the local knowledge base.

        Returns:
            Relevant document excerpts with stable citation IDs,
            source paths, sections, pages, and relevance scores.
        """

        results = retriever.retrieve(
            query
        )

        return format_retrieval_results(
            results
        )

    return search_knowledge_base


def create_knowledge_base_tools(
    *,
    retriever: KnowledgeBaseRetriever,
) -> list[BaseTool]:
    """Create semantic search and bounded context-expansion tools."""

    search_knowledge_base = create_knowledge_base_tool(
        retriever=retriever
    )

    @tool("get_knowledge_context", parse_docstring=True)
    def get_knowledge_context(
        document_id: str,
        chunk_index: int,
        before: int = 1,
        after: int = 1,
    ) -> str:
        """Retrieve a known knowledge-base chunk and nearby chunks.

        Args:
            document_id: Exact document identifier returned by semantic search.
            chunk_index: Exact zero-based chunk index returned by semantic search.
            before: Number of preceding chunks to include.
            after: Number of following chunks to include.
        """

        results = retriever.retrieve_chunk_context(
            document_id=document_id,
            chunk_index=chunk_index,
            before=before,
            after=after,
        )
        return format_retrieval_results(results)

    @tool("get_knowledge_section", parse_docstring=True)
    def get_knowledge_section(
        document_id: str,
        section: str,
        max_chunks: int = 5,
    ) -> str:
        """Retrieve ordered chunks from one known document section.

        Args:
            document_id: Exact document identifier returned by semantic search.
            section: Exact section path returned by semantic search.
            max_chunks: Maximum number of ordered chunks to return.
        """

        results = retriever.retrieve_section(
            document_id=document_id,
            section=section,
            max_chunks=max_chunks,
        )
        return format_retrieval_results(results)

    return [
        search_knowledge_base,
        get_knowledge_context,
        get_knowledge_section,
    ]
