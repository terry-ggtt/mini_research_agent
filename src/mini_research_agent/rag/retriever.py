"""Online retrieval from the project knowledge base."""

from dataclasses import dataclass

from langchain_core.vectorstores import VectorStore

from mini_research_agent.config.rag import RagSettings
from mini_research_agent.rag.schemas import (
    ChunkMetadata,
    RetrievalResult,
)


MetadataFilterValue = str | int | float | bool
MetadataFilter = dict[str, MetadataFilterValue]


class RetrievalDataError(RuntimeError):
    """Raised when stored vector metadata is invalid."""


@dataclass(frozen=True, slots=True)
class KnowledgeBaseRetriever:
    """Retrieve structured evidence from a vector store."""

    vector_store: VectorStore
    settings: RagSettings

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant chunks for one query."""

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "Retrieval query cannot be empty."
            )

        result_limit = (
            top_k
            if top_k is not None
            else self.settings.retrieval_top_k
        )

        if result_limit <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        search_arguments = {
            "query": normalized_query,
            "k": result_limit,
        }

        if metadata_filter:
            search_arguments["filter"] = metadata_filter

        matches = (
            self.vector_store
            .similarity_search_with_relevance_scores(
                **search_arguments
            )
        )

        retrieval_results: list[RetrievalResult] = []
        observed_chunk_ids: set[str] = set()

        for document, score in matches:
            relevance_score = float(score)

            if (
                self.settings.score_threshold is not None
                and relevance_score
                < self.settings.score_threshold
            ):
                continue

            try:
                metadata = ChunkMetadata.model_validate(
                    document.metadata
                )
            except ValueError as exc:
                raise RetrievalDataError(
                    "Stored chunk contains invalid metadata."
                ) from exc

            if metadata.chunk_id in observed_chunk_ids:
                continue

            observed_chunk_ids.add(
                metadata.chunk_id
            )

            retrieval_results.append(
                RetrievalResult(
                    content=document.page_content,
                    score=relevance_score,
                    metadata=metadata,
                )
            )

        return retrieval_results

    def _retrieve_document_chunks(
        self,
        *,
        document_id: str,
    ) -> list[RetrievalResult]:
        """Load indexed chunks for one document without semantic search."""

        normalized_document_id = document_id.strip()
        if not normalized_document_id:
            raise ValueError("document_id cannot be empty.")

        getter = getattr(self.vector_store, "get", None)
        if getter is None:
            raise RetrievalDataError(
                "The configured vector store does not support exact metadata retrieval."
            )

        payload = getter(
            where={"document_id": normalized_document_id},
            include=["documents", "metadatas"],
        )
        documents = payload.get("documents") or []
        metadatas = payload.get("metadatas") or []

        if len(documents) != len(metadatas):
            raise RetrievalDataError(
                "Vector-store documents and metadata have different lengths."
            )

        results = []
        observed_chunk_ids: set[str] = set()
        for content, raw_metadata in zip(documents, metadatas):
            if not content or raw_metadata is None:
                continue
            try:
                metadata = ChunkMetadata.model_validate(raw_metadata)
            except ValueError as exc:
                raise RetrievalDataError(
                    "Stored chunk contains invalid metadata."
                ) from exc
            if metadata.chunk_id in observed_chunk_ids:
                continue
            observed_chunk_ids.add(metadata.chunk_id)
            results.append(
                RetrievalResult(
                    content=str(content),
                    score=None,
                    metadata=metadata,
                )
            )

        return sorted(
            results,
            key=lambda result: result.metadata.chunk_index,
        )

    def retrieve_chunk_context(
        self,
        *,
        document_id: str,
        chunk_index: int,
        before: int = 1,
        after: int = 1,
    ) -> list[RetrievalResult]:
        """Retrieve a target chunk and bounded neighboring chunks."""

        if chunk_index < 0:
            raise ValueError("chunk_index cannot be negative.")
        if before < 0 or after < 0:
            raise ValueError("before and after cannot be negative.")

        first_index = max(0, chunk_index - before)
        final_index = chunk_index + after
        return [
            result
            for result in self._retrieve_document_chunks(
                document_id=document_id
            )
            if first_index
            <= result.metadata.chunk_index
            <= final_index
        ]

    def retrieve_section(
        self,
        *,
        document_id: str,
        section: str,
        max_chunks: int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve ordered chunks from one exact document section."""

        normalized_section = section.strip()
        if not normalized_section:
            raise ValueError("section cannot be empty.")
        if max_chunks <= 0:
            raise ValueError("max_chunks must be greater than 0.")

        matching = [
            result
            for result in self._retrieve_document_chunks(
                document_id=document_id
            )
            if result.metadata.section == normalized_section
        ]
        return matching[:max_chunks]


def create_retriever(
    *,
    settings: RagSettings,
    vector_store: VectorStore,
) -> KnowledgeBaseRetriever:
    """Create the project knowledge-base retriever."""

    return KnowledgeBaseRetriever(
        vector_store=vector_store,
        settings=settings,
    )
