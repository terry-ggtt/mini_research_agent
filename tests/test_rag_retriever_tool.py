from pathlib import Path

from langchain_core.documents import Document

from mini_research_agent.config.rag import RagSettings
from mini_research_agent.rag.retriever import KnowledgeBaseRetriever
from mini_research_agent.rag.schemas import ChunkMetadata, RetrievalResult
from mini_research_agent.rag.tool import (
    NO_RESULTS_MESSAGE,
    create_knowledge_base_tool,
    create_knowledge_base_tools,
)


def make_settings(tmp_path: Path, *, threshold=0.5) -> RagSettings:
    return RagSettings(
        knowledge_base_path=tmp_path / "knowledge",
        vector_store_path=tmp_path / "vectors",
        collection_name="test",
        embedding_provider="huggingface",
        embedding_model="test-model",
        embedding_base_url=None,
        embedding_device="cpu",
        chunk_size=100,
        chunk_overlap=10,
        retrieval_top_k=3,
        score_threshold=threshold,
        supported_extensions=(".md", ".txt"),
        ingestion_batch_size=2,
    )


def metadata(chunk_id: str, chunk_index: int) -> dict:
    return ChunkMetadata(
        chunk_id=chunk_id,
        document_id="doc-1",
        chunk_index=chunk_index,
        title="Policy",
        source="data/knowledge_base/policy.md",
        section="Refunds",
        page=None,
        content_hash=f"hash-{chunk_id}",
    ).model_dump(exclude_none=True)


class SearchStore:
    def __init__(self, matches):
        self.matches = matches
        self.calls = []

    def similarity_search_with_relevance_scores(self, **kwargs):
        self.calls.append(kwargs)
        return self.matches


class ExactStore(SearchStore):
    def __init__(self, documents):
        super().__init__([])
        self.documents = documents
        self.get_calls = []

    def get(self, **kwargs):
        self.get_calls.append(kwargs)
        document_id = kwargs["where"]["document_id"]
        selected = [
            document
            for document in self.documents
            if document.metadata["document_id"] == document_id
        ]
        return {
            "documents": [document.page_content for document in selected],
            "metadatas": [document.metadata for document in selected],
        }


def test_retriever_filters_scores_and_duplicate_chunks(tmp_path):
    store = SearchStore(
        [
            (Document(page_content="relevant", metadata=metadata("chunk-1", 0)), 0.9),
            (Document(page_content="duplicate", metadata=metadata("chunk-1", 0)), 0.8),
            (Document(page_content="weak", metadata=metadata("chunk-2", 1)), 0.2),
        ]
    )
    retriever = KnowledgeBaseRetriever(
        vector_store=store,
        settings=make_settings(tmp_path),
    )

    results = retriever.retrieve(" refund policy ")

    assert [result.content for result in results] == ["relevant"]
    assert store.calls == [{"query": "refund policy", "k": 3}]


class StubRetriever:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return self.results


def test_knowledge_base_tool_formats_traceable_evidence():
    result = RetrievalResult(
        content="Refunds require evidence.",
        score=0.91,
        metadata=ChunkMetadata.model_validate(metadata("chunk-7", 7)),
    )
    retriever = StubRetriever([result])
    knowledge_tool = create_knowledge_base_tool(retriever=retriever)

    output = knowledge_tool.invoke({"query": "refund evidence"})

    assert retriever.queries == ["refund evidence"]
    assert "Citation ID: KB:chunk-7" in output
    assert "Section: Refunds" in output
    assert "Refunds require evidence." in output


def test_knowledge_base_tool_reports_no_results():
    knowledge_tool = create_knowledge_base_tool(
        retriever=StubRetriever([]),
    )

    assert knowledge_tool.invoke({"query": "missing"}) == NO_RESULTS_MESSAGE


def test_retriever_gets_neighboring_chunks_and_exact_section(tmp_path):
    documents = [
        Document(page_content="zero", metadata=metadata("chunk-0", 0)),
        Document(page_content="one", metadata=metadata("chunk-1", 1)),
        Document(page_content="two", metadata=metadata("chunk-2", 2)),
        Document(
            page_content="other section",
            metadata={**metadata("chunk-3", 3), "section": "Exceptions"},
        ),
    ]
    retriever = KnowledgeBaseRetriever(
        vector_store=ExactStore(documents),
        settings=make_settings(tmp_path),
    )

    context = retriever.retrieve_chunk_context(
        document_id="doc-1",
        chunk_index=1,
        before=1,
        after=1,
    )
    section = retriever.retrieve_section(
        document_id="doc-1",
        section="Refunds",
        max_chunks=2,
    )

    assert [result.metadata.chunk_index for result in context] == [0, 1, 2]
    assert [result.metadata.chunk_index for result in section] == [0, 1]


def test_knowledge_base_tool_factory_exposes_context_tools():
    class ContextRetriever(StubRetriever):
        def retrieve_chunk_context(self, **_kwargs):
            return []

        def retrieve_section(self, **_kwargs):
            return []

    tools = create_knowledge_base_tools(
        retriever=ContextRetriever([]),
    )

    assert [tool.name for tool in tools] == [
        "search_knowledge_base",
        "get_knowledge_context",
        "get_knowledge_section",
    ]
