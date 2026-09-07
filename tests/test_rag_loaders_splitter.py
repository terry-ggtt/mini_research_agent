from pathlib import Path

import pytest

from mini_research_agent.config.rag import RagSettings
from mini_research_agent.rag.loaders import (
    DocumentLoadError,
    EmptyDocumentError,
    discover_document_paths,
    load_document,
)
from mini_research_agent.rag.splitter import split_document


def make_settings(tmp_path: Path) -> RagSettings:
    return RagSettings(
        knowledge_base_path=tmp_path / "knowledge",
        vector_store_path=tmp_path / "vectors",
        collection_name="test",
        embedding_provider="huggingface",
        embedding_model="test-model",
        embedding_base_url=None,
        embedding_device="cpu",
        chunk_size=80,
        chunk_overlap=10,
        retrieval_top_k=3,
        score_threshold=None,
        supported_extensions=(".md", ".txt"),
        ingestion_batch_size=2,
    )


def test_discover_and_load_document(tmp_path):
    settings = make_settings(tmp_path)
    settings.knowledge_base_path.mkdir()
    source = settings.knowledge_base_path / "Policy.MD"
    source.write_bytes(
        "# Policy\r\n\r\nKeep citations.\r\n".encode("utf-8")
    )
    (settings.knowledge_base_path / "ignored.json").write_text("{}", encoding="utf-8")

    discovered = discover_document_paths(
        settings.knowledge_base_path,
        settings.supported_extensions,
    )
    document = load_document(
        source=source,
        knowledge_base_path=settings.knowledge_base_path,
    )

    assert discovered == [source.resolve()]
    assert document.document_type == "markdown"
    assert document.content == "# Policy\n\nKeep citations."
    assert document.document_id.startswith("doc--")
    assert len(document.content_hash) == 64


def test_load_document_rejects_empty_and_outside_files(tmp_path):
    settings = make_settings(tmp_path)
    settings.knowledge_base_path.mkdir()
    empty = settings.knowledge_base_path / "empty.txt"
    empty.write_text("   \n", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    with pytest.raises(EmptyDocumentError):
        load_document(
            source=empty,
            knowledge_base_path=settings.knowledge_base_path,
        )

    with pytest.raises(DocumentLoadError, match="outside"):
        load_document(
            source=outside,
            knowledge_base_path=settings.knowledge_base_path,
        )


def test_split_markdown_preserves_section_and_stable_metadata(tmp_path):
    settings = make_settings(tmp_path)
    settings.knowledge_base_path.mkdir()
    source = settings.knowledge_base_path / "policy.md"
    source.write_text(
        "# Policy\n\n## Refunds\n\nRefund requests require evidence. " * 4,
        encoding="utf-8",
    )
    document = load_document(
        source=source,
        knowledge_base_path=settings.knowledge_base_path,
    )

    first = split_document(document, settings=settings)
    second = split_document(document, settings=settings)

    assert len(first) > 1
    assert [chunk.metadata.chunk_id for chunk in first] == [
        chunk.metadata.chunk_id for chunk in second
    ]
    assert all(chunk.metadata.document_id == document.document_id for chunk in first)
    assert any(chunk.metadata.section for chunk in first)
