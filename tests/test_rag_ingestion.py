from dataclasses import replace
from pathlib import Path

from mini_research_agent.config.rag import RagSettings
from mini_research_agent.rag.ingestion import (
    _manifest_matches_settings,
    ingest_knowledge_base,
)
from mini_research_agent.rag.schemas import IngestionManifest
from mini_research_agent.scripts.ingest import run_ingestion


class MemoryVectorStore:
    def __init__(self):
        self.documents = {}
        self.delete_calls = []
        self.fail_add = False

    def add_documents(self, documents, ids):
        if self.fail_add:
            raise RuntimeError("simulated vector write failure")
        for document, chunk_id in zip(documents, ids):
            self.documents[chunk_id] = document

    def delete(self, ids):
        self.delete_calls.append(list(ids))
        for chunk_id in ids:
            self.documents.pop(chunk_id, None)


def make_settings(tmp_path: Path) -> RagSettings:
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    return RagSettings(
        knowledge_base_path=knowledge,
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


def test_ingestion_is_incremental_and_removes_deleted_documents(tmp_path):
    settings = make_settings(tmp_path)
    store = MemoryVectorStore()
    source = settings.knowledge_base_path / "notes.txt"
    source.write_text("first version of the research notes", encoding="utf-8")

    first = ingest_knowledge_base(settings=settings, vector_store=store)
    first_ids = set(store.documents)
    second = ingest_knowledge_base(settings=settings, vector_store=store)

    source.write_text(
        "second version of the research notes with changed evidence",
        encoding="utf-8",
    )
    third = ingest_knowledge_base(settings=settings, vector_store=store)
    third_ids = set(store.documents)

    source.unlink()
    fourth = ingest_knowledge_base(settings=settings, vector_store=store)

    assert first.added_documents == 1
    assert first.chunks_written >= 1
    assert second.unchanged_documents == 1
    assert third.updated_documents == 1
    assert first_ids.isdisjoint(third_ids)
    assert fourth.deleted_documents == 1
    assert store.documents == {}
    assert settings.manifest_path.exists()


def test_manifest_schema_version_must_match(tmp_path):
    settings = make_settings(tmp_path)
    manifest = IngestionManifest(
        schema_version=0,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    assert not _manifest_matches_settings(manifest, settings)


def test_rag_settings_reject_invalid_score_threshold(tmp_path):
    settings = make_settings(tmp_path)
    invalid = replace(
        settings,
        score_threshold=1.1,
    )

    try:
        invalid.validate()
    except ValueError as exc:
        assert "RAG_SCORE_THRESHOLD" in str(exc)
    else:
        raise AssertionError("Invalid score threshold was accepted.")


def test_failed_update_preserves_previous_vectors_and_manifest(tmp_path):
    settings = make_settings(tmp_path)
    store = MemoryVectorStore()
    source = settings.knowledge_base_path / "notes.txt"
    source.write_text("stable original evidence", encoding="utf-8")
    ingest_knowledge_base(settings=settings, vector_store=store)
    previous_ids = set(store.documents)
    previous_manifest = settings.manifest_path.read_text(encoding="utf-8")

    source.write_text("replacement evidence that cannot be indexed", encoding="utf-8")
    store.fail_add = True
    report = ingest_knowledge_base(settings=settings, vector_store=store)

    assert len(report.failures) == 1
    assert set(store.documents) == previous_ids
    assert settings.manifest_path.read_text(encoding="utf-8") == previous_manifest


def test_ingestion_entrypoint_accepts_prebuilt_vector_store(tmp_path):
    settings = make_settings(tmp_path)
    store = MemoryVectorStore()
    (settings.knowledge_base_path / "notes.txt").write_text(
        "entrypoint evidence",
        encoding="utf-8",
    )

    report = run_ingestion(
        settings=settings,
        vector_store=store,
    )

    assert report.added_documents == 1
    assert store.documents
