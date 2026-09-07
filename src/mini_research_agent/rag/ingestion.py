"""The offline, idempotent RAG ingestion pipeline belongs here."""

from collections.abc import Iterator, Sequence

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from mini_research_agent.config.rag import RagSettings
from mini_research_agent.rag.loaders import (
    discover_document_paths,
    load_document,
)
from mini_research_agent.rag.schemas import (
    ChunkRecord,
    IndexedDocument,
    IngestionFailure,
    IngestionManifest,
    IngestionReport,
)
from mini_research_agent.rag.splitter import split_document


def _create_empty_manifest(
    settings: RagSettings,
) -> IngestionManifest:
    """Create a manifest matching the current index configuration."""

    return IngestionManifest(
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def _load_manifest(
    settings: RagSettings,
) -> IngestionManifest:
    """Load the existing manifest or create an empty one."""

    manifest_path = settings.manifest_path

    if not manifest_path.exists():
        return _create_empty_manifest(settings)

    try:
        manifest_content = manifest_path.read_text(
            encoding="utf-8"
        )

        return IngestionManifest.model_validate_json(
            manifest_content
        )
    except OSError as exc:
        raise RuntimeError(
            f"Unable to read ingestion manifest: "
            f"{manifest_path}"
        ) from exc
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid ingestion manifest: "
            f"{manifest_path}"
        ) from exc


def _save_manifest(
    manifest: IngestionManifest,
    settings: RagSettings,
) -> None:
    """Write the manifest using an atomic file replacement."""

    manifest_path = settings.manifest_path

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = manifest_path.with_suffix(
        ".json.tmp"
    )

    try:
        temporary_path.write_text(
            manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )

        temporary_path.replace(manifest_path)
    except OSError as exc:
        raise RuntimeError(
            f"Unable to write ingestion manifest: "
            f"{manifest_path}"
        ) from exc


def _manifest_matches_settings(
    manifest: IngestionManifest,
    settings: RagSettings,
) -> bool:
    """Check whether existing vectors use the current configuration."""

    return (
        manifest.schema_version == 1
        and manifest.embedding_provider
        == settings.embedding_provider
        and manifest.embedding_model
        == settings.embedding_model
        and manifest.chunk_size
        == settings.chunk_size
        and manifest.chunk_overlap
        == settings.chunk_overlap
    )


def _chunk_to_document(
    chunk: ChunkRecord,
) -> Document:
    """Convert one domain ChunkRecord into a LangChain Document."""

    metadata = chunk.metadata.model_dump(
        exclude_none=True
    )

    return Document(
        page_content=chunk.content,
        metadata=metadata,
    )


def _batched(
    values: Sequence[ChunkRecord],
    batch_size: int,
) -> Iterator[Sequence[ChunkRecord]]:
    """Yield fixed-size batches without copying the full collection."""

    for start in range(0, len(values), batch_size):
        yield values[start:start + batch_size]


def _delete_chunk_ids(
    vector_store: VectorStore,
    chunk_ids: Sequence[str],
) -> None:
    """Delete chunks only when the ID collection is non-empty."""

    if not chunk_ids:
        return

    vector_store.delete(
        ids=list(chunk_ids)
    )


def _write_chunks(
    vector_store: VectorStore,
    chunks: Sequence[ChunkRecord],
    *,
    batch_size: int,
) -> None:
    """Write chunks to the vector store in bounded batches."""

    for batch in _batched(chunks, batch_size):
        documents = [
            _chunk_to_document(chunk)
            for chunk in batch
        ]

        chunk_ids = [
            chunk.metadata.chunk_id
            for chunk in batch
        ]

        vector_store.add_documents(
            documents=documents,
            ids=chunk_ids,
        )


def _clear_manifest_index(
    vector_store: VectorStore,
    manifest: IngestionManifest,
) -> None:
    """Delete all chunks referenced by an incompatible manifest."""

    chunk_ids = [
        chunk_id
        for indexed_document in manifest.documents.values()
        for chunk_id in indexed_document.chunk_ids
    ]

    _delete_chunk_ids(
        vector_store,
        chunk_ids,
    )


def ingest_knowledge_base(
    *,
    settings: RagSettings,
    vector_store: VectorStore,
) -> IngestionReport:
    """Synchronize local knowledge-base files with the vector store."""

    settings.validate()

    document_paths = discover_document_paths(
        settings.knowledge_base_path,
        settings.supported_extensions,
    )

    report = IngestionReport(
        discovered_documents=len(document_paths)
    )

    manifest = _load_manifest(settings)

    if not _manifest_matches_settings(
        manifest,
        settings,
    ):
        _clear_manifest_index(
            vector_store,
            manifest,
        )

        manifest = _create_empty_manifest(settings)

    discovered_sources = {
        str(path.resolve())
        for path in document_paths
    }

    deleted_document_ids = [
        document_id
        for document_id, indexed_document
        in manifest.documents.items()
        if indexed_document.source
        not in discovered_sources
    ]

    for document_id in deleted_document_ids:
        indexed_document = manifest.documents[
            document_id
        ]

        _delete_chunk_ids(
            vector_store,
            indexed_document.chunk_ids,
        )

        del manifest.documents[document_id]
        report.deleted_documents += 1

    for source in document_paths:
        try:
            document = load_document(
                source=source,
                knowledge_base_path=(
                    settings.knowledge_base_path
                ),
            )

            previous = manifest.documents.get(
                document.document_id
            )

            if (
                previous is not None
                and previous.content_hash
                == document.content_hash
            ):
                report.unchanged_documents += 1
                continue

            chunks = split_document(
                document,
                settings=settings,
            )

            new_chunk_ids = [
                chunk.metadata.chunk_id
                for chunk in chunks
            ]
            previous_chunk_ids = set(
                previous.chunk_ids
                if previous is not None
                else []
            )
            new_chunk_id_set = set(new_chunk_ids)
            new_only_chunk_ids = list(
                new_chunk_id_set - previous_chunk_ids
            )
            obsolete_chunk_ids = list(
                previous_chunk_ids - new_chunk_id_set
            )

            try:
                _write_chunks(
                    vector_store,
                    chunks,
                    batch_size=(
                        settings.ingestion_batch_size
                    ),
                )
            except Exception:
                _delete_chunk_ids(
                    vector_store,
                    new_only_chunk_ids,
                )
                raise

            try:
                _delete_chunk_ids(
                    vector_store,
                    obsolete_chunk_ids,
                )
            except Exception:
                _delete_chunk_ids(
                    vector_store,
                    new_only_chunk_ids,
                )
                raise

            manifest.documents[
                document.document_id
            ] = IndexedDocument(
                document_id=document.document_id,
                source=str(document.source),
                content_hash=document.content_hash,
                chunk_ids=new_chunk_ids,
                updated_at=document.updated_at,
            )

            report.chunks_written += len(chunks)

            if previous is None:
                report.added_documents += 1
            else:
                report.updated_documents += 1

        except Exception as exc:
            report.failures.append(
                IngestionFailure(
                    source=str(source),
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )

    _save_manifest(
        manifest,
        settings,
    )

    return report
