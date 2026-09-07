"""RAG document, chunk, and retrieval-result contracts."""

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    """A source document loaded from the knowledge base."""

    document_id: str = Field(
        min_length=1,
        description="Stable unique identifier for the document.",
    )

    title: str = Field(
        min_length=1,
        description="Title of the document.",
    )

    source: Path = Field(
        description="Original file path of the document.",
    )

    document_type: Literal["markdown", "text"] = Field(
        description="Type of the source document.",
    )

    content: str = Field(
        min_length=1,
        description="Normalized textual content of the document.",
    )

    content_hash: str = Field(
        min_length=1,
        description="Hash of the complete document content.",
    )

    updated_at: datetime = Field(
        description="Last modification time of the source document.",
    )


class ChunkMetadata(BaseModel):
    """Metadata associated with one document chunk."""

    chunk_id: str = Field(
        min_length=1,
        description="Stable unique identifier for the chunk.",
    )

    document_id: str = Field(
        min_length=1,
        description="Identifier of the source document.",
    )

    chunk_index: int = Field(
        ge=0,
        description="Zero-based position of the chunk in the document.",
    )

    title: str = Field(
        min_length=1,
        description="Title of the source document.",
    )

    source: str = Field(
        min_length=1,
        description="Serializable path of the source document.",
    )

    section: str | None = Field(
        default=None,
        description="Section containing the chunk, when available.",
    )

    page: int | None = Field(
        default=None,
        ge=1,
        description="One-based page number, when available.",
    )

    content_hash: str = Field(
        min_length=1,
        description="Hash of this chunk's content.",
    )


class ChunkRecord(BaseModel):
    """A retrievable chunk and its metadata."""

    content: str = Field(
        min_length=1,
        description="Textual content of the chunk.",
    )

    metadata: ChunkMetadata = Field(
        description="Source and identity metadata for the chunk.",
    )


class RetrievalResult(BaseModel):
    """One result returned by the knowledge-base retriever."""

    content: str = Field(
        min_length=1,
        description="Retrieved textual content.",
    )

    score: float | None = Field(
        default=None,
        description="Similarity or relevance score returned by the store.",
    )

    metadata: ChunkMetadata = Field(
        description="Metadata identifying the retrieved source.",
    )


class IndexedDocument(BaseModel):
    """One document recorded in the ingestion manifest."""

    document_id: str
    source: str
    content_hash: str
    chunk_ids: list[str]
    updated_at: datetime


class IngestionManifest(BaseModel):
    """Persistent state describing the current vector index."""

    schema_version: int = 1

    embedding_provider: str
    embedding_model: str

    chunk_size: int
    chunk_overlap: int

    documents: dict[str, IndexedDocument] = Field(
        default_factory=dict
    )


class IngestionFailure(BaseModel):
    """One document that failed during ingestion."""

    source: str
    error_type: str
    message: str


class IngestionReport(BaseModel):
    """Summary of one ingestion run."""

    discovered_documents: int = 0
    added_documents: int = 0
    updated_documents: int = 0
    unchanged_documents: int = 0
    deleted_documents: int = 0
    chunks_written: int = 0

    failures: list[IngestionFailure] = Field(
        default_factory=list
    )