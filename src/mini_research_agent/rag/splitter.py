"""Split source documents into stable retrievable chunks."""

import hashlib
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from mini_research_agent.config.rag import RagSettings
from mini_research_agent.rag.schemas import (
    ChunkMetadata,
    ChunkRecord,
    DocumentRecord,
)


MARKDOWN_HEADERS = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]


def _create_recursive_splitter(
    settings: RagSettings,
) -> RecursiveCharacterTextSplitter:
    """Create the final size-limiting text splitter."""

    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        keep_separator=True,
        separators=[
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            ". ",
            "; ",
            ", ",
            "，",
            " ",
            "",
        ],
    )


def _create_content_hash(content: str) -> str:
    """Create a SHA-256 hash for one chunk."""

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def _create_chunk_id(
    *,
    document_id: str,
    chunk_index: int,
    content_hash: str,
) -> str:
    """Create a stable chunk identifier."""

    identity = (
        f"{document_id}:"
        f"{chunk_index}:"
        f"{content_hash}"
    )

    identity_hash = hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()

    return f"chunk-{identity_hash}"


def _format_markdown_section(
    metadata: dict,
) -> str | None:
    """Build a readable section path from Markdown headers."""

    headers = [
        metadata.get("Header 1"),
        metadata.get("Header 2"),
        metadata.get("Header 3"),
        metadata.get("Header 4"),
    ]

    section_parts = [
        str(header).strip()
        for header in headers
        if header and str(header).strip()
    ]

    if not section_parts:
        return None

    return " > ".join(section_parts)


def _split_markdown_document(
    document: DocumentRecord,
    *,
    recursive_splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    """Split Markdown by headers and then by final chunk size."""

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS,
        strip_headers=False,
        return_each_line=False,
    )

    section_documents = header_splitter.split_text(
        document.content
    )

    if not section_documents:
        section_documents = [
            Document(
                page_content=document.content,
                metadata={},
            )
        ]

    return recursive_splitter.split_documents(
        section_documents
    )


def _split_text_document(
    document: DocumentRecord,
    *,
    recursive_splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    """Split a plain-text document by semantic separators."""

    return recursive_splitter.create_documents(
        texts=[document.content],
        metadatas=[{}],
    )


def _split_source_document(
    document: DocumentRecord,
    *,
    recursive_splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    """Select the correct splitting strategy for a document."""

    if document.document_type == "markdown":
        return _split_markdown_document(
            document,
            recursive_splitter=recursive_splitter,
        )

    if document.document_type == "text":
        return _split_text_document(
            document,
            recursive_splitter=recursive_splitter,
        )

    raise ValueError(
        f"Unsupported document type for splitting: "
        f"{document.document_type}"
    )


def split_document(
    document: DocumentRecord,
    *,
    settings: RagSettings,
) -> list[ChunkRecord]:
    """Split one source document into ChunkRecord objects."""

    recursive_splitter = _create_recursive_splitter(
        settings
    )

    split_documents = _split_source_document(
        document,
        recursive_splitter=recursive_splitter,
    )

    chunks: list[ChunkRecord] = []

    for chunk_index, split in enumerate(split_documents):
        content = split.page_content.strip()

        if not content:
            continue

        content_hash = _create_content_hash(content)

        metadata = ChunkMetadata(
            chunk_id=_create_chunk_id(
                document_id=document.document_id,
                chunk_index=chunk_index,
                content_hash=content_hash,
            ),
            document_id=document.document_id,
            chunk_index=chunk_index,
            title=document.title,
            source=str(document.source),
            section=_format_markdown_section(
                split.metadata
            ),
            page=None,
            content_hash=content_hash,
        )

        chunks.append(
            ChunkRecord(
                content=content,
                metadata=metadata,
            )
        )

    if not chunks:
        raise ValueError(
            f"Document produced no chunks: "
            f"{document.source}"
        )

    return chunks


def split_documents(
    documents: Sequence[DocumentRecord],
    *,
    settings: RagSettings,
) -> list[ChunkRecord]:
    """Split multiple documents while preserving source order."""

    chunks: list[ChunkRecord] = []

    for document in documents:
        chunks.extend(
            split_document(
                document,
                settings=settings,
            )
        )

    return chunks