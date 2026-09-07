"""RAG source-document loading belongs here."""
import hashlib
from pathlib import Path
from datetime import datetime , timezone

from mini_research_agent.rag.schemas import DocumentRecord

DOCUMENT_TYPE_BY_EXTENSION = {
    ".md": "markdown",
    ".txt": "text",
}

class DocumentLoadError(Exception):
    """Base exception raised when a source document cannot be loaded."""


class UnsupportedDocumentTypeError(DocumentLoadError):
    """Raised when the document extension is not supported."""


class EmptyDocumentError(DocumentLoadError):
    """Raised when the document contains no usable text."""


def _normalize_document_content(content: str)-> str:
     """Normalize line endings and remove surrounding whitespace."""

     return (
         content.replace("\r\n", "\n")
         .replace("\r", "\n")
         .strip()
     )
def _create_content_hash(content: str) -> str:
    """Create a SHA-256 hash from normalized document content."""

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

def _create_document_id(
        source: Path,
        knowledge_base_path: Path,
) -> str:
    """Create a unique identifier for a document."""
    relative_source = source.relative_to(knowledge_base_path)

    normalized_source = (
        relative_source.as_posix().lower()
        
    )

    source_hash = hashlib.sha256(normalized_source.encode("utf-8")).hexdigest()
    return f"doc--{source_hash}"

def discover_document_paths(
    knowledge_base_path: Path,
    supported_extensions: tuple[str, ...],
) -> list[Path]:
    """Discover supported source files inside the knowledge base."""

    root = knowledge_base_path.resolve()

    if not root.exists():
        raise FileNotFoundError(
            f"Knowledge base directory does not exist: {root}"
        )

    if not root.is_dir():
        raise NotADirectoryError(
            f"Knowledge base path is not a directory: {root}"
        )

    normalized_extensions = {
        extension.lower()
        for extension in supported_extensions
    }

    document_paths = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if path.is_symlink():
            continue

        if path.suffix.lower() not in normalized_extensions:
            continue

        if path.suffix.lower() not in DOCUMENT_TYPE_BY_EXTENSION:
            continue

        document_paths.append(path.resolve())

    return sorted(
        document_paths,
        key=lambda path: path.as_posix().lower(),
    )

def load_document(
    source:Path,
    knowledge_base_path:Path,
) -> DocumentRecord:
    """Load one supported source file into a DocumentRecord."""

    root = knowledge_base_path.resolve()
    resolved_source = source.resolve()

    if not resolved_source.is_relative_to(root):
        raise DocumentLoadError(
            f"Document is outside the knowledge base: "
            f"{resolved_source}"
        )

    if not resolved_source.exists():
        raise FileNotFoundError(
            f"Document does not exist: {resolved_source}"
        )

    if not resolved_source.is_file():
        raise DocumentLoadError(
            f"Document path is not a file: {resolved_source}"
        )
    extension = resolved_source.suffix.lower()

    document_type = DOCUMENT_TYPE_BY_EXTENSION.get(
        extension    
        )
    if document_type is None:
        raise UnsupportedDocumentTypeError(
            f"Unsupported document type: {extension}"
        )

    try:
        raw_content = resolved_source.read_text(encoding="utf-8-sig")

    except UnicodeDecodeError as exc:
        raise DocumentLoadError(
            f"Document is not valid UTF-8: {resolved_source}"
        ) from exc
    except OSError as exc:
        raise DocumentLoadError(
            f"Unable to read document: {resolved_source}"
        ) from exc

    content = _normalize_document_content(raw_content)
    if not content:
        raise EmptyDocumentError(
            f"Document contains no usable text: "
            f"{resolved_source}"
        )

    file_stat = resolved_source.stat()

    updated_at = datetime.fromtimestamp(file_stat.st_mtime , tz = timezone.utc)

    return DocumentRecord(
        document_id = _create_document_id(resolved_source , root),
        title = resolved_source.stem,
        source = resolved_source,
        document_type = document_type,
        content = content,
        content_hash = _create_content_hash(content),
        updated_at = updated_at,
    )

    


