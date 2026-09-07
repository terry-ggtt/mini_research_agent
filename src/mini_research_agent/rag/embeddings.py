"""Create embedding models for the RAG pipeline."""

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from mini_research_agent.config.rag import RagSettings


SUPPORTED_EMBEDDING_PROVIDERS = frozenset(
    {
        "huggingface",
    }
)


def create_embeddings(
    *,
    settings: RagSettings,
) -> Embeddings:
    """Create an embedding model from RAG settings."""

    provider = (
        settings.embedding_provider
        .strip()
        .lower()
    )

    if provider == "huggingface":
        return _create_huggingface_embeddings(
            settings=settings
        )

    supported_providers = ", ".join(
        sorted(SUPPORTED_EMBEDDING_PROVIDERS)
    )

    raise ValueError(
        f"Unsupported embedding provider: "
        f"{settings.embedding_provider!r}. "
        f"Supported providers: {supported_providers}."
    )


def _create_huggingface_embeddings(
    *,
    settings: RagSettings,
) -> Embeddings:
    """Create a local Hugging Face embedding model."""

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={
            "device": settings.embedding_device,
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": settings.ingestion_batch_size,
        },
        query_encode_kwargs={
            "normalize_embeddings": True,
        },
        show_progress=False,
    )