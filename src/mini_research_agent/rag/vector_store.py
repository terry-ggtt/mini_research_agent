"""Vector-store construction and persistence belong here."""

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from mini_research_agent.config.rag import RagSettings


def create_vector_store(
    *,
    settings: RagSettings,
    embeddings: Embeddings,
) -> VectorStore:
    """Create or connect to the persistent Chroma vector store."""

    settings.vector_store_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(
            settings.vector_store_path
        ),
    )