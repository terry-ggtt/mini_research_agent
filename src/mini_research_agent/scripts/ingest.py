"""Command-line entry point for offline RAG ingestion."""

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from mini_research_agent.config.rag import (
    RagSettings,
    create_rag_settings,
)
from mini_research_agent.rag.embeddings import create_embeddings
from mini_research_agent.rag.ingestion import ingest_knowledge_base
from mini_research_agent.rag.schemas import IngestionReport
from mini_research_agent.rag.vector_store import create_vector_store


def run_ingestion(
    *,
    settings: RagSettings | None = None,
    embeddings: Embeddings | None = None,
    vector_store: VectorStore | None = None,
) -> IngestionReport:
    """Build dependencies and synchronize the knowledge base."""

    active_settings = settings or create_rag_settings()
    active_vector_store = vector_store
    if active_vector_store is None:
        active_embeddings = embeddings or create_embeddings(
            settings=active_settings,
        )
        active_vector_store = create_vector_store(
            settings=active_settings,
            embeddings=active_embeddings,
        )

    return ingest_knowledge_base(
        settings=active_settings,
        vector_store=active_vector_store,
    )


def main() -> None:
    """Run ingestion and print a machine-readable report."""

    report = run_ingestion()
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
