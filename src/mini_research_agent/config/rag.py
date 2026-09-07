import os 
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# D:\ai_agent\deep_research_from_scratch\mini_research_agent\src\mini_research_agent\config\rag.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]

def _resolve_project_path(Value: str)->Path:
    """将相对路径转换为相对于项目根目录的绝对路径。"""
    path = Path(Value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()

def _parse_extensions(value:str)->tuple[str,...]:
    """把 '.md,.txt' 转换成规范化的扩展名元组。"""
    extensions =  []

    for extension in value.split(','):
        extension = extension.strip().lower()
        if not extension: continue
        if not extension.startswith('.'):
            extension = f".{extension}"
        extensions.append(extension)
    return tuple(extensions)

def _parse_optional_float(value: str | None) -> float | None:
    """把可选环境变量转换为浮点数。"""

    if value is None or not value.strip():
        return None

    return float(value)


@dataclass(frozen=True , slots=True)
class RagSettings:
    """RAG 入库和检索配置。"""
    knowledge_base_path: Path
    vector_store_path: Path
    collection_name: str

    embedding_provider: str
    embedding_model: str
    embedding_base_url: str | None
    embedding_device: str

    chunk_size: int
    chunk_overlap: int

    retrieval_top_k: int
    score_threshold: float | None

    supported_extensions: tuple[str, ...]
    ingestion_batch_size: int
    @property
    def manifest_path(self) -> Path:
        """索引清单文件的位置。"""

        return self.vector_store_path / "manifest.json"

    def validate(self) -> None:
        """在创建 RAG 运行组件前验证配置。"""

        if not self.embedding_provider:
            raise ValueError("RAG_EMBEDDING_PROVIDER cannot be empty.")

        if not self.embedding_model:
            raise ValueError("RAG_EMBEDDING_MODEL cannot be empty.")

        if not self.collection_name:
            raise ValueError("RAG_COLLECTION_NAME cannot be empty.")

        if self.chunk_size <= 0:
            raise ValueError("RAG_CHUNK_SIZE must be greater than 0.")

        if self.chunk_overlap < 0:
            raise ValueError("RAG_CHUNK_OVERLAP cannot be negative.")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE."
            )

        if self.retrieval_top_k <= 0:
            raise ValueError("RAG_RETRIEVAL_TOP_K must be greater than 0.")

        if (
            self.score_threshold is not None
            and not 0.0 <= self.score_threshold <= 1.0
        ):
            raise ValueError(
                "RAG_SCORE_THRESHOLD must be between 0.0 and 1.0."
            )

        if self.ingestion_batch_size <= 0:
            raise ValueError(
                "RAG_INGESTION_BATCH_SIZE must be greater than 0."
            )
        if not self.embedding_device:
            raise ValueError(
                "RAG_EMBEDDING_DEVICE cannot be empty."
            )
        if not self.supported_extensions:
            raise ValueError(
                "RAG_SUPPORTED_EXTENSIONS must contain at least one extension."
            )


def create_rag_settings() -> RagSettings:
    """从环境变量创建一份经过验证的 RAG 配置。"""

    load_dotenv()

    settings = RagSettings(
        knowledge_base_path=_resolve_project_path(
            os.getenv(
                "RAG_KNOWLEDGE_BASE_PATH",
                "data/knowledge_base",
            )
        ),
        vector_store_path=_resolve_project_path(
            os.getenv(
                "RAG_VECTOR_STORE_PATH",
                "data/vector_store",
            )
        ),
        collection_name=os.getenv(
            "RAG_COLLECTION_NAME",
            "mini_research_knowledge",
        ),
        embedding_provider=os.getenv(
            "RAG_EMBEDDING_PROVIDER",
            "",
        ).strip(),
        embedding_model=os.getenv(
            "RAG_EMBEDDING_MODEL",
            "",
        ).strip(),
        embedding_base_url=(
            os.getenv("RAG_EMBEDDING_BASE_URL") or None
        ),
        embedding_device=os.getenv(
            "RAG_EMBEDDING_DEVICE",
            "cpu",
        ).strip(),
        chunk_size=int(
            os.getenv("RAG_CHUNK_SIZE", "1000")
        ),
        chunk_overlap=int(
            os.getenv("RAG_CHUNK_OVERLAP", "150")
        ),
        retrieval_top_k=int(
            os.getenv("RAG_RETRIEVAL_TOP_K", "5")
        ),
        score_threshold=_parse_optional_float(
            os.getenv("RAG_SCORE_THRESHOLD")
        ),
        supported_extensions=_parse_extensions(
            os.getenv(
                "RAG_SUPPORTED_EXTENSIONS",
                ".md,.txt",
            )
        ),
        ingestion_batch_size=int(
            os.getenv("RAG_INGESTION_BATCH_SIZE", "32")
        ),
    )

    settings.validate()

    return settings


  



