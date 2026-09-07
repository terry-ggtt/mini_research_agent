from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool

from mini_research_agent.config.rag import (
    create_rag_settings,
)
from mini_research_agent.config.mcp import (
    create_mcp_read_settings,
)
from mini_research_agent.graphs.full_agent import (
    create_full_agent,
)
from mini_research_agent.graphs.research import (
    create_research_graph,
)
from mini_research_agent.graphs.supervisor import (
    create_supervisor_graph,
)
from mini_research_agent.rag.embeddings import (
    create_embeddings,
)
from mini_research_agent.rag.retriever import (
    create_retriever,
)
from mini_research_agent.rag.tool import (
    create_knowledge_base_tools,
)
from mini_research_agent.rag.vector_store import (
    create_vector_store,
)
from mini_research_agent.tools.mcp import (
    MCPRuntime,
    create_filesystem_mcp_config,

    create_mcp_runtime,
    filter_safe_mcp_tools,
)
from mini_research_agent.tools.file_reader import (
    create_bounded_file_tools,
)
from mini_research_agent.tools.web_search import (
    create_research_tools,
)


@dataclass(slots=True)
class ApplicationRuntime:
    """Resources that must live for the application lifetime."""

    graph: Any
    mcp: MCPRuntime

async def create_application(
    *,
    checkpointer=None,
    rag_settings=None,
    embeddings=None,
    vector_store=None,
    retriever=None,
    knowledge_tools: Sequence[BaseTool] | None = None,
    web_tools: Sequence[BaseTool] | None = None,
    mcp_runtime: MCPRuntime | None = None,
    mcp_read_settings=None,
    bounded_file_tools: Sequence[BaseTool] | None = None,
    research_model=None,
    compression_model=None,
    supervisor_model=None,
    writer_model=None,
    scope_graph=None,
) -> ApplicationRuntime:
    active_knowledge_tools = knowledge_tools
    if active_knowledge_tools is None:
        active_rag_settings = (
            rag_settings
            if rag_settings is not None
            else create_rag_settings()
        )
        active_embeddings = (
            embeddings
            if embeddings is not None
            else create_embeddings(settings=active_rag_settings)
        )
        active_vector_store = (
            vector_store
            if vector_store is not None
            else create_vector_store(
                settings=active_rag_settings,
                embeddings=active_embeddings,
            )
        )
        active_retriever = (
            retriever
            if retriever is not None
            else create_retriever(
                settings=active_rag_settings,
                vector_store=active_vector_store,
            )
        )
        active_knowledge_tools = create_knowledge_base_tools(
            retriever=active_retriever,
        )

    active_web_tools = (
        web_tools
        if web_tools is not None
        else create_research_tools()
    )

        # 两类工具共享同一份配置。
    active_mcp_read_settings = (
        mcp_read_settings
        if mcp_read_settings is not None
        else create_mcp_read_settings()
    )

    active_mcp_read_settings.validate()

    allowed_root = active_mcp_read_settings.allowed_root.resolve()

    # 在启动 MCP Server 前检查目录。
    if not allowed_root.is_dir():
        raise NotADirectoryError(
            f"MCP allowed root is not a directory: {allowed_root}"
        )

    # 将同一个目录传入 MCP Server 配置。
    active_mcp_runtime = (
        mcp_runtime
        if mcp_runtime is not None
        else await create_mcp_runtime(
            config=create_filesystem_mcp_config(
                files_path=allowed_root,
            )
        )
    )

    safe_mcp_tools = filter_safe_mcp_tools(
        active_mcp_runtime.tools,
    )

    # 受控读取工具使用同一份配置。
    active_bounded_file_tools = (
        bounded_file_tools
        if bounded_file_tools is not None
        else create_bounded_file_tools(
            settings=active_mcp_read_settings,
        )
    )
    if active_bounded_file_tools is None:
        active_mcp_read_settings = (
            mcp_read_settings
            if mcp_read_settings is not None
            else create_mcp_read_settings()
        )
        active_bounded_file_tools = create_bounded_file_tools(
            settings=active_mcp_read_settings,
        )

    research_tools = [
        *active_web_tools,
        *active_knowledge_tools,
        *safe_mcp_tools,
        *active_bounded_file_tools,
    ]

    research_graph = create_research_graph(
        model=research_model,
        compress_model=compression_model,
        tools=research_tools,
    )

    supervisor_graph = create_supervisor_graph(
        model=supervisor_model,
        research_graph=research_graph,
    )

    full_agent = create_full_agent(
        writer_model=writer_model,
        scope_graph=scope_graph,
        supervisor_graph=supervisor_graph,
        checkpointer=checkpointer,
    )

    return ApplicationRuntime(
        graph=full_agent,
        mcp=active_mcp_runtime,
    )
