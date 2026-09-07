"""Configuration and runtime helpers for MCP tools."""

from dataclasses import dataclass
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import (
    MultiServerMCPClient,
)

from mini_research_agent.utils.paths import (
    get_project_root,
)


SAFE_MCP_DISCOVERY_TOOL_NAMES = frozenset(
    {
        "list_allowed_directories",
        "list_directory",
        "list_directory_with_sizes",
        "directory_tree",
        "search_files",
        "get_file_info",
    }
)


@dataclass(slots=True)
class MCPRuntime:
    """MCP client and tools kept alive by the application."""

    client: MultiServerMCPClient
    tools: list[BaseTool]


def create_filesystem_mcp_config(
    files_path: Path | None = None,
) -> dict:
    """Build the filesystem MCP server configuration."""

    allowed_path = (
        files_path
        or (get_project_root() / "files")
    ).resolve()

    return {
        "filesystem": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                str(allowed_path),
            ],
            "transport": "stdio",
        }
    }


async def create_mcp_runtime(
    *,
    config: dict | None = None,
    client: MultiServerMCPClient | None = None,
) -> MCPRuntime:
    """Create the MCP client and load its tools once."""

    runtime_client = (
        client
        if client is not None
        else MultiServerMCPClient(
            config
            if config is not None
            else create_filesystem_mcp_config()
        )
    )

    tools = await runtime_client.get_tools()

    return MCPRuntime(
        client=runtime_client,
        tools=list(tools),
    )


def filter_safe_mcp_tools(
    tools: list[BaseTool],
) -> list[BaseTool]:
    """Expose only read-only discovery tools from filesystem MCP."""

    return [
        tool
        for tool in tools
        if tool.name in SAFE_MCP_DISCOVERY_TOOL_NAMES
    ]
