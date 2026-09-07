"""Configuration for safe local-file tools used with MCP discovery."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from mini_research_agent.utils.paths import get_project_root


def _parse_extensions(value: str) -> tuple[str, ...]:
    extensions = []
    for item in value.split(","):
        extension = item.strip().lower()
        if not extension:
            continue
        if not extension.startswith("."):
            extension = f".{extension}"
        extensions.append(extension)
    return tuple(extensions)


@dataclass(frozen=True, slots=True)
class MCPReadSettings:
    """Hard limits for local-file evidence retrieval."""

    allowed_root: Path
    max_full_file_bytes: int = 100_000
    max_range_lines: int = 300
    max_output_characters: int = 30_000
    allowed_extensions: tuple[str, ...] = (
        ".md",
        ".txt",
        ".py",
        ".json",
        ".yaml",
        ".yml",
        ".csv",
    )

    def validate(self) -> None:
        if self.max_full_file_bytes <= 0:
            raise ValueError("MCP_MAX_FULL_FILE_BYTES must be greater than 0.")
        if self.max_range_lines <= 0:
            raise ValueError("MCP_MAX_RANGE_LINES must be greater than 0.")
        if self.max_output_characters <= 0:
            raise ValueError("MCP_MAX_OUTPUT_CHARACTERS must be greater than 0.")
        if not self.allowed_extensions:
            raise ValueError("MCP_ALLOWED_EXTENSIONS must not be empty.")


def create_mcp_read_settings() -> MCPReadSettings:
    """Load validated local-file limits from environment variables."""

    load_dotenv()
    root_value = os.getenv("MCP_FILES_ROOT", "files")
    root = Path(root_value)
    if not root.is_absolute():
        root = get_project_root() / root

    settings = MCPReadSettings(
        allowed_root=root.resolve(),
        max_full_file_bytes=int(
            os.getenv("MCP_MAX_FULL_FILE_BYTES", "100000")
        ),
        max_range_lines=int(
            os.getenv("MCP_MAX_RANGE_LINES", "300")
        ),
        max_output_characters=int(
            os.getenv("MCP_MAX_OUTPUT_CHARACTERS", "30000")
        ),
        allowed_extensions=_parse_extensions(
            os.getenv(
                "MCP_ALLOWED_EXTENSIONS",
                ".md,.txt,.py,.json,.yaml,.yml,.csv",
            )
        ),
    )
    settings.validate()
    return settings

