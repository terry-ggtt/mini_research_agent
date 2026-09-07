from pathlib import Path

import pytest
from langchain_core.tools import tool

from mini_research_agent.config.mcp import MCPReadSettings
from mini_research_agent.tools.file_reader import create_bounded_file_tools
from mini_research_agent.tools.mcp import (
    create_mcp_runtime,
    filter_safe_mcp_tools,
)


def make_settings(tmp_path: Path, **overrides) -> MCPReadSettings:
    values = {
        "allowed_root": tmp_path / "files",
        "max_full_file_bytes": 100,
        "max_range_lines": 3,
        "max_output_characters": 20,
        "allowed_extensions": (".txt", ".md"),
    }
    values.update(overrides)
    return MCPReadSettings(**values)


def tool_map(settings):
    return {
        research_tool.name: research_tool
        for research_tool in create_bounded_file_tools(settings=settings)
    }


def test_inspect_and_read_file_range(tmp_path):
    settings = make_settings(tmp_path, max_output_characters=100)
    settings.allowed_root.mkdir()
    source = settings.allowed_root / "notes.txt"
    source.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    tools = tool_map(settings)

    inspection = tools["inspect_file"].invoke({"path": "notes.txt"})
    output = tools["read_file_range"].invoke(
        {"path": "notes.txt", "start_line": 2, "end_line": 3}
    )

    assert "Source: notes.txt" in inspection
    assert "Citation ID: FILE:notes.txt#L2-L3" in output
    assert "two\nthree" in output
    assert "four" not in output


def test_file_reader_rejects_escape_extension_and_large_range(tmp_path):
    settings = make_settings(tmp_path)
    settings.allowed_root.mkdir()
    (settings.allowed_root / "data.bin").write_bytes(b"binary")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    tools = tool_map(settings)

    with pytest.raises(ValueError, match="outside the allowed root"):
        tools["inspect_file"].invoke({"path": str(outside)})

    with pytest.raises(ValueError, match="extension is not allowed"):
        tools["inspect_file"].invoke({"path": "data.bin"})

    source = settings.allowed_root / "notes.txt"
    source.write_text("1\n2\n3\n4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds 3 lines"):
        tools["read_file_range"].invoke(
            {"path": "notes.txt", "start_line": 1, "end_line": 4}
        )


def test_file_reader_marks_character_truncation(tmp_path):
    settings = make_settings(tmp_path, max_output_characters=5)
    settings.allowed_root.mkdir()
    (settings.allowed_root / "notes.txt").write_text(
        "abcdefghij\n",
        encoding="utf-8",
    )
    reader = tool_map(settings)["read_file_range"]

    output = reader.invoke(
        {"path": "notes.txt", "start_line": 1, "end_line": 1}
    )

    assert "Truncated: true" in output
    assert "More content available: true" in output
    assert output.endswith("abcde")


@tool("list_directory")
def list_directory(path: str) -> str:
    """List a directory."""

    return path


@tool("read_file")
def read_file(path: str) -> str:
    """Read an unrestricted file."""

    return path


@tool("write_file")
def write_file(path: str, content: str) -> str:
    """Write a file."""

    return path + content


def test_mcp_filter_only_keeps_read_only_discovery_tools():
    filtered = filter_safe_mcp_tools(
        [list_directory, read_file, write_file]
    )

    assert [item.name for item in filtered] == ["list_directory"]


@pytest.mark.asyncio
async def test_mcp_runtime_loads_tools_once_per_runtime_creation():
    class StubClient:
        def __init__(self):
            self.calls = 0

        async def get_tools(self):
            self.calls += 1
            return [list_directory]

    client = StubClient()
    runtime = await create_mcp_runtime(client=client)

    assert client.calls == 1
    assert runtime.client is client
    assert runtime.tools == [list_directory]

