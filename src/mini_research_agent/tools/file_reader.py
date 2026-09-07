"""Bounded, read-only local-file tools for exact evidence retrieval."""

from pathlib import Path

from langchain_core.tools import BaseTool, tool

from mini_research_agent.config.mcp import MCPReadSettings


def _resolve_safe_file(path: str, settings: MCPReadSettings) -> Path:
    root = settings.allowed_root.resolve()
    requested = Path(path)
    resolved = (
        requested if requested.is_absolute() else root / requested
    ).resolve()

    if resolved != root and root not in resolved.parents:
        raise ValueError("Requested file is outside the allowed root.")
    if not resolved.exists():
        raise FileNotFoundError(f"File does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Path is not a file: {resolved}")
    if resolved.suffix.lower() not in settings.allowed_extensions:
        raise ValueError(
            f"File extension is not allowed: {resolved.suffix.lower()}"
        )
    return resolved


def _source_label(path: Path, settings: MCPReadSettings) -> str:
    return path.relative_to(settings.allowed_root.resolve()).as_posix()


def create_bounded_file_tools(
    *,
    settings: MCPReadSettings,
) -> list[BaseTool]:
    """Create safe inspection and line-range reading tools."""

    settings.validate()
    settings.allowed_root.mkdir(parents=True, exist_ok=True)

    @tool("inspect_file", parse_docstring=True)
    def inspect_file(path: str) -> str:
        """Inspect one allowed local file without reading its content.

        Args:
            path: Relative path inside the configured files directory, or an
                absolute path that resolves inside that directory.
        """

        resolved = _resolve_safe_file(path, settings)
        size = resolved.stat().st_size
        source = _source_label(resolved, settings)
        return "\n".join(
            [
                f"Source: {source}",
                f"Size bytes: {size}",
                f"Extension: {resolved.suffix.lower()}",
                "Full read allowed: "
                + str(size <= settings.max_full_file_bytes).lower(),
                f"Maximum range lines: {settings.max_range_lines}",
            ]
        )

    @tool("read_file_range", parse_docstring=True)
    def read_file_range(
        path: str,
        start_line: int,
        end_line: int,
    ) -> str:
        """Read a bounded line range from one allowed local text file.

        Args:
            path: Relative path inside the configured files directory, or an
                absolute path that resolves inside that directory.
            start_line: One-based first line to read.
            end_line: One-based final line to read, inclusive.
        """

        if start_line < 1:
            raise ValueError("start_line must be greater than 0.")
        if end_line < start_line:
            raise ValueError("end_line must be greater than or equal to start_line.")
        if end_line - start_line + 1 > settings.max_range_lines:
            raise ValueError(
                f"Requested range exceeds {settings.max_range_lines} lines."
            )

        resolved = _resolve_safe_file(path, settings)
        source = _source_label(resolved, settings)
        content_parts: list[str] = []
        character_count = 0
        actual_end = start_line - 1
        truncated = False
        more_content = False

        try:
            with resolved.open("r", encoding="utf-8-sig") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if line_number < start_line:
                        continue
                    if line_number > end_line:
                        more_content = True
                        break

                    remaining = settings.max_output_characters - character_count
                    if remaining <= 0:
                        truncated = True
                        break
                    if len(line) > remaining:
                        content_parts.append(line[:remaining])
                        character_count += remaining
                        actual_end = line_number
                        truncated = True
                        break

                    content_parts.append(line)
                    character_count += len(line)
                    actual_end = line_number
        except UnicodeDecodeError as exc:
            raise ValueError(f"File is not valid UTF-8 text: {source}") from exc

        if actual_end < start_line:
            raise ValueError(
                f"start_line {start_line} is beyond the end of file: {source}"
            )

        citation = f"FILE:{source}#L{start_line}-L{actual_end}"
        return "\n".join(
            [
                f"Citation ID: {citation}",
                f"Source: {source}",
                f"Lines: {start_line}-{actual_end}",
                f"Truncated: {str(truncated).lower()}",
                f"More content available: {str(more_content or truncated).lower()}",
                "",
                "CONTENT:",
                "".join(content_parts),
            ]
        )

    return [inspect_file, read_file_range]

