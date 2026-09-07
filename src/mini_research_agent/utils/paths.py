"""Filesystem path helpers."""

from pathlib import Path


def get_current_dir() -> Path:
    """Return the directory containing this module."""

    return Path(__file__).resolve().parent


def get_project_root() -> Path:
    """Return the package project's root directory."""

    return Path(__file__).resolve().parents[3]
