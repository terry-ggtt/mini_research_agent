"""Small infrastructure-independent utility functions."""

from .dates import get_today_str
from .paths import get_current_dir, get_project_root

__all__ = ["get_current_dir", "get_project_root", "get_today_str"]
