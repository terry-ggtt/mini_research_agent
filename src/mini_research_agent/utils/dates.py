"""Date formatting utilities used by prompts."""

from datetime import datetime


def get_today_str() -> str:
    """Return today's date in the format expected by project prompts."""

    today = datetime.now()
    return f"{today:%a %b} {today.day},{today:%Y}"
