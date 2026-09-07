import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


load_dotenv()


def create_chat_model(
    *,
    max_tokens: int | None = None,
):
    """Create a unified deepseek model for the project"""

    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing DEEPSEEK_API_KEY. Add it to the project's .env file."
        )

    config = {
        "model": "deepseek-v4-flash",
        "model_provider": "openai",
        "api_key": api_key,
        "base_url": "https://api.deepseek.com",
        "temperature": 0.0,
        "extra_body": {
            "thinking": {
                "type": "disabled",
            },
        },
    }

    if max_tokens is not None:
        config["max_tokens"] = max_tokens

    return init_chat_model(**config)
