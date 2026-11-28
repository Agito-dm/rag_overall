import math
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768
DEFAULT_EMBEDDING_INTERVAL = 1.0

WEAVIATE_HOST = "127.0.0.1"
WEAVIATE_HTTP_PORT = 8081
WEAVIATE_GRPC_PORT = 50051


def load_gemini_api_key() -> str:
    """Загружает API-ключ Gemini из окружения."""
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise ValueError("Не задан GEMINI_API_KEY.")

    return api_key


def load_embedding_interval() -> float:
    """Загружает интервал между запросами embeddings."""
    load_dotenv(PROJECT_ROOT / ".env")

    value = os.getenv(
        "EMBEDDING_REQUEST_INTERVAL_SECONDS",
        str(DEFAULT_EMBEDDING_INTERVAL),
    )

    interval = float(value)

    if not math.isfinite(interval) or interval < 0:
        raise ValueError("Интервал должен быть конечным и неотрицательным.")

    return interval
