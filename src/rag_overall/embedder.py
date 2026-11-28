import math
from time import monotonic, sleep

from google import genai
from google.genai import types

from rag_overall.config import (
    DEFAULT_EMBEDDING_INTERVAL,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
)


def normalize_vector(vector: list[float]) -> list[float]:
    """Выполняет L2-нормализацию вектора."""
    norm = math.hypot(*vector)

    if norm == 0 or not math.isfinite(norm):
        raise ValueError("Некорректная норма embedding-вектора.")

    return [value / norm for value in vector]


class Embedder:
    """Получает embeddings с ограничением частоты запросов."""

    def __init__(
        self,
        client: genai.Client,
        request_interval_seconds: float = DEFAULT_EMBEDDING_INTERVAL,
    ) -> None:
        if not math.isfinite(request_interval_seconds) or request_interval_seconds < 0:
            raise ValueError("Некорректный интервал запросов.")

        self._client = client
        self._request_interval_seconds = request_interval_seconds
        self._last_request_started: float | None = None

    def _wait_for_request(self) -> None:
        """Ожидает разрешённого времени следующего запроса."""
        if self._last_request_started is not None:
            elapsed = monotonic() - self._last_request_started
            remaining = self._request_interval_seconds - elapsed

            if remaining > 0:
                sleep(remaining)

        self._last_request_started = monotonic()

    def embed_text(
        self,
        text: str,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[float]:
        """Получает нормализованный embedding текста."""
        if not text.strip():
            raise ValueError("Нельзя получить embedding пустого текста.")

        self._wait_for_request()

        response = self._client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )

        if not response.embeddings or len(response.embeddings) != 1:
            raise ValueError("Gemini вернул некорректное количество embeddings.")

        values = response.embeddings[0].values

        if values is None or len(values) != EMBEDDING_DIMENSIONS:
            raise ValueError("Некорректная размерность embedding.")

        return normalize_vector(values)
