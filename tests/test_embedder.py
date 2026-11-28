import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from rag_overall.config import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL
from rag_overall.embedder import Embedder, normalize_vector


def test_normalize_vector() -> None:
    result = normalize_vector([3.0, 4.0])

    assert result == pytest.approx([0.6, 0.8])
    assert math.hypot(*result) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "vector",
    [
        pytest.param([0.0, 0.0], id="zero-vector"),
        pytest.param([float("nan")], id="nan-value"),
        pytest.param([float("inf")], id="infinite-value"),
    ],
)
def test_normalize_invalid_vector(vector: list[float]) -> None:
    with pytest.raises(ValueError):
        normalize_vector(vector)


@pytest.mark.parametrize(
    "task_type",
    [
        "RETRIEVAL_DOCUMENT",
        "RETRIEVAL_QUERY",
    ],
)
def test_embed_text(task_type: str) -> None:
    client = MagicMock()
    client.models.embed_content.return_value = SimpleNamespace(
        embeddings=[SimpleNamespace(values=[2.0] * EMBEDDING_DIMENSIONS)]
    )

    embedder = Embedder(client, request_interval_seconds=0)
    vector = embedder.embed_text("Тестовый текст", task_type=task_type)

    assert len(vector) == EMBEDDING_DIMENSIONS
    assert math.hypot(*vector) == pytest.approx(1.0)

    client.models.embed_content.assert_called_once()

    arguments = client.models.embed_content.call_args.kwargs

    assert arguments["model"] == EMBEDDING_MODEL
    assert arguments["contents"] == "Тестовый текст"
    assert arguments["config"].task_type == task_type
    assert arguments["config"].output_dimensionality == EMBEDDING_DIMENSIONS


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty"),
        pytest.param(" \t\n ", id="whitespace"),
    ],
)
def test_embed_empty_text(text: str) -> None:
    client = MagicMock()
    embedder = Embedder(client)

    with pytest.raises(ValueError):
        embedder.embed_text(text)

    client.models.embed_content.assert_not_called()


@pytest.mark.parametrize(
    "embeddings",
    [
        pytest.param(None, id="missing-embeddings"),
        pytest.param([], id="empty-embeddings"),
        pytest.param(
            [SimpleNamespace(values=[]), SimpleNamespace(values=[])],
            id="multiple-embeddings",
        ),
        pytest.param(
            [SimpleNamespace(values=None)],
            id="missing-values",
        ),
        pytest.param(
            [SimpleNamespace(values=[1.0, 2.0])],
            id="wrong-dimension",
        ),
    ],
)
def test_embed_invalid_response(
    embeddings: list[SimpleNamespace] | None,
) -> None:
    client = MagicMock()
    client.models.embed_content.return_value = SimpleNamespace(embeddings=embeddings)

    embedder = Embedder(client, request_interval_seconds=0)

    with pytest.raises(ValueError):
        embedder.embed_text("Тестовый текст")


@pytest.mark.parametrize(
    "interval",
    [
        pytest.param(-1.0, id="negative-interval"),
        pytest.param(float("nan"), id="nan-interval"),
        pytest.param(float("inf"), id="infinite-interval"),
    ],
)
def test_invalid_request_interval(interval: float) -> None:
    with pytest.raises(ValueError):
        Embedder(MagicMock(), request_interval_seconds=interval)


@pytest.mark.parametrize(
    ("second_call_time", "expected_sleep"),
    [
        pytest.param(102.0, 4.0, id="wait-required"),
        pytest.param(108.0, None, id="no-wait-required"),
    ],
)
def test_request_interval(
    second_call_time: float,
    expected_sleep: float | None,
) -> None:
    client = MagicMock()
    client.models.embed_content.return_value = SimpleNamespace(
        embeddings=[SimpleNamespace(values=[1.0] * EMBEDDING_DIMENSIONS)]
    )

    embedder = Embedder(client, request_interval_seconds=6)

    next_start = max(106.0, second_call_time)

    with (
        patch(
            "rag_overall.embedder.monotonic",
            side_effect=[100.0, second_call_time, next_start],
        ),
        patch("rag_overall.embedder.sleep") as mock_sleep,
    ):
        embedder.embed_text("Первый текст")
        embedder.embed_text("Второй текст")

    if expected_sleep is None:
        mock_sleep.assert_not_called()
    else:
        mock_sleep.assert_called_once_with(expected_sleep)

    assert client.models.embed_content.call_count == 2
