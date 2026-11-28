from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rag_overall.schemas import Chunk
from rag_overall.weaviate_store import (
    COLLECTION_NAME,
    ensure_collection,
    get_chunk_uuid,
    index_chunk,
)


@pytest.mark.parametrize(
    "collection_exists",
    [
        pytest.param(True, id="existing-collection"),
        pytest.param(False, id="missing-collection"),
    ],
)
def test_ensure_collection(collection_exists: bool) -> None:
    client = MagicMock()
    client.collections.exists.return_value = collection_exists

    ensure_collection(client)

    client.collections.exists.assert_called_once_with(COLLECTION_NAME)

    if collection_exists:
        client.collections.create.assert_not_called()
    else:
        client.collections.create.assert_called_once()

        arguments = client.collections.create.call_args.kwargs

        assert arguments["name"] == COLLECTION_NAME
        assert {prop.name for prop in arguments["properties"]} == {
            "document_id",
            "source_name",
            "chunk_id",
            "text",
        }


def test_chunk_uuid_is_deterministic() -> None:
    chunk = Chunk(
        document_id="01.txt",
        source_name="01.txt",
        chunk_id="01.txt:000",
        text="Тестовый текст",
    )

    another_chunk = chunk.model_copy(update={"chunk_id": "01.txt:001"})

    assert get_chunk_uuid(chunk) == get_chunk_uuid(chunk.model_copy())
    assert get_chunk_uuid(chunk) != get_chunk_uuid(another_chunk)


@pytest.mark.parametrize(
    "scenario",
    ["new", "existing", "changed"],
)
def test_index_chunk(scenario: str) -> None:
    client = MagicMock()
    embedder = MagicMock()
    collection = client.collections.use.return_value

    chunk = Chunk(
        document_id="01.txt",
        source_name="01.txt",
        chunk_id="01.txt:000",
        text="Тестовый текст",
    )

    properties = chunk.model_dump()
    vector = [0.5] * 768
    embedder.embed_text.return_value = vector

    if scenario == "new":
        collection.query.fetch_object_by_id.return_value = None
    elif scenario == "existing":
        collection.query.fetch_object_by_id.return_value = SimpleNamespace(
            properties=properties
        )
    else:
        collection.query.fetch_object_by_id.return_value = SimpleNamespace(
            properties={**properties, "text": "Старый текст"}
        )

    if scenario == "changed":
        with pytest.raises(ValueError, match="изменился"):
            index_chunk(client, embedder, chunk)
    else:
        result = index_chunk(client, embedder, chunk)
        assert result is (scenario == "new")

    collection.query.fetch_object_by_id.assert_called_once_with(get_chunk_uuid(chunk))

    if scenario == "new":
        embedder.embed_text.assert_called_once_with(chunk.text)
        collection.data.insert.assert_called_once_with(
            uuid=get_chunk_uuid(chunk),
            properties=properties,
            vector={"default": vector},
        )
    else:
        embedder.embed_text.assert_not_called()
        collection.data.insert.assert_not_called()
