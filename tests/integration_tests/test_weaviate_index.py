import math

import pytest
import weaviate

from rag_overall.chunking import chunk_documents
from rag_overall.config import (
    EMBEDDING_DIMENSIONS,
    PROJECT_ROOT,
    WEAVIATE_GRPC_PORT,
    WEAVIATE_HOST,
    WEAVIATE_HTTP_PORT,
)
from rag_overall.loader import load_documents
from rag_overall.weaviate_store import COLLECTION_NAME, get_chunk_uuid


@pytest.mark.integration
def test_weaviate_index() -> None:
    documents = load_documents(PROJECT_ROOT / "knowledge_base")
    chunks = chunk_documents(documents)

    assert chunks

    expected = {get_chunk_uuid(chunk): chunk.model_dump() for chunk in chunks}

    with weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_HTTP_PORT,
        grpc_port=WEAVIATE_GRPC_PORT,
    ) as client:
        assert client.is_ready()
        assert client.collections.exists(COLLECTION_NAME)

        collection = client.collections.use(COLLECTION_NAME)
        objects = list(collection.iterator(include_vector=True))

    assert len(objects) == len(expected)
    assert {str(obj.uuid) for obj in objects} == set(expected)

    for obj in objects:
        assert obj.properties == expected[str(obj.uuid)]

        vector = obj.vector["default"]

        assert len(vector) == EMBEDDING_DIMENSIONS
        assert all(math.isfinite(value) for value in vector)
        assert math.hypot(*vector) == pytest.approx(1.0, abs=1e-5)

    print(f"\nПроверено объектов: {len(objects)}")
    print(f"Размерность векторов: {EMBEDDING_DIMENSIONS}")
