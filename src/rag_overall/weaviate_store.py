import logging

from weaviate import WeaviateClient
from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    VectorDistances,
)
from weaviate.util import generate_uuid5

from rag_overall.embedder import Embedder
from rag_overall.schemas import Chunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "KnowledgeChunk"


def ensure_collection(client: WeaviateClient) -> None:
    """Создаёт коллекцию чанков, если она ещё не существует."""
    if client.collections.exists(COLLECTION_NAME):
        logger.info("Коллекция %s уже существует.", COLLECTION_NAME)
        return

    client.collections.create(
        name=COLLECTION_NAME,
        properties=[
            Property(name="document_id", data_type=DataType.TEXT),
            Property(name="source_name", data_type=DataType.TEXT),
            Property(name="chunk_id", data_type=DataType.TEXT),
            Property(name="text", data_type=DataType.TEXT),
        ],
        vector_config=Configure.Vectors.self_provided(
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
            ),
        ),
    )

    logger.info("Создана коллекция %s.", COLLECTION_NAME)


def get_chunk_uuid(chunk: Chunk) -> str:
    """Возвращает детерминированный UUID чанка."""
    return generate_uuid5(chunk.chunk_id)


def index_chunk(
    client: WeaviateClient,
    embedder: Embedder,
    chunk: Chunk,
) -> bool:
    """Сохраняет новый чанк или пропускает неизменённый."""
    collection = client.collections.use(COLLECTION_NAME)
    object_uuid = get_chunk_uuid(chunk)
    properties = chunk.model_dump()

    existing = collection.query.fetch_object_by_id(object_uuid)

    if existing is not None:
        if existing.properties != properties:
            raise ValueError(
                f"Чанк {chunk.chunk_id} изменился. Необходимо перестроить индекс."
            )

        return False

    vector = embedder.embed_text(chunk.text)

    collection.data.insert(
        uuid=object_uuid,
        properties=properties,
        vector={"default": vector},
    )

    return True
