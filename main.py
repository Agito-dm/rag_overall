import argparse
import logging

import weaviate
from google import genai
from google.genai.errors import APIError
from weaviate.exceptions import WeaviateBaseError

from rag_overall.chunking import chunk_documents
from rag_overall.config import (
    PROJECT_ROOT,
    WEAVIATE_GRPC_PORT,
    WEAVIATE_HOST,
    WEAVIATE_HTTP_PORT,
    load_embedding_interval,
    load_gemini_api_key,
)
from rag_overall.embedder import Embedder
from rag_overall.loader import load_documents
from rag_overall.schemas import Chunk
from rag_overall.weaviate_store import (
    COLLECTION_NAME,
    ensure_collection,
    index_chunk,
)

logger = logging.getLogger(__name__)


def run_index(chunks: list[Chunk]) -> None:
    """Индексирует чанки в Weaviate."""
    inserted = 0
    skipped = 0

    with (
        weaviate.connect_to_local(
            host=WEAVIATE_HOST,
            port=WEAVIATE_HTTP_PORT,
            grpc_port=WEAVIATE_GRPC_PORT,
        ) as client,
        genai.Client(api_key=load_gemini_api_key()) as gemini,
    ):
        ensure_collection(client)
        embedder = Embedder(gemini, load_embedding_interval())

        for index, chunk in enumerate(chunks, start=1):
            if index_chunk(client, embedder, chunk):
                inserted += 1
                status = "добавлен"
            else:
                skipped += 1
                status = "пропущен"

            print(
                f"[{index}/{len(chunks)}] {chunk.chunk_id}: {status}",
                flush=True,
            )

        collection = client.collections.use(COLLECTION_NAME)
        total = collection.aggregate.over_all(total_count=True).total_count

    print(f"\nДобавлено: {inserted}")
    print(f"Пропущено: {skipped}")
    print(f"Всего объектов в Weaviate: {total}")

    if total != len(chunks):
        raise ValueError(
            "Количество объектов не совпадает с количеством чанков. "
            "Необходимо перестроить индекс."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Учебный RAG-пайплайн")

    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--chunks",
        action="store_true",
        help="Показать разбиение документов на чанки.",
    )
    modes.add_argument(
        "--index",
        action="store_true",
        help="Загрузить чанки и embeddings в Weaviate.",
    )

    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=80)
    args = parser.parse_args()

    logging.basicConfig(format="%(levelname)s: %(message)s")
    knowledge_base = PROJECT_ROOT / "knowledge_base"

    try:
        documents = load_documents(knowledge_base)

        if not documents:
            logger.error("В базе знаний нет непустых .txt-файлов.")
            return 1

        print(f"Загружено документов: {len(documents)}")
        for document in documents:
            print(f"{document.source_name} — {len(document.text)} символов")

        if args.chunks or args.index:
            chunks = chunk_documents(
                documents,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            )

            print(f"\nВсего чанков: {len(chunks)}")
            print(f"Размер чанка: {args.chunk_size}")
            print(f"Перекрытие: {args.overlap}")

            if args.chunks:
                for chunk in chunks[:3]:
                    print(
                        f"\nchunk_id={chunk.chunk_id}\n"
                        f"document_id={chunk.document_id}\n"
                        f"source_name={chunk.source_name}\n"
                        f"length={len(chunk.text)}"
                    )
                    print(f"Текст: {chunk.text[:120]!r}")

            if args.index:
                run_index(chunks)

    except (OSError, ValueError, APIError, WeaviateBaseError) as error:
        logger.error("Ошибка выполнения: %s", error)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
