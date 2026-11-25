import argparse
import logging
from pathlib import Path

from rag_overall.chunking import chunk_documents
from rag_overall.loader import load_documents

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Учебный RAG-пайплайн")
    parser.add_argument(
        "--chunks",
        action="store_true",
        help="Показать разбиение документов на чанки.",
    )
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=80)
    args = parser.parse_args()

    logging.basicConfig(format="%(levelname)s: %(message)s")
    knowledge_base = Path(__file__).resolve().parent / "knowledge_base"

    try:
        documents = load_documents(knowledge_base)
    except (OSError, ValueError) as error:
        logger.error("Не удалось загрузить базу знаний: %s", error)
        return 1

    if not documents:
        logger.error("В базе знаний нет непустых .txt-файлов.")
        return 1

    print(f"Загружено документов: {len(documents)}")
    for document in documents:
        print(f"{document.source_name} — {len(document.text)} символов")

    if args.chunks:
        try:
            chunks = chunk_documents(
                documents,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            )
        except ValueError as error:
            logger.error("Ошибка chunking: %s", error)
            return 1

        print(f"\nВсего чанков: {len(chunks)}")
        print(f"Размер чанка: {args.chunk_size}")
        print(f"Перекрытие: {args.overlap}")

        for chunk in chunks[:3]:
            print(
                f"\nchunk_id={chunk.chunk_id}\n"
                f"document_id={chunk.document_id}\n"
                f"source_name={chunk.source_name}\n"
                f"length={len(chunk.text)}"
            )
            print(f"Текст: {chunk.text[:120]!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
