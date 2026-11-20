import logging
from pathlib import Path

from rag_overall.loader import load_documents

logger = logging.getLogger(__name__)


def main() -> int:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
