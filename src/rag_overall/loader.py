import logging
from pathlib import Path

from rag_overall.schemas import Document

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Нормализует пробелы и убирает пустые строки."""
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def load_documents(directory: Path) -> list[Document]:
    """Загружает непустые текстовые файлы из каталога."""
    documents: list[Document] = []

    for path in sorted(directory.iterdir(), key=lambda path: path.name):
        if not path.is_file() or path.suffix.lower() != ".txt":
            continue

        try:
            text = clean_text(path.read_text(encoding="utf-8-sig"))
        except UnicodeDecodeError as error:
            raise ValueError(f"Файл {path.name} должен быть в UTF-8.") from error

        if not text:
            logger.warning("Пропущен пустой документ: %s", path.name)
            continue

        documents.append(
            Document(document_id=path.name, source_name=path.name, text=text)
        )

    return documents
