from rag_overall.schemas import Chunk, Document


def split_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[str]:
    """Разбивает текст на чанки с перекрытием."""
    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть больше нуля.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap должен быть от 0 до chunk_size - 1.")

    step = chunk_size - overlap
    chunks: list[str] = []

    for start in range(0, len(text), step):
        end = start + chunk_size
        chunks.append(text[start:end])

        if end >= len(text):
            break

    return chunks


def chunk_document(
    document: Document,
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[Chunk]:
    """Разбивает документ и добавляет метаданные каждому чанку."""
    texts = split_text(document.text, chunk_size=chunk_size, overlap=overlap)

    return [
        Chunk(
            document_id=document.document_id,
            source_name=document.source_name,
            chunk_id=f"{document.document_id}:{index:03d}",
            text=text,
        )
        for index, text in enumerate(texts)
    ]


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[Chunk]:
    """Разбивает несколько документов, сохраняя исходный порядок."""
    chunks: list[Chunk] = []

    for document in documents:
        chunks.extend(chunk_document(document, chunk_size=chunk_size, overlap=overlap))

    return chunks
