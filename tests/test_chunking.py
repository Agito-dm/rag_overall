from itertools import pairwise
from pathlib import Path

import pytest

from rag_overall.chunking import chunk_document, chunk_documents, split_text
from rag_overall.loader import load_documents
from rag_overall.schemas import Document


@pytest.mark.parametrize(
    ("length", "expected_lengths"),
    [
        pytest.param(0, [], id="empty"),
        pytest.param(100, [100], id="short"),
        pytest.param(400, [400], id="exact-size"),
        pytest.param(401, [400, 81], id="one-more-than-size"),
        pytest.param(651, [400, 331], id="day01-document"),
        pytest.param(720, [400, 400], id="exact-two-windows"),
        pytest.param(721, [400, 400, 81], id="three-windows"),
    ],
)
def test_split_text_lengths(length: int, expected_lengths: list[int]) -> None:
    chunks = split_text("x" * length)

    assert [len(chunk) for chunk in chunks] == expected_lengths


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        pytest.param(0, 0, id="zero-size"),
        pytest.param(-1, 0, id="negative-size"),
        pytest.param(400, -1, id="negative-overlap"),
        pytest.param(400, 400, id="overlap-equals-size"),
        pytest.param(400, 401, id="overlap-larger-than-size"),
    ],
)
def test_split_text_invalid_parameters(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        split_text("Текст", chunk_size=chunk_size, overlap=overlap)


@pytest.mark.parametrize(
    ("length", "chunk_size", "overlap"),
    [
        pytest.param(651, 400, 80, id="two-chunks"),
        pytest.param(721, 400, 80, id="three-chunks"),
        pytest.param(1001, 400, 80, id="four-chunks"),
        pytest.param(651, 400, 0, id="without-overlap"),
    ],
)
def test_split_text_preserves_content(
    length: int, chunk_size: int, overlap: int
) -> None:
    text = "".join(chr(ord("А") + index % 32) for index in range(length))
    chunks = split_text(text, chunk_size=chunk_size, overlap=overlap)

    assert chunks

    for previous, current in pairwise(chunks):
        if overlap:
            assert previous[-overlap:] == current[:overlap]

    restored = chunks[0] + "".join(chunk[overlap:] for chunk in chunks[1:])
    assert restored == text


def test_chunk_document_preserves_metadata() -> None:
    document = Document(
        document_id="03_chunking.txt",
        source_name="03_chunking.txt",
        text="A" * 651,
    )

    chunks = chunk_document(document)

    assert [chunk.chunk_id for chunk in chunks] == [
        "03_chunking.txt:000",
        "03_chunking.txt:001",
    ]
    assert all(chunk.document_id == document.document_id for chunk in chunks)
    assert all(chunk.source_name == document.source_name for chunk in chunks)
    assert [len(chunk.text) for chunk in chunks] == [400, 331]


def test_chunk_documents_preserves_order() -> None:
    documents = [
        Document(document_id="01.txt", source_name="01.txt", text="A" * 651),
        Document(document_id="02.txt", source_name="02.txt", text="B" * 25),
    ]

    chunks = chunk_documents(documents)

    assert [chunk.chunk_id for chunk in chunks] == [
        "01.txt:000",
        "01.txt:001",
        "02.txt:000",
    ]
    assert chunks[-1].text == "B" * 25


def test_chunk_documents_empty_input() -> None:
    assert chunk_documents([]) == []


def test_chunk_real_knowledge_base() -> None:
    project_root = Path(__file__).resolve().parents[1]
    documents = load_documents(project_root / "knowledge_base")
    chunks = chunk_documents(documents)

    assert len(documents) >= 5
    assert len(chunks) > len(documents)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert {chunk.source_name for chunk in chunks} == {
        document.source_name for document in documents
    }
