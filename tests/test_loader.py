from pathlib import Path

import pytest

from rag_overall.loader import clean_text, load_documents


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        pytest.param(
            "  При индексации\t   приложение  ",
            "При индексации приложение",
            id="extra-spaces-and-tab",
        ),
        pytest.param(
            "Заголовок\r\n\r\n  \r\nАбзац\r\n",
            "Заголовок\nАбзац",
            id="windows-newlines-and-blank-lines",
        ),
        pytest.param(" \t\n ", "", id="whitespace-only"),
        pytest.param("Первая\nВторая", "Первая\nВторая", id="already-clean"),
    ],
)
def test_clean_text(raw_text: str, expected: str) -> None:
    assert clean_text(raw_text) == expected


def test_load_documents_reads_utf8_in_order(tmp_path: Path) -> None:
    (tmp_path / "02.TXT").write_text("  Второй\t текст ", encoding="utf-8-sig")
    (tmp_path / "01.txt").write_text("Первый текст", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert [doc.document_id for doc in documents] == ["01.txt", "02.TXT"]
    assert [doc.source_name for doc in documents] == ["01.txt", "02.TXT"]
    assert [doc.text for doc in documents] == ["Первый текст", "Второй текст"]


def test_load_documents_skips_empty_and_other_entries(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "empty.txt").write_text(" \t\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("Заметка", encoding="utf-8")
    (tmp_path / "nested.txt").mkdir()

    assert load_documents(tmp_path) == []
    assert "Пропущен пустой документ: empty.txt" in caplog.text


def test_load_documents_reports_invalid_encoding(tmp_path: Path) -> None:
    (tmp_path / "bad.txt").write_bytes(b"\xff")

    with pytest.raises(ValueError, match=r"bad\.txt.*UTF-8"):
        load_documents(tmp_path)
