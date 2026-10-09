from pathlib import Path

import pytest
from pypdf import PdfWriter

from ragforge.ingestion.pdf import PDFPage, parse_pdf


def test_parse_pdf_preserves_empty_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)

    with pdf_path.open("wb") as file:
        writer.write(file)

    pages = parse_pdf(pdf_path)

    assert len(pages) == 2
    assert [page.page_number for page in pages] == [1, 2]
    assert all(page.source == "sample.pdf" for page in pages)
    assert all(page.text == "" for page in pages)
    assert all(not page.has_text for page in pages)


def test_parse_pdf_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        parse_pdf(tmp_path / "missing.pdf")


def test_page_has_text() -> None:
    page = PDFPage(source="sample.pdf", page_number=1, text="Hello")

    assert page.has_text
