"""Regression tests for PDF ingestion limits without text extraction."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ragforge.ingestion import pdf


@pytest.fixture
def document(tmp_path, monkeypatch):
    path = tmp_path / "sample.pdf"
    path.write_bytes(b"12345")
    pages = [
        SimpleNamespace(extract_text=Mock(return_value=" abc ")),
        SimpleNamespace(extract_text=Mock(return_value="def")),
    ]
    reader = Mock(return_value=SimpleNamespace(pages=pages))
    monkeypatch.setattr(pdf, "PdfReader", reader)
    return SimpleNamespace(path=path, pages=pages, reader=reader)


def test_file_size_limit_rejects_before_opening_reader(document):
    with pytest.raises(ValueError, match="file size limit"):
        pdf.parse_pdf(document.path, max_bytes=4)
    document.reader.assert_not_called()


def test_page_limit_rejects_before_extracting_text(document):
    with pytest.raises(ValueError, match="page limit"):
        pdf.parse_pdf(document.path, max_pages=1)
    for page in document.pages:
        page.extract_text.assert_not_called()


def test_exact_limits_are_accepted(document):
    pages = pdf.parse_pdf(
        document.path, max_bytes=5, max_pages=2, max_chars=8
    )
    assert [page.text for page in pages] == ["abc", "def"]
    assert [page.page_number for page in pages] == [1, 2]


def test_character_limit_is_cumulative(document):
    with pytest.raises(ValueError, match="extracted text limit"):
        pdf.parse_pdf(document.path, max_chars=7)
    for page in document.pages:
        page.extract_text.assert_called_once()


def test_character_limit_counts_unstripped_text(document):
    with pytest.raises(ValueError, match="extracted text limit"):
        pdf.parse_pdf(document.path, max_chars=4)
    document.pages[0].extract_text.assert_called_once()
    document.pages[1].extract_text.assert_not_called()


@pytest.mark.parametrize("limit", ["max_bytes", "max_pages", "max_chars"])
@pytest.mark.parametrize(
    ("value", "error"),
    [
        (True, TypeError),
        (1.5, TypeError),
        ("10", TypeError),
        (None, TypeError),
        (0, ValueError),
        (-1, ValueError),
    ],
)
def test_invalid_limits_fail_before_opening_reader(document, limit, value, error):
    with pytest.raises(error, match=limit):
        pdf.parse_pdf(document.path, **{limit: value})
    document.reader.assert_not_called()
