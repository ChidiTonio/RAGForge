"""Test page-aware chunking using synthetic text."""

from itertools import pairwise

import pytest

from ragforge.ingestion.chunking import chunk_pages
from ragforge.ingestion.pdf import PDFPage


def test_short_page_preserves_text_and_metadata():
    page = PDFPage(source="example.pdf", page_number=3, text="Hello world")

    chunks = chunk_pages([page], chunk_size=20, overlap=5)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.source == "example.pdf"
    assert chunk.page_number == 3
    assert chunk.chunk_number == 1
    assert chunk.start_char == 0
    assert chunk.end_char == 11
    assert chunk.text == page.text


def test_overlap_and_offsets():
    page = PDFPage(
        source="example.pdf", page_number=1, text="abcdefghijklmn"
    )

    chunks = chunk_pages([page], chunk_size=6, overlap=2)

    assert [chunk.text for chunk in chunks] == [
        "abcdef",
        "efghij",
        "ijklmn",
    ]
    assert [(chunk.start_char, chunk.end_char) for chunk in chunks] == [
        (0, 6),
        (4, 10),
        (8, 14),
    ]
    assert [chunk.chunk_number for chunk in chunks] == [1, 2, 3]

    for chunk in chunks:
        assert chunk.text == page.text[chunk.start_char:chunk.end_char]
    for previous, current in pairwise(chunks):
        assert previous.end_char - current.start_char == 2
        assert previous.text[-2:] == current.text[:2]

    
def test_zero_overlap_preserves_short_final_chunk():
    page = PDFPage(source="example.pdf", page_number=1, text="abcdefg")

    chunks = chunk_pages([page], chunk_size=3, overlap=0)

    assert [chunk.text for chunk in chunks] == ["abc", "def", "g"]
    assert [(chunk.start_char, chunk.end_char) for chunk in chunks] == [
        (0, 3),
        (3, 6),
        (6, 7),
    ]
    assert "".join(chunk.text for chunk in chunks) == page.text


def test_chunks_stay_on_their_page_and_numbering_restarts():
    pages = [
        PDFPage(source="example.pdf", page_number=1, text="abcdef"),
        PDFPage(source="example.pdf", page_number=2, text="XYZ"),
    ]

    chunks = chunk_pages(pages, chunk_size=4, overlap=1)

    assert [
        (chunk.page_number, chunk.chunk_number, chunk.text)
        for chunk in chunks
    ] == [
        (1, 1, "abcd"),
        (1, 2, "def"),
        (2, 1, "XYZ"),
    ]

    for chunk in chunks:
        page = pages[chunk.page_number - 1]
        assert chunk.source == page.source
        assert chunk.text == page.text[chunk.start_char:chunk.end_char]


def test_blank_pages_are_skipped_without_renumbering_pages():
    pages = [
        PDFPage(source="example.pdf", page_number=1, text=""),
        PDFPage(source="example.pdf", page_number=2, text=" \n\t "),
        PDFPage(source="example.pdf", page_number=3, text="Hello"),
    ]

    chunks = chunk_pages(pages, chunk_size=10, overlap=0)

    assert len(chunks) == 1
    assert chunks[0].page_number == 3
    assert chunks[0].chunk_number == 1
    assert chunks[0].text == "Hello"


def test_whitespace_only_chunks_are_skipped():
    page = PDFPage(
        source="example.pdf", page_number=1, text="abc      xyz"
    )

    chunks = chunk_pages([page], chunk_size=3, overlap=0)

    assert [chunk.text for chunk in chunks] == ["abc", "xyz"]
    assert [chunk.chunk_number for chunk in chunks] == [1, 2]
    assert [(chunk.start_char, chunk.end_char) for chunk in chunks] == [
        (0, 3),
        (9, 12),
    ]


def test_empty_input_returns_no_chunks():
    assert chunk_pages([]) == []


def test_accepts_a_generator():
    pages = (
        PDFPage(source="example.pdf", page_number=number, text="Hello")
        for number in range(1, 3)
    )

    chunks = chunk_pages(pages, chunk_size=10, overlap=0)

    assert [chunk.page_number for chunk in chunks] == [1, 2]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),
        (-1, 0),
        (10, -1),
        (10, 10),
        (10, 11),
    ],
)
def test_invalid_parameters_raise_value_error(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_pages([], chunk_size=chunk_size, overlap=overlap)