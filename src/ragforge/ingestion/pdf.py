"""Extract page-level text and source metadata from PDFs."""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class PDFPage:
    """One PDF page, with a one-based page number."""

    source: str
    page_number: int
    text: str

    @property
    def has_text(self) -> bool:
        """Whether the page contains non-whitespace extracted text."""
        return bool(self.text.strip())


def parse_pdf(path: str | Path) -> list[PDFPage]:
    """Preserve all PDF pages and extract existing text without OCR."""
    pdf_path = Path(path)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    reader = PdfReader(pdf_path)
    pages: list[PDFPage] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(
            PDFPage(
                source=pdf_path.name,
                page_number=page_number,
                text=text.strip(),
            )
        )

    return pages