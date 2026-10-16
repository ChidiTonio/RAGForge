"""Extract page-level text and source metadata from PDFs with size limits."""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 200
MAX_EXTRACTED_CHARS = 1_000_000


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


def parse_pdf(
    path: str | Path,
    *,
    max_bytes: int = MAX_PDF_BYTES,
    max_pages: int = MAX_PDF_PAGES,
    max_chars: int = MAX_EXTRACTED_CHARS,
) -> list[PDFPage]:
    """Preserve pages and extract existing text without OCR.

    Limits reject oversized documents rather than silently truncating them.
    These checks are not a sandbox or a PDF extraction timeout.
    """
    for name, value in (
        ("max_bytes", max_bytes),
        ("max_pages", max_pages),
        ("max_chars", max_chars),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero.")

    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if pdf_path.stat().st_size > max_bytes:
        raise ValueError(f"PDF exceeds the file size limit of {max_bytes} bytes.")

    reader = PdfReader(pdf_path)
    if len(reader.pages) > max_pages:
        raise ValueError(f"PDF exceeds the page limit of {max_pages}.")

    pages: list[PDFPage] = []
    total_chars = 0

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        total_chars += len(text)
        if total_chars > max_chars:
            raise ValueError(
                f"PDF exceeds the extracted text limit of {max_chars} characters."
            )
        pages.append(
            PDFPage(
                source=pdf_path.name,
                page_number=page_number,
                text=text.strip(),
            )
        )

    return pages
