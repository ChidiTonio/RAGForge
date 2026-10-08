"""Export a single PDF's chunks for manual retrieval-evaluation labeling."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from ragforge.ingestion.chunking import chunk_pages
from ragforge.ingestion.pdf import parse_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/equity_research"),
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=150)
    args = parser.parse_args()

    if not args.pdf.is_file():
        parser.error(f"PDF file not found: {args.pdf}")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be greater than zero.")
    if not 0 <= args.overlap < args.chunk_size:
        parser.error("--overlap must satisfy 0 <= overlap < --chunk-size.")

    json_path = args.output_dir / "corpus.json"
    text_path = args.output_dir / "chunks.txt"
    if json_path.exists() or text_path.exists():
        parser.error(
            "Export files already exist. Choose a new --output-dir or "
            "remove the old export deliberately before regenerating."
        )

    pages = parse_pdf(args.pdf)
    chunks = chunk_pages(
        pages,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    if not chunks:
        parser.error("No text chunks extracted. This script does not perform OCR.")

    records = []
    sections = []
    for chunk in chunks:
        # IDs are scoped to this PDF and these chunking settings.
        chunk_id = f"p{chunk.page_number:03d}-c{chunk.chunk_number:03d}"
        records.append({"id": chunk_id, **asdict(chunk)})
        sections.append(
            f"{'=' * 72}\n"
            f"ID: {chunk_id}\n"
            f"Source: {chunk.source}\n"
            f"Page: {chunk.page_number} | Chunk: {chunk.chunk_number}\n"
            f"Characters: [{chunk.start_char}, {chunk.end_char})\n"
            f"{'-' * 72}\n"
            f"{chunk.text}\n"
        )

    manifest = {
        "schema_version": 1,
        "pdf_filename": args.pdf.name,
        "pdf_sha256": hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "chunks": records,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    text_path.write_text("\n".join(sections), encoding="utf-8")

    print(f"Exported {len(chunks)} chunks from {len(pages)} pages.")
    print(f"Read passages: {text_path}")
    print(f"Corpus manifest: {json_path}")
    print("Keep this export fixed while labeling and evaluating.")


if __name__ == "__main__":
    main()