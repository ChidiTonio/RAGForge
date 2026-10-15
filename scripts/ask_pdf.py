"""Ask questions about one PDF using hybrid retrieval and local Ollama."""

import argparse
import time
from pathlib import Path

from ragforge.application import build_pdf_session
from ragforge.generation.core import InvalidLLMResponse
from ragforge.generation.ollama import OllamaError


def _report_progress(message: str) -> None:
    """Display backend progress in the terminal."""
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Path to a text-based PDF")
    parser.add_argument(
        "question", nargs="?", help="Question to answer; omit for an interactive session"
    )
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument(
        "--point-count",
        type=int,
        default=None,
        metavar="N",
        help="Require exactly N points in supported answers; applies to every question in this session. Abstentions remain allowed.",
    )
    parser.add_argument(
        "--embedding-device",
        default="cpu",
        help="Default: cpu, leaving GPU memory available for Ollama",
    )
    parser.add_argument(
        "--cache-dir", type=Path, help="Enable persistent document indexes in this directory"
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild and replace the index; requires --cache-dir",
    )
    parser.add_argument(
        "--embedding-revision",
        default=None,
        help="Embedding model revision; use a commit hash for reproducibility",
    )
    args = parser.parse_args()
    if args.point_count is not None and args.point_count <= 0:
        parser.error("--point-count must be greater than zero")
    if args.rebuild_index and args.cache_dir is None:
        parser.error("--rebuild-index requires --cache-dir")
    if args.embedding_revision is not None and (not args.embedding_revision.strip()):
        parser.error("--embedding-revision must not be blank")
    try:
        session = build_pdf_session(
            args.pdf,
            model=args.model,
            embedding_device=args.embedding_device,
            cache_dir=args.cache_dir,
            rebuild_index=args.rebuild_index,
            embedding_revision=args.embedding_revision,
            use_paper_profile=args.pdf.name == "paper_434b99d68f495a91946c64b36b4602f9.pdf",
            report=_report_progress,
        )
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")

    def answer_question(question: str) -> None:
        started = time.perf_counter()
        try:
            answer = session.answer(question, point_count=args.point_count)
        except (OllamaError, InvalidLLMResponse, ValueError) as exc:
            print(f"\nQuestion failed: {exc}", flush=True)
            return
        print(f"\nSTATUS: {answer.status}")
        print(f"ANSWER: {answer.text}")
        print(f"Question time: {time.perf_counter() - started:.2f}s")
        print(f"Evidence chunks supplied: {len(answer.evidence)}")
        print("\nCITED PASSAGES:")
        if not answer.citations:
            print("(none)")
        for item in answer.citations:
            chunk = item.chunk
            print(
                f"\n[{item.label}] {chunk.source} | PDF page {chunk.page_number} | chunk {chunk.chunk_number} | characters {chunk.start_char}:{chunk.end_char}"
            )
            print(chunk.text)
        print("\nALL SUPPLIED EVIDENCE:")
        for item in answer.evidence:
            chunk = item.chunk
            print(f"\n[{item.label}] {chunk.source} | PDF page {chunk.page_number}")
            print(chunk.text)

    if args.question is not None:
        answer_question(args.question)
        return
    if args.point_count is not None:
        print(f"Supported answers must contain exactly {args.point_count} point(s).", flush=True)
    print("\nEnter a question, or type 'exit' to stop.")
    while True:
        try:
            question = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            answer_question(question)


if __name__ == "__main__":
    main()
