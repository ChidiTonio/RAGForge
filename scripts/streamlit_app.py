"""Local browser interface for the shared PDF backend."""

import hashlib
import re
import tempfile
import time
from pathlib import Path

import streamlit as st

from ragforge.application import build_pdf_session
from ragforge.generation.core import InvalidLLMResponse
from ragforge.generation.ollama import OllamaError
from ragforge.ingestion.pdf import MAX_PDF_BYTES


def upload_filename(name: str) -> str:
    """Keep uploaded names readable without allowing directory traversal."""
    basename = name.replace("\\", "/").rsplit("/", 1)[-1]
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", basename).strip(".")
    if not safe:
        return "document.pdf"
    return safe[:120] if safe.lower().endswith(".pdf") else safe[:116] + ".pdf"


def display_passage(item) -> None:
    chunk = item.chunk
    st.caption(
        f"[{item.label}] {chunk.source} | PDF page {chunk.page_number} | "
        f"chunk {chunk.chunk_number} | "
        f"characters {chunk.start_char}:{chunk.end_char}"
    )
    st.text(chunk.text)


def display_answer(result) -> None:
    answer = result["answer"]
    st.subheader("Answer")
    st.write(f"Status: {answer.status}")
    st.text(answer.text)
    st.caption(
        f"Question time: {result['seconds']:.2f}s | "
        f"Evidence chunks supplied: {len(answer.evidence)}"
    )

    with st.expander("Cited passages", expanded=True):
        if not answer.citations:
            st.write("No cited passages.")
        for item in answer.citations:
            display_passage(item)

    with st.expander("All supplied evidence"):
        if not answer.evidence:
            st.write("No evidence supplied.")
        for item in answer.evidence:
            display_passage(item)


def main() -> None:
    st.set_page_config(page_title="RAGForge", page_icon="📄", layout="wide")
    st.title("RAGForge")
    st.caption("Ask evidence-grounded questions about one PDF using local Ollama.")
    st.info(
        "Text-based PDFs only; OCR is not supported. Generic retrieval is used "
        "for all uploads. Initial model downloads may require internet access."
    )

    with st.sidebar:
        st.header("Session settings")
        model = st.text_input("Ollama model", value="qwen2.5:3b").strip()
        revision = (
            st.text_input(
                "Embedding revision (optional)",
                help="An immutable commit hash improves reproducibility.",
            ).strip()
            or None
        )
        st.caption("Embeddings and reranker run on CPU.")
        if st.button("Clear prepared session"):
            for key in ("pdf_session", "session_key", "last_result"):
                st.session_state.pop(key, None)
        st.caption(
            "The upload widget may retain your file until you remove it. "
            "Temporary build files are deleted; no persistent PDF index is saved."
        )

    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded is not None and uploaded.size > MAX_PDF_BYTES:
        for key in ("pdf_session", "session_key", "last_result"):
            st.session_state.pop(key, None)
        st.error(
            f"PDF exceeds the upload limit of {MAX_PDF_BYTES // (1024 * 1024)} MiB. "
            "Please choose a smaller file."
        )
        return

    data = uploaded.getvalue() if uploaded is not None else None

    filename = upload_filename(uploaded.name) if uploaded is not None else None
    current_key = (
        (hashlib.sha256(data).hexdigest(), filename, model, revision) if data is not None else None
    )

    if st.session_state.get("session_key") != current_key:
        for key in ("pdf_session", "session_key", "last_result"):
            st.session_state.pop(key, None)

    if st.button(
        "Build PDF session",
        type="primary",
        disabled=not data or not model,
    ):
        for key in ("pdf_session", "session_key", "last_result"):
            st.session_state.pop(key, None)
        progress = st.empty()
        try:
            with (
                st.spinner("Preparing retrieval models and document index..."),
                tempfile.TemporaryDirectory(prefix="ragforge-") as directory,
            ):
                pdf = Path(directory) / filename
                pdf.write_bytes(data)
                session = build_pdf_session(
                    pdf,
                    model=model,
                    embedding_device="cpu",
                    embedding_revision=revision,
                    use_paper_profile=False,
                    report=progress.text,
                )
            st.session_state["pdf_session"] = session
            st.session_state["session_key"] = current_key
        except Exception as exc:  # noqa: BLE001 -- build-time UI error boundary
            # Keep ingestion/model failures readable at the UI boundary.
            progress.empty()
            st.error(f"Could not build the PDF session: {exc}")
        else:
            progress.empty()

    session = st.session_state.get("pdf_session")
    if session is None:
        st.caption("Choose a non-empty PDF, then build its session.")
        return

    st.success(f"Session ready: {filename}")
    with st.form("question_form"):
        question = st.text_area("Question")
        require_points = st.checkbox("Require an exact number of answer points")
        point_count = st.number_input(
            "Number of points", min_value=1, max_value=20, value=3, step=1
        )
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted:
        st.session_state.pop("last_result", None)
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            started = time.perf_counter()
            try:
                with st.spinner("Retrieving evidence and asking local Ollama..."):
                    answer = session.answer(
                        question.strip(),
                        point_count=int(point_count) if require_points else None,
                    )
                st.session_state["last_result"] = {
                    "answer": answer,
                    "seconds": time.perf_counter() - started,
                }
            except (OllamaError, InvalidLLMResponse, ValueError) as exc:
                st.error(f"Question failed: {exc}")

    if "last_result" in st.session_state:
        display_answer(st.session_state["last_result"])


if __name__ == "__main__":
    main()
