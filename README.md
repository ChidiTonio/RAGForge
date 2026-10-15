# RAGForge

Evidence-grounded research intelligence with a local PDF question-answering CLI.

## Setup

Requires Python 3.11+ and a separately running Ollama service.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,dense]'
```

In the environment where Ollama is installed, obtain the tested model:

```bash
ollama pull qwen2.5:7b
```

The CLI connects to http://127.0.0.1:11434. Its adapter does not read
OLLAMA_HOST. First use may download embedding and reranking models.

## Usage

From the repository root:

```bash
python scripts/ask_pdf.py "/absolute/path/document.pdf" \
    "Summarize the abstract." \
    --model qwen2.5:7b \
    --point-count 1 \
    --cache-dir data/indexes
```

Replace the PDF path with an existing text-based PDF. Quote paths with spaces.
In WSL, use paths such as /mnt/c/Users/... rather than Windows paths.

Omit the question for an interactive session; type exit or quit to stop.
The default generation model is qwen2.5:3b; live validation used qwen2.5:7b.

Supported answers contain numbered points with citations. The CLI prints
cited passages and all supplied evidence. Abstention remains allowed.
An explicit --point-count N enforces the supported answer count through
bounded validation retry; in interactive mode it applies to every question.
Natural-language count requests alone do not activate enforcement.

## Retrieval and caching

The PDF CLI uses BM25/dense hybrid retrieval and cross-encoder reranking.
Special section routing applies only to
paper_434b99d68f495a91946c64b36b4602f9.pdf; other filenames use generic
reranked retrieval. This is not a general section parser.

Persistent indexing is opt-in with --cache-dir. Compatible cache hits skip
PDF parsing and document embedding, but models still load each process.
The PDF must remain accessible for hashing.

Pin an immutable embedding commit with --embedding-revision for reproducible
cache use. Unpinned usage emits a warning. --rebuild-index requires --cache-dir.

See [index caching](docs/index-caching.md) for details.

## Validation and limitations

Run:

```bash
python -m ruff check .
python -m pytest -q
```

At the 2026-10-15 checkpoint, Ruff and all 588 tests passed.
Live qwen2.5:7b checks covered supported answers, multi-point output,
abstention, explicit counts, and two-process cache saving/reuse.

- No OCR: text-based PDFs only.
- Citation membership does not prove factual support or quotation fidelity.
- Count validation does not guarantee topic coverage or prose quality.
- Live checks are development checks, not held-out quality certification.
- Embedding pinning does not pin the reranker or Ollama model.
- Cache files and logs contain document text; protect them accordingly.
- data/ is ignored by Git; exclude other cache directories separately.
- Retrieved evidence is sent to the configured Ollama service.
- This checkpoint validates a local CLI, not a deployed web service.

See [reranking results](docs/reranking-experiment.md) and
[unresolved idle latency](docs/performance/idle-latency.md).
