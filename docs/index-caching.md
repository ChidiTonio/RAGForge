# Persistent PDF indexes

## Enable caching

Index caching is opt-in. From the project directory, run:

    python scripts/ask_pdf.py "/path/to/document.pdf" --cache-dir data/indexes

The first run parses the PDF, embeds usable chunks, and saves an index.
A compatible subsequent run restores chunks and document vectors, skipping
PDF parsing and document embedding.

The PDF must remain accessible: its bytes are hashed to determine cache
compatibility, including on a cache hit.

Without `--cache-dir`, the CLI uses its normal nonpersistent indexing path.

## Embedding revision and reproducibility

For persistent-cache usage, pin the embedding model to an immutable
Hugging Face model commit hash:

    python scripts/ask_pdf.py "/path/to/document.pdf" --cache-dir data/indexes --embedding-revision MODEL_COMMIT_HASH

Replace `MODEL_COMMIT_HASH` with a real commit hash for the configured
embedding model. This is a placeholder, not a runnable revision.
A moving branch such as `main` is not an immutable revision.

Without a revision, the CLI warns because an upstream model update may
make newly generated query vectors incompatible with cached document
vectors. Equal vector dimensions alone do not establish compatibility.

Pinning the embedding revision improves embedding/index reproducibility;
it does not by itself guarantee identical generated answers or pin the
cross-encoder and Ollama models.

## Force a rebuild

    python scripts/ask_pdf.py "/path/to/document.pdf" --cache-dir data/indexes --rebuild-index

When using a pinned revision, include the same `--embedding-revision`
argument when rebuilding.

`--rebuild-index` requires `--cache-dir`. It bypasses cache loading and
replaces the index for the selected compatibility configuration.

## Compatibility and recovery

Cache identity includes PDF content, source filename, ingestion settings,
embedding configuration, and selected dependency versions. Configuration
changes can select a new cache file; old cache files are not automatically
removed.

Missing, invalid, or unreadable caches trigger rebuilding. A loaded index
with a vector dimension different from the embedding model also triggers
rebuilding. If saving fails, the CLI warns and continues using the
in-memory index.

Index writes use the validated atomic persistence layer.

## Why models still load

Each new CLI process loads the embedding model for questions and the
cross-encoder for reranking. The document index cache does not persist
running model instances.

## Hugging Face authentication

Public model downloads may work without authentication, subject to rate
limits. Hugging Face authentication is optional for these public downloads
and may help if download rate limits are encountered.

Never commit authentication tokens or put them in project source files.

## Local cache files

`data/` is ignored by this repository, including `data/indexes/`.
If choosing another cache directory, ensure generated indexes are excluded
from version control as appropriate.

Cached chunks contain extracted document text. Treat cache files as
document data and protect them accordingly.
