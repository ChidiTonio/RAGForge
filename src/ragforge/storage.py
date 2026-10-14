"""Versioned, validated storage for reusable document indexes.

Compatibility metadata is supplied by the caller and compared exactly.
It must identify source content, ingestion settings/version, and embedding
configuration. Retrieval and generation settings need not be included.

Loading never uses pickle. This is a local cache format, not a sandbox for
arbitrary untrusted archives or a guarantee against oversized inputs.
"""

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from numpy.typing import NDArray

from ragforge.ingestion.chunking import TextChunk
from ragforge.retrieval.bm25 import tokenize
from ragforge.retrieval.dense import _normalize_embeddings

FORMAT_VERSION = 1
_CHUNK_FIELDS = {
    "source", "page_number", "chunk_number", "start_char", "end_char", "text"
}


class InvalidIndexCache(ValueError):
    """A cache is malformed, unsupported, or incompatible."""


@dataclass(frozen=True)
class StoredIndex:
    """Ordered chunks and independently owned raw document embeddings."""

    chunks: tuple[TextChunk, ...]
    document_embeddings: NDArray[np.float32]


def hash_file(path: Path) -> str:
    """Return the SHA-256 digest of file bytes without parsing the PDF."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value: object) -> str:
    """Produce deterministic JSON, rejecting lossy/non-JSON inputs."""
    def validate(item):
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float:
            if not np.isfinite(item):
                raise ValueError("Metadata numbers must be finite.")
            return
        if type(item) is list:
            for child in item:
                validate(child)
            return
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise TypeError("Metadata keys must be strings.")
            for child in item.values():
                validate(child)
            return
        raise TypeError("Metadata must contain only JSON-compatible values.")

    validate(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)


def _compatibility_json(metadata: dict[str, object]) -> str:
    if type(metadata) is not dict or not metadata:
        raise ValueError("Compatibility metadata must be a nonempty dictionary.")
    return _json(metadata)


def _validated_chunks(records: object) -> tuple[TextChunk, ...]:
    if type(records) is not list or not records:
        raise ValueError("At least one chunk is required.")

    chunks = []
    for record in records:
        if type(record) is not dict or set(record) != _CHUNK_FIELDS:
            raise ValueError("Unexpected chunk fields.")
        if type(record["source"]) is not str or not record["source"].strip():
            raise ValueError("Chunk source must be a nonblank string.")
        if type(record["text"]) is not str or not tokenize(record["text"]):
            raise ValueError("Chunk text must contain at least one word token.")
        for field in ("page_number", "chunk_number", "start_char", "end_char"):
            if type(record[field]) is not int:
                raise ValueError("Chunk numbers and offsets must be integers.")
        if record["page_number"] <= 0 or record["chunk_number"] <= 0:
            raise ValueError("Page and chunk numbers must be positive.")
        if (
            record["start_char"] < 0
            or record["end_char"] - record["start_char"] != len(record["text"])
        ):
            raise ValueError("Invalid chunk offsets.")
        chunks.append(TextChunk(**record))

    result = tuple(chunks)
    if len(set(result)) != len(result):
        raise ValueError("Duplicate chunks are not allowed.")
    return result


def save_index(
    path: Path,
    *,
    chunks: Iterable[TextChunk],
    document_embeddings: NDArray[np.float32],
    compatibility: dict[str, object],
) -> None:
    """Validate and atomically replace one cache file.

    Validation happens before touching the destination. Filesystem failures
    propagate to the caller. Temporary files are cleaned up on failure.
    """
    compatibility_json = _compatibility_json(compatibility)
    batch = tuple(chunks)
    if any(not isinstance(chunk, TextChunk) for chunk in batch):
        raise TypeError("Every chunk must be a TextChunk.")
    records = [asdict(chunk) for chunk in batch]
    validated = _validated_chunks(records)

    # Share exactly the dense retriever's vector validation rules.
    _normalize_embeddings(document_embeddings, expected_rows=len(validated))
    vectors = document_embeddings.copy()
    manifest = _json({
        "format_version": FORMAT_VERSION,
        "compatibility": json.loads(compatibility_json),
        "chunks": records,
    })

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez(
                stream,
                manifest=np.array(manifest),
                document_embeddings=vectors,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_index(
    path: Path,
    *,
    compatibility: dict[str, object],
) -> StoredIndex:
    """Load a compatible index, or raise InvalidIndexCache.

    Missing files and filesystem access errors propagate unchanged. The
    caller decides whether to rebuild a missing or invalid cache.
    """
    expected = _compatibility_json(compatibility)
    try:
        with Path(path).open("rb") as stream, np.load(stream, allow_pickle=False) as archive:
            if sorted(archive.files) != ["document_embeddings", "manifest"]:
                raise ValueError("Unexpected cache entries.")
            encoded = archive["manifest"]
            if encoded.shape != () or encoded.dtype.kind != "U":
                raise ValueError("Manifest must be a scalar Unicode string.")
            manifest = json.loads(encoded.item())
            if type(manifest) is not dict or set(manifest) != {
                "format_version", "compatibility", "chunks"
            }:
                raise ValueError("Unexpected manifest fields.")
            if (
                type(manifest["format_version"]) is not int
                or manifest["format_version"] != FORMAT_VERSION
            ):
                raise ValueError("Unsupported cache format version.")
            if _compatibility_json(manifest["compatibility"]) != expected:
                raise ValueError("Cache compatibility metadata does not match.")
            chunks = _validated_chunks(manifest["chunks"])
            vectors = archive["document_embeddings"]
            _normalize_embeddings(vectors, expected_rows=len(chunks))
            return StoredIndex(chunks, vectors.copy())
    except (ValueError, TypeError, KeyError, EOFError, BadZipFile) as exc:
        raise InvalidIndexCache(f"Invalid index cache: {exc}") from exc
