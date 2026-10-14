"""Exercise index persistence without PDFs or real embedding models."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from ragforge import storage
from ragforge.ingestion.chunking import TextChunk
from ragforge.retrieval.hybrid import HybridRetriever


@pytest.fixture
def data():
    chunks = [
        TextChunk("paper.pdf", 1, 1, 3, 8, "alpha"),
        TextChunk("paper.pdf", 2, 1, 0, 4, "beta"),
    ]
    vectors = np.array([[3, 4], [1, 0]], dtype=np.float32)
    metadata = {
        "pdf_sha256": "a" * 64,
        "source": "paper.pdf",
        "ingestion_version": "test-v1",
        "chunk_size": 800,
        "overlap": 120,
        "embedding_model": "test-model",
        "embedding_revision": "test-revision",
    }
    return chunks, vectors, metadata


def write(path, data):
    chunks, vectors, metadata = data
    storage.save_index(
        path, chunks=chunks, document_embeddings=vectors, compatibility=metadata
    )


def test_roundtrip_preserves_chunks_order_offsets_and_raw_vectors(tmp_path, data):
    path = tmp_path / "nested" / "index.npz"
    write(path, data)
    loaded = storage.load_index(path, compatibility=data[2])

    assert loaded.chunks == tuple(data[0])
    assert loaded.document_embeddings.dtype == np.float32
    np.testing.assert_array_equal(loaded.document_embeddings, data[1])
    data[1][:] = 0
    assert loaded.document_embeddings[0].tolist() == [3, 4]


def test_loaded_hybrid_index_matches_fresh_without_document_embedding(tmp_path, data):
    class Embedder:
        def __init__(self, outputs):
            self.outputs = list(outputs)
            self.calls = []

        def embed(self, texts):
            self.calls.append(list(texts))
            return self.outputs.pop(0)

    path = tmp_path / "index.npz"
    write(path, data)
    loaded = storage.load_index(path, compatibility=data[2])
    query = np.array([[1, 0]], dtype=np.float32)
    fresh_embedder = Embedder([data[1], query])
    loaded_embedder = Embedder([query])
    fresh = HybridRetriever(data[0], embedder=fresh_embedder)
    restored = HybridRetriever(
        loaded.chunks,
        embedder=loaded_embedder,
        document_embeddings=loaded.document_embeddings,
    )

    assert restored.search("alpha") == fresh.search("alpha")
    assert loaded_embedder.calls == [["alpha"]]
    assert len(fresh_embedder.calls) == 2


@pytest.mark.parametrize(
    "field",
    ["pdf_sha256", "source", "ingestion_version", "chunk_size",
     "overlap", "embedding_model", "embedding_revision"],
)
def test_changed_compatibility_is_rejected(tmp_path, data, field):
    path = tmp_path / "index.npz"
    write(path, data)
    changed = dict(data[2], **{field: "changed"})
    with pytest.raises(storage.InvalidIndexCache, match="compatibility"):
        storage.load_index(path, compatibility=changed)


@pytest.mark.parametrize(
    "damage",
    ["version", "offset", "boolean_page", "duplicate", "extra_field",
     "rows", "dtype", "nan", "zero", "object", "entries"],
)
def test_malformed_archive_is_rejected(tmp_path, data, damage):
    path = tmp_path / "index.npz"
    write(path, data)
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(archive["manifest"].item())
        vectors = archive["document_embeddings"].copy()

    if damage == "version":
        manifest["format_version"] = 999
    elif damage == "offset":
        manifest["chunks"][0]["end_char"] += 1
    elif damage == "boolean_page":
        manifest["chunks"][0]["page_number"] = True
    elif damage == "duplicate":
        manifest["chunks"][1] = dict(manifest["chunks"][0])
    elif damage == "extra_field":
        manifest["chunks"][0]["extra"] = "unexpected"
    elif damage == "rows":
        vectors = vectors[:1]
    elif damage == "dtype":
        vectors = vectors.astype(np.float64)
    elif damage == "nan":
        vectors[0, 0] = np.nan
    elif damage == "zero":
        vectors[0] = 0
    elif damage == "object":
        vectors = vectors.astype(object)

    entries = {
        "manifest": np.array(json.dumps(manifest)),
        "document_embeddings": vectors,
    }
    if damage == "entries":
        entries["extra"] = np.array(1)
    np.savez(path, **entries)

    with pytest.raises(storage.InvalidIndexCache):
        storage.load_index(path, compatibility=data[2])


def test_non_archive_bytes_are_rejected(tmp_path, data):
    path = tmp_path / "index.npz"
    path.write_bytes(b"not an index archive")
    with pytest.raises(storage.InvalidIndexCache):
        storage.load_index(path, compatibility=data[2])


def test_missing_cache_propagates_file_not_found(tmp_path, data):
    with pytest.raises(FileNotFoundError):
        storage.load_index(tmp_path / "missing.npz", compatibility=data[2])


def test_invalid_save_leaves_existing_cache_unchanged(tmp_path, data):
    path = tmp_path / "index.npz"
    write(path, data)
    before = path.read_bytes()
    invalid = [replace(data[0][0], start_char=-1), data[0][1]]
    with pytest.raises(ValueError, match="offsets"):
        write(path, (invalid, data[1], data[2]))
    assert path.read_bytes() == before


def test_failed_replacement_preserves_cache_and_cleans_temporary_file(
    tmp_path, data, monkeypatch
):
    path = tmp_path / "index.npz"
    write(path, data)
    before = path.read_bytes()

    def fail_replace(*args):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(storage.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        write(path, data)

    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_successful_replacement_updates_vectors(tmp_path, data):
    path = tmp_path / "index.npz"
    write(path, data)
    replacement = np.array([[0, 1], [1, 0]], dtype=np.float32)
    write(path, (data[0], replacement, data[2]))
    loaded = storage.load_index(path, compatibility=data[2])
    np.testing.assert_array_equal(loaded.document_embeddings, replacement)
    assert list(tmp_path.iterdir()) == [path]


def test_hash_file_uses_content_not_filename(tmp_path):
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    content = b"synthetic file bytes"
    first.write_bytes(content)
    second.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert storage.hash_file(first) == storage.hash_file(second) == expected
    second.write_bytes(content + b"changed")
    assert storage.hash_file(second) != expected
