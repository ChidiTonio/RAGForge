"""Test evaluation validation and metrics without private data or models."""

import hashlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ragforge.ingestion.chunking import TextChunk

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_retrieval.py"
SPEC = importlib.util.spec_from_file_location("ragforge_eval_under_test", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load the retrieval evaluator.")
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


@pytest.fixture
def synthetic_chunks():
    return [
        TextChunk(
            source="synthetic.pdf",
            page_number=index,
            chunk_number=1,
            start_char=0,
            end_char=len(text),
            text=text,
        )
        for index, text in enumerate(["alpha evidence", "beta evidence"], start=1)
    ]


@pytest.fixture
def input_documents(synthetic_chunks):
    manifest = {
        "schema_version": 1,
        "pdf_filename": "synthetic.pdf",
        "chunk_count": len(synthetic_chunks),
        "chunks": [
            {"id": f"p{chunk.page_number:03d}-c001", **asdict(chunk)}
            for chunk in synthetic_chunks
        ],
    }
    dataset = {
        "source": "synthetic.pdf",
        "questions": [
            {
                "id": "synthetic-q001",
                "question": "Where is alpha evidence?",
                "relevant_chunk_ids": ["p001-c001"],
            }
        ],
    }
    return manifest, dataset


def write_json(path, document):
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_load_inputs_preserves_chunks_and_exact_file_hashes(
    tmp_path, input_documents, synthetic_chunks
):
    manifest, dataset = input_documents
    corpus_path = write_json(tmp_path / "corpus.json", manifest)
    questions_path = write_json(tmp_path / "questions.json", dataset)

    loaded = evaluator.load_inputs(corpus_path, questions_path)

    assert loaded[0] == manifest
    assert loaded[1] == dataset
    assert loaded[2] == synthetic_chunks
    assert loaded[3] == {
        synthetic_chunks[0]: "p001-c001",
        synthetic_chunks[1]: "p002-c001",
    }
    assert loaded[4] == hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    assert loaded[5] == hashlib.sha256(questions_path.read_bytes()).hexdigest()


@pytest.mark.parametrize("document", [[], None, "not an object", 42])
def test_load_json_rejects_non_objects(tmp_path, document):
    path = write_json(tmp_path / "invalid.json", document)
    with pytest.raises(TypeError, match="top-level JSON must be an object"):
        evaluator.load_json(path)


@pytest.mark.parametrize(
    ("case", "exception", "message"),
    [
        ("chunk_type", TypeError, "Chunk record 1 must be an object"),
        ("duplicate_chunk", ValueError, "Duplicate chunk ID"),
        ("offsets", ValueError, "text length does not match offsets"),
        ("question_type", TypeError, "Each question must be an object"),
        ("duplicate_question", ValueError, "Duplicate question ID"),
        ("unknown_label", ValueError, "unknown relevant IDs"),
        ("duplicate_label", ValueError, "duplicate relevance labels"),
    ],
)
def test_load_inputs_rejects_invalid_documents(
    tmp_path, input_documents, case, exception, message
):
    manifest, dataset = input_documents
    if case == "chunk_type":
        manifest["chunks"][0] = []
    elif case == "duplicate_chunk":
        manifest["chunks"][1] = dict(manifest["chunks"][0])
    elif case == "offsets":
        manifest["chunks"][0]["end_char"] += 1
    elif case == "question_type":
        dataset["questions"][0] = []
    elif case == "duplicate_question":
        dataset["questions"].append(dict(dataset["questions"][0]))
    elif case == "unknown_label":
        dataset["questions"][0]["relevant_chunk_ids"] = ["p999-c001"]
    elif case == "duplicate_label":
        dataset["questions"][0]["relevant_chunk_ids"] = ["p001-c001", "p001-c001"]

    corpus_path = write_json(tmp_path / "corpus.json", manifest)
    questions_path = write_json(tmp_path / "questions.json", dataset)

    with pytest.raises(exception, match=message):
        evaluator.load_inputs(corpus_path, questions_path)


def test_evaluate_computes_hits_and_first_relevant_reciprocal_rank(synthetic_chunks):
    alpha, beta = synthetic_chunks
    chunk_to_id = {alpha: "p001-c001", beta: "p002-c001"}
    questions = [
        {
            "id": f"synthetic-q{index:03d}",
            "question": f"synthetic query {index}",
            "relevant_chunk_ids": ["p001-c001"],
        }
        for index in range(1, 4)
    ]
    retriever = Mock()
    retriever.search.side_effect = [
        [SimpleNamespace(chunk=alpha, score=1.0)],
        [
            SimpleNamespace(chunk=beta, score=1.0),
            SimpleNamespace(chunk=alpha, score=0.5),
        ],
        [],
    ]

    metrics = evaluator.evaluate(retriever, questions, chunk_to_id)

    assert metrics["hit_rate_at_k"] == pytest.approx(2 / 3)
    assert metrics["mrr_at_k"] == pytest.approx(0.5)
    assert [row["first_relevant_rank"] for row in metrics["per_question"]] == [
        1, 2, None
    ]
    assert [row["hit_at_k"] for row in metrics["per_question"]] == [1, 1, 0]
    assert metrics["per_question"][1]["ranking"][0]["is_relevant"] is False
    assert metrics["per_question"][1]["ranking"][1]["is_relevant"] is True
    assert retriever.search.call_count == 3
    for query in questions:
        retriever.search.assert_any_call(query["question"], top_k=evaluator.TOP_K)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("duplicate", "duplicate chunks"),
        ("outside", "outside the corpus"),
        ("nan", "nonfinite score"),
        ("infinity", "nonfinite score"),
        ("too_many", "more than top_k"),
    ],
)
def test_evaluate_rejects_invalid_results(synthetic_chunks, case, message):
    alpha, beta = synthetic_chunks
    chunk_to_id = {alpha: "p001-c001"}
    questions = [{
        "id": "synthetic-q001",
        "question": "synthetic query",
        "relevant_chunk_ids": ["p001-c001"],
    }]
    result = SimpleNamespace(chunk=alpha, score=1.0)
    results = [result]
    if case == "duplicate":
        results = [result, result]
    elif case == "outside":
        results = [SimpleNamespace(chunk=beta, score=1.0)]
    elif case == "nan":
        results = [SimpleNamespace(chunk=alpha, score=float("nan"))]
    elif case == "infinity":
        results = [SimpleNamespace(chunk=alpha, score=float("inf"))]
    elif case == "too_many":
        results = [result] * (evaluator.TOP_K + 1)

    retriever = Mock()
    retriever.search.return_value = results

    with pytest.raises(ValueError, match=message):
        evaluator.evaluate(retriever, questions, chunk_to_id)
