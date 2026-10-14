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


@pytest.mark.parametrize(
    ("status_output", "expected_dirty"),
    [("", False), (" M scripts/evaluate_retrieval.py\n", True)],
)
def test_run_metadata_records_git_runtime_and_script(
    monkeypatch, status_output, expected_dirty
):
    runner = Mock(side_effect=[
        SimpleNamespace(stdout="a" * 40 + "\n"),
        SimpleNamespace(stdout=status_output),
    ])
    monkeypatch.setattr(evaluator.subprocess, "run", runner)
    monkeypatch.setattr(evaluator.platform, "python_version", lambda: "3.11.test")
    monkeypatch.setattr(evaluator, "version", lambda name: "test-version")

    provenance, packages = evaluator.collect_run_metadata()

    assert provenance["git_commit"] == "a" * 40
    assert provenance["git_working_tree_dirty"] is expected_dirty
    assert provenance["git_metadata_status"] == "available"
    assert provenance["git_metadata_error"] is None
    assert provenance["python_version"] == "3.11.test"
    assert provenance["evaluator_sha256"] == hashlib.sha256(
        SCRIPT_PATH.read_bytes()
    ).hexdigest()
    assert provenance["captured_at_utc"]
    assert packages == {
        name: "test-version"
        for name in ("sentence-transformers", "transformers", "torch", "numpy")
    }
    assert runner.call_count == 2
    assert runner.call_args_list[0].args[0] == ["git", "rev-parse", "HEAD"]
    assert runner.call_args_list[1].args[0] == [
        "git", "status", "--porcelain=v1", "--untracked-files=normal"
    ]
    for call in runner.call_args_list:
        assert call.kwargs["cwd"] == SCRIPT_PATH.resolve().parent
        assert call.kwargs["check"] is True
        assert call.kwargs["capture_output"] is True
        assert call.kwargs["timeout"] == 5


@pytest.mark.parametrize("failure_at", [0, 1])
@pytest.mark.parametrize("failure_kind", ["missing", "command", "timeout"])
def test_run_metadata_handles_unavailable_git(
    monkeypatch, failure_at, failure_kind
):
    failures = {
        "missing": FileNotFoundError("git unavailable"),
        "command": evaluator.subprocess.CalledProcessError(128, ["git"]),
        "timeout": evaluator.subprocess.TimeoutExpired(["git"], 5),
    }
    failure = failures[failure_kind]
    responses = [SimpleNamespace(stdout="a" * 40)] * failure_at
    runner = Mock(side_effect=responses + [failure])
    monkeypatch.setattr(evaluator.subprocess, "run", runner)
    monkeypatch.setattr(evaluator, "version", lambda name: "test-version")

    provenance, packages = evaluator.collect_run_metadata()

    assert provenance["git_commit"] is None
    assert provenance["git_working_tree_dirty"] is None
    assert provenance["git_metadata_status"] == "unavailable"
    assert provenance["git_metadata_error"] == type(failure).__name__
    assert packages["torch"] == "test-version"


def test_run_metadata_marks_missing_package_explicitly(monkeypatch):
    monkeypatch.setattr(
        evaluator.subprocess,
        "run",
        Mock(side_effect=[
            SimpleNamespace(stdout="a" * 40),
            SimpleNamespace(stdout=""),
        ]),
    )

    def package_version(name):
        if name == "torch":
            raise evaluator.PackageNotFoundError(name)
        return "test-version"

    monkeypatch.setattr(evaluator, "version", package_version)

    _, packages = evaluator.collect_run_metadata()

    assert packages["torch"] is None
    assert packages["numpy"] == "test-version"


def test_main_serializes_metadata_before_model_construction(
    monkeypatch, tmp_path, input_documents
):
    import sys

    original_manifest, dataset = input_documents
    manifest = dict(original_manifest)
    manifest.update({
        "pdf_sha256": "synthetic-pdf-hash",
        "chunk_size": 1000,
        "overlap": 150,
    })
    corpus = write_json(tmp_path / "corpus.json", manifest)
    questions = write_json(tmp_path / "questions.json", dataset)
    output = tmp_path / "report.json"
    events = []
    provenance = {
        "git_commit": "a" * 40,
        "git_working_tree_dirty": False,
        "git_metadata_status": "available",
    }
    packages = {"torch": None, "numpy": "test-version"}

    def metadata():
        events.append("metadata")
        return provenance, packages

    def embedder(*args, **kwargs):
        events.append("model")
        assert events == ["metadata", "model"]
        assert kwargs["revision"] == evaluator.MODEL_REVISION
        return object()

    retriever = Mock()
    retriever.search.return_value = []
    monkeypatch.setattr(evaluator, "collect_run_metadata", metadata)
    monkeypatch.setattr(evaluator, "SentenceTransformerEmbedder", embedder)
    for name in ("BM25Retriever", "DenseRetriever", "HybridRetriever"):
        monkeypatch.setattr(evaluator, name, Mock(return_value=retriever))
    monkeypatch.setattr(sys, "argv", [
        "evaluate_retrieval.py",
        "--corpus", str(corpus),
        "--questions", str(questions),
        "--output", str(output),
    ])

    evaluator.main()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["provenance"] == provenance
    assert report["package_versions"] == packages
    assert report["report_schema_version"] == 1
    assert set(report["methods"]) == {"bm25", "dense", "hybrid"}
    assert report["corpus_sha256"] == hashlib.sha256(
        corpus.read_bytes()
    ).hexdigest()
    assert report["questions_sha256"] == hashlib.sha256(
        questions.read_bytes()
    ).hexdigest()
    for metrics in report["methods"].values():
        assert metrics["hit_rate_at_k"] == 0.0
        assert metrics["mrr_at_k"] == 0.0
    assert events == ["metadata", "model"]
