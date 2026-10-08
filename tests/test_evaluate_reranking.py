"""Download-free tests for the fixed-candidate experiment script."""

import hashlib
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from ragforge.ingestion.chunking import TextChunk
from ragforge.retrieval.types import SearchResult


@pytest.fixture
def experiment(monkeypatch, tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    module = importlib.import_module("evaluate_reranking")

    corpus_path = tmp_path / "corpus.json"
    questions_path = tmp_path / "questions.json"
    output_path = tmp_path / "report.json"
    corpus_path.write_bytes(b"synthetic corpus fixture\n")
    questions_path.write_bytes(b"synthetic questions fixture\n")

    chunks = [
        TextChunk(
            source="synthetic.pdf",
            page_number=index + 1,
            chunk_number=1,
            start_char=0,
            end_char=len(text),
            text=text,
        )
        for index in range(6)
        for text in [f"passage {index + 1}"]
    ]
    chunk_to_id = {
        chunk: f"p{index + 1:03d}-c001"
        for index, chunk in enumerate(chunks)
    }
    dataset = {
        "dataset_id": "synthetic-development",
        "questions": [
            {
                "id": "q001",
                "question": "query one",
                "relevant_chunk_ids": ["p006-c001"],
            },
            {
                "id": "q002",
                "question": "query two",
                "relevant_chunk_ids": ["p006-c001"],
            },
        ],
    }
    manifest = {
        "pdf_filename": "synthetic.pdf",
        "pdf_sha256": "synthetic-pdf-hash",
        "chunk_size": 1000,
        "overlap": 150,
    }
    loader = Mock(return_value=(
        manifest,
        dataset,
        chunks,
        chunk_to_id,
        hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        hashlib.sha256(questions_path.read_bytes()).hexdigest(),
    ))
    monkeypatch.setattr(module, "load_inputs", loader)

    pools = {
        "query one": [
            SearchResult(chunk=chunk, score=10.0 - index)
            for index, chunk in enumerate(chunks)
        ],
        "query two": [
            SearchResult(chunk=chunk, score=10.0 - index)
            for index, chunk in enumerate(chunks[:5])
        ],
    }
    events = []

    def search(query, *, top_k):
        events.append(("search", query))
        assert top_k == 20
        return list(pools[query])

    def score(query, passages):
        events.append(("score", query))
        return [float(index) for index in range(len(passages))]

    hybrid = Mock()
    hybrid.search.side_effect = search
    scorer = Mock()
    scorer.score.side_effect = score
    embedder_factory = Mock()
    hybrid_factory = Mock(return_value=hybrid)
    scorer_factory = Mock(return_value=scorer)
    monkeypatch.setattr(module, "SentenceTransformerEmbedder", embedder_factory)
    monkeypatch.setattr(module, "HybridRetriever", hybrid_factory)
    monkeypatch.setattr(module, "CrossEncoderScorer", scorer_factory)
    monkeypatch.setattr(module, "version", lambda name: "synthetic-version")
    monkeypatch.setattr(sys, "argv", [
        "evaluate_reranking.py",
        "--corpus", str(corpus_path),
        "--questions", str(questions_path),
        "--output", str(output_path),
    ])

    return {
        "module": module,
        "corpus": corpus_path,
        "questions": questions_path,
        "output": output_path,
        "dataset": dataset,
        "loader": loader,
        "pools": pools,
        "events": events,
        "factories": (embedder_factory, hybrid_factory, scorer_factory),
    }


def test_fixed_pools_metrics_and_report(experiment):
    original_inputs = [
        experiment["corpus"].read_bytes(),
        experiment["questions"].read_bytes(),
    ]
    original_scores = {
        query: [result.score for result in pool]
        for query, pool in experiment["pools"].items()
    }

    experiment["module"].main()

    report = json.loads(experiment["output"].read_text(encoding="utf-8"))
    assert report["methods"]["hybrid"]["hit_rate_at_k"] == 0.0
    assert report["methods"]["hybrid"]["mrr_at_k"] == 0.0
    assert report["methods"]["hybrid_reranked"]["hit_rate_at_k"] == 0.5
    assert report["methods"]["hybrid_reranked"]["mrr_at_k"] == 0.5
    assert report["candidate_pool"]["hit_rate"] == 0.5
    assert report["candidate_pool"]["mean_relevant_label_coverage"] == 0.5

    first, second = report["paired_diagnostics"]
    assert first["candidate_count"] == 6
    assert first["hybrid_candidates"][-1]["chunk_id"] == "p006-c001"
    assert first["reranked_candidates"][0]["chunk_id"] == "p006-c001"
    assert first["hit_at_k_delta"] == 1
    assert second["candidate_count"] == 5
    assert second["candidate_pool_hit"] == 0
    assert second["hit_at_k_delta"] == 0

    for row in report["paired_diagnostics"]:
        assert {
            item["chunk_id"] for item in row["hybrid_candidates"]
        } == {
            item["chunk_id"] for item in row["reranked_candidates"]
        }
        assert row["retrieval_ms"] >= 0
        assert row["reranking_ms"] >= 0

    assert experiment["events"] == [
        ("search", "query one"),
        ("score", "query one"),
        ("search", "query one"),
        ("search", "query two"),
        ("score", "query one"),
        ("score", "query two"),
    ]
    assert original_scores == {
        query: [result.score for result in pool]
        for query, pool in experiment["pools"].items()
    }
    assert original_inputs == [
        experiment["corpus"].read_bytes(),
        experiment["questions"].read_bytes(),
    ]
    assert report["corpus_sha256"] == hashlib.sha256(original_inputs[0]).hexdigest()
    assert report["questions_sha256"] == hashlib.sha256(original_inputs[1]).hexdigest()
    assert report["settings"]["backend_candidate_k"] == 20
    assert report["settings"]["final_pool_k"] == 20
    assert report["settings"]["model_revision_pinned"] is False
    assert report["latency"]["combined"]["mean_ms"] == pytest.approx(
        report["latency"]["retrieval"]["mean_ms"]
        + report["latency"]["reranking"]["mean_ms"]
    )


def test_existing_report_is_not_overwritten(experiment):
    experiment["output"].write_bytes(b"preserve this report")

    with pytest.raises(SystemExit) as caught:
        experiment["module"].main()

    assert caught.value.code == 2
    assert experiment["output"].read_bytes() == b"preserve this report"
    experiment["loader"].assert_not_called()
    for factory in experiment["factories"]:
        factory.assert_not_called()


def test_input_validation_failure_prevents_model_loading(experiment):
    experiment["loader"].side_effect = ValueError("synthetic invalid input")

    with pytest.raises(SystemExit) as caught:
        experiment["module"].main()

    assert caught.value.code == 2
    assert not experiment["output"].exists()
    for factory in experiment["factories"]:
        factory.assert_not_called()


def test_duplicate_query_text_prevents_model_loading(experiment):
    experiment["dataset"]["questions"][1]["question"] = "query one"

    with pytest.raises(SystemExit) as caught:
        experiment["module"].main()

    assert caught.value.code == 2
    assert not experiment["output"].exists()
    for factory in experiment["factories"]:
        factory.assert_not_called()


def test_frozen_retriever_returns_a_separate_list(experiment):
    pool = experiment["pools"]["query one"]
    retriever = experiment["module"].FrozenRetriever({"query": pool})

    results = retriever.search("query", top_k=2)
    assert results == pool[:2]
    results.clear()
    assert len(pool) == 6


def test_latency_summary(experiment):
    summary = experiment["module"].latency_summary([1.0, 2.0, 3.0])

    assert summary["total_ms"] == 6.0
    assert summary["mean_ms"] == 2.0
    assert summary["median_ms"] == 2.0
    assert summary["p95_ms"] == pytest.approx(2.9)
    assert summary["min_ms"] == 1.0
    assert summary["max_ms"] == 3.0
