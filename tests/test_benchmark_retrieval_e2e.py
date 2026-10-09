"""Download-free tests for the direct end-to-end benchmark."""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ragforge.ingestion.chunking import TextChunk
from ragforge.retrieval.types import SearchResult


@pytest.fixture
def benchmark(monkeypatch, tmp_path):
    events = []
    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True),
        ),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: events.append("sync"),
            get_device_name=lambda: "synthetic GPU",
        ),
        version=SimpleNamespace(cuda="synthetic CUDA"),
        get_num_threads=lambda: 6,
        get_num_interop_threads=lambda: 12,
        set_num_threads=Mock(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    # Ensure this import uses the fake dependency, regardless of test order.
    monkeypatch.delitem(sys.modules, "benchmark_retrieval_e2e", raising=False)
    module = importlib.import_module("benchmark_retrieval_e2e")

    corpus = tmp_path / "corpus.json"
    questions = tmp_path / "questions.json"
    output = tmp_path / "report.json"
    corpus.write_bytes(b"synthetic corpus")
    questions.write_bytes(b"synthetic questions")
    chunks = [
        TextChunk(
            source="synthetic.pdf", page_number=index + 1,
            chunk_number=1, start_char=0, end_char=len(text), text=text,
        )
        for index in range(6)
        for text in [f"passage {index + 1}"]
    ]
    chunk_to_id = {
        chunk: f"p{index + 1:03d}-c001"
        for index, chunk in enumerate(chunks)
    }
    dataset = {
        "evaluation_role": "heldout",
        "questions": [
            {
                "id": f"q{index + 1:03d}",
                "question": f"query {index + 1}",
                "relevant_chunk_ids": ["p006-c001"],
            }
            for index in range(2)
        ],
    }
    loader = Mock(return_value=(
        {}, dataset, chunks, chunk_to_id, "corpus-hash", "questions-hash",
    ))
    monkeypatch.setattr(module, "load_inputs", loader)

    def search(query, *, top_k):
        events.append(("search", query))
        assert top_k == 20
        return [
            SearchResult(chunk=chunk, score=10.0 - index)
            for index, chunk in enumerate(chunks)
        ]

    def score(query, passages):
        events.append(("score", query))
        return [float(index) for index in range(len(passages))]

    hybrid = Mock()
    hybrid.search.side_effect = search
    scorer = Mock()
    scorer.score.side_effect = score
    scorer._model = SimpleNamespace(
        model=SimpleNamespace(
            parameters=lambda: iter([
                SimpleNamespace(dtype="torch.float32", device="synthetic device")
            ])
        )
    )
    factories = (
        Mock(),
        Mock(return_value=hybrid),
        Mock(return_value=scorer),
    )
    for name, factory in zip(
        ("SentenceTransformerEmbedder", "HybridRetriever", "CrossEncoderScorer"),
        factories, strict=True,
    ):
        monkeypatch.setattr(module, name, factory)

    original_validation = module.candidate_ranking

    def validate(*args):
        events.append("validate")
        return original_validation(*args)

    clock = iter(range(1000))

    def perf_counter():
        events.append("clock")
        return float(next(clock))

    monkeypatch.setattr(module, "candidate_ranking", validate)
    monkeypatch.setattr(module, "perf_counter", perf_counter)
    monkeypatch.setattr(module, "version", lambda name: "synthetic version")
    monkeypatch.setattr(module, "git_output", lambda *args: "synthetic git")
    monkeypatch.setattr(sys, "argv", [
        "benchmark_retrieval_e2e.py",
        "--corpus", str(corpus),
        "--questions", str(questions),
        "--output", str(output),
        "--reranker-device", "cpu",
        "--warmup-passes", "1",
        "--passes", "2",
    ])
    yield SimpleNamespace(
        module=module, output=output, corpus=corpus, questions=questions,
        dataset=dataset, loader=loader, events=events, factories=factories,
        hybrid=hybrid, scorer=scorer, torch=fake_torch,
    )
    # Avoid leaving a module containing the fake dependency cached.
    sys.modules.pop("benchmark_retrieval_e2e", None)


def read_report(benchmark):
    return json.loads(benchmark.output.read_text(encoding="utf-8"))


def test_fresh_requests_metrics_and_samples(benchmark):
    originals = [benchmark.corpus.read_bytes(), benchmark.questions.read_bytes()]
    benchmark.module.main()
    report = read_report(benchmark)

    # One first query + two warm-up queries + four measured queries.
    assert benchmark.hybrid.search.call_count == 7
    assert benchmark.scorer.score.call_count == 7
    assert report["timed_query_count"] == 4
    assert report["settings"]["warmup_queries"] == 2
    assert [sample["pass"] for sample in report["samples"]] == [1, 1, 2, 2]
    assert report["repeat_comparison_count"] == 2
    assert report["repeat_top_k_order_agreement"] == 1.0
    assert all(sample["candidate_count"] == 6 for sample in report["samples"])
    assert all(len(sample["top_k_chunk_ids"]) == 5 for sample in report["samples"])
    quality = report["quality_first_measured_pass"]
    assert quality["hybrid"]["hit_rate_at_k"] == 0.0
    assert quality["hybrid_reranked"]["hit_rate_at_k"] == 1.0
    assert quality["hybrid_reranked"]["mrr_at_k"] == 1.0
    assert report["latency"]["end_to_end"]["mean_ms"] == 1000.0
    assert report["latency"]["sequential_service_rate_qps"] == 1.0
    assert originals == [
        benchmark.corpus.read_bytes(), benchmark.questions.read_bytes()
    ]
    assert report["settings"]["matmul_allow_tf32"] is False
    assert report["settings"]["cudnn_allow_tf32"] is False


def test_cuda_timer_boundaries(benchmark, monkeypatch):
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--reranker-device", "cuda"])
    benchmark.module.main()
    events = benchmark.events
    searches = [
        index for index, event in enumerate(events)
        if isinstance(event, tuple) and event[0] == "search"
    ]
    assert len(searches) == 7
    for index in searches:
        query = events[index][1]
        assert events[index - 2:index + 6] == [
            "sync", "clock",
            ("search", query), ("score", query),
            "sync", "clock", "validate", "validate",
        ]
    report = read_report(benchmark)
    assert report["settings"]["cuda_synchronized"] is True
    assert report["environment"]["gpu_name"] == "synthetic GPU"
    assert benchmark.factories[0].call_args.kwargs["device"] == "cpu"
    assert benchmark.factories[2].call_args.kwargs["device"] == "cuda"


def test_repeat_changes_do_not_replace_first_pass_quality(benchmark):
    calls = 0

    def changing_score(query, passages):
        nonlocal calls
        calls += 1
        # Calls 6 and 7 are the second measured pass.
        scores = [float(index) for index in range(len(passages))]
        return scores if calls < 6 else list(reversed(scores))

    benchmark.scorer.score.side_effect = changing_score
    benchmark.module.main()
    report = read_report(benchmark)
    assert report["repeat_top_k_order_agreement"] == 0.0
    assert [
        sample["matches_first_measured_pass"] for sample in report["samples"]
    ] == [True, True, False, False]
    assert report["quality_first_measured_pass"]["hybrid_reranked"]["mrr_at_k"] == 1.0


def test_one_pass_has_no_repeat_comparisons(benchmark, monkeypatch):
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--passes", "1"])
    benchmark.module.main()
    report = read_report(benchmark)
    assert report["timed_query_count"] == 2
    assert report["repeat_comparison_count"] == 0
    assert report["repeat_top_k_order_agreement"] is None


@pytest.mark.parametrize("target", ["output", "corpus", "questions"])
def test_existing_files_are_protected(benchmark, monkeypatch, target):
    path = getattr(benchmark, target)
    if target == "output":
        path.write_bytes(b"preserve existing report")
    original = path.read_bytes()
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--output", str(path)])
    with pytest.raises(SystemExit) as caught:
        benchmark.module.main()
    assert caught.value.code == 2
    assert path.read_bytes() == original
    benchmark.loader.assert_not_called()
    for factory in benchmark.factories:
        factory.assert_not_called()


def test_invalid_inputs_prevent_model_loading(benchmark):
    benchmark.loader.side_effect = ValueError("synthetic invalid inputs")
    with pytest.raises(SystemExit) as caught:
        benchmark.module.main()
    assert caught.value.code == 2
    assert not benchmark.output.exists()
    for factory in benchmark.factories:
        factory.assert_not_called()


def test_unavailable_cuda_prevents_model_loading(benchmark, monkeypatch):
    benchmark.torch.cuda.is_available = lambda: False
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--reranker-device", "cuda"])
    with pytest.raises(SystemExit) as caught:
        benchmark.module.main()
    assert caught.value.code == 2
    assert not benchmark.output.exists()
    for factory in benchmark.factories:
        factory.assert_not_called()
