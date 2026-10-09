"""Download-free tests with explicit fake-clock idle timing."""

import importlib
import json
import random
import sys
from collections import Counter
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from test_benchmark_retrieval_e2e import benchmark as e2e_benchmark  # noqa: F401


@pytest.fixture
def idle(e2e_benchmark, monkeypatch):  # noqa: F811
    base = e2e_benchmark
    monkeypatch.delitem(sys.modules, "benchmark_retrieval_idle", raising=False)
    module = importlib.import_module("benchmark_retrieval_idle")
    chunks = base.loader.return_value[2]
    manifest = {
        "schema_version": 1,
        "chunk_count": len(chunks),
        "chunks": [
            {"id": f"p{index + 1:03d}-c001", **asdict(chunk)}
            for index, chunk in enumerate(chunks)
        ],
    }
    base.corpus.write_text(json.dumps(manifest), encoding="utf-8")
    now = [0.0]
    events = base.events

    def clock():
        events.append("clock")
        return now[0]

    def sleep(seconds):
        events.append(("sleep", seconds))
        now[0] += seconds + 0.005  # Simulated scheduling overshoot.

    def sync():
        events.append("sync")
        now[0] += 0.01

    original_search = base.hybrid.search.side_effect
    original_score = base.scorer.score.side_effect
    original_validation = base.module.candidate_ranking

    def search(*args, **kwargs):
        now[0] += 0.02
        return original_search(*args, **kwargs)

    def score(*args, **kwargs):
        now[0] += 0.03
        return original_score(*args, **kwargs)

    def validate(*args):
        result = original_validation(*args)
        now[0] += 0.2
        return result

    def synchronizer(device):
        if device == "cuda" and not base.torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        return sync if device == "cuda" else lambda: None

    base.hybrid.search.side_effect = search
    base.scorer.score.side_effect = score
    monkeypatch.setattr(base.module, "candidate_ranking", validate)
    monkeypatch.setattr(base.module, "device_synchronizer", synchronizer)
    monkeypatch.setattr(module, "perf_counter", clock)
    monkeypatch.setattr(module, "sleep", sleep)
    monkeypatch.setattr(module, "version", lambda name: "synthetic version")
    monkeypatch.setattr(sys, "argv", [
        "benchmark_retrieval_idle.py",
        "--corpus", str(base.corpus),
        "--output", str(base.output),
        "--reranker-device", "cuda",
        "--idle-seconds", "0", "1",
        "--trials", "2",
        "--warmup-queries", "1",
        "--seed", "42",
    ])
    yield SimpleNamespace(
        module=module, base=base, manifest=manifest, events=events,
    )
    sys.modules.pop("benchmark_retrieval_idle", None)


def report(idle):
    return json.loads(idle.base.output.read_text(encoding="utf-8"))


def test_latency_excludes_sleep_and_validation(idle):
    original = idle.base.corpus.read_bytes()
    idle.module.main()
    data = report(idle)

    # First + warmup + reference + (conditioning + measured) * four.
    assert idle.base.hybrid.search.call_count == 11
    assert idle.base.scorer.score.call_count == 11
    assert len(data["samples"]) == 4
    for sample in data["samples"]:
        # Search 20 ms + score 30 ms + final synchronization 10 ms.
        assert sample["end_to_end_ms"] == pytest.approx(60.0)
        assert sample["conditioning_end_to_end_ms"] == pytest.approx(60.0)
        gap = sample["requested_idle_seconds"]
        overshoot = 0.005 if gap else 0.0
        assert sample["sleep_window_seconds"] == pytest.approx(gap + overshoot)
        # Both conditioning validations occur after its end clock.
        assert sample["application_idle_seconds"] == pytest.approx(
            0.4 + gap + overshoot
        )
        assert sample["candidate_count"] == 6
        assert len(sample["top_k_chunk_ids"]) == 5
        assert sample["matches_warm_reference"] is True
    assert data["first_query_after_setup"]["end_to_end_ms"] == pytest.approx(60)
    assert data["warm_reference"]["end_to_end_ms"] == pytest.approx(60)
    assert idle.base.corpus.read_bytes() == original
    idle.base.loader.assert_not_called()
    assert "quality_first_measured_pass" not in data


def test_cuda_event_boundaries_and_conditioning(idle):
    idle.module.main()
    events = idle.events
    searches = [
        i for i, event in enumerate(events)
        if isinstance(event, tuple) and event[0] == "search"
    ]
    query = idle.module.DEFAULT_QUERY
    for index in searches:
        assert events[index - 1:index + 6] == [
            "clock", ("search", query), ("score", query),
            "sync", "clock", "validate", "validate",
        ]
    sleeps = [
        i for i, event in enumerate(events)
        if isinstance(event, tuple) and event[0] == "sleep"
    ]
    assert len(sleeps) == 2
    for index in sleeps:
        assert events[index - 4:index] == [
            "clock", "validate", "validate", "clock",
        ]
        assert events[index + 1:index + 3] == [
            "clock", ("search", query),
        ]
    assert len(searches) == 3 + 2 * len(report(idle)["samples"])


def test_schedule_counts_and_reproducibility(idle, monkeypatch):
    idle.module.main()
    first = report(idle)
    schedule = [
        (sample["trial"], sample["requested_idle_seconds"])
        for sample in first["samples"]
    ]
    assert Counter(gap for _, gap in schedule) == {0.0: 2, 1.0: 2}
    rng = random.Random(42)
    expected = []
    for trial in (1, 2):
        gaps = [0.0, 1.0]
        rng.shuffle(gaps)
        expected.extend((trial, gap) for gap in gaps)
    assert schedule == expected
    assert [s["sequence"] for s in first["samples"]] == [1, 2, 3, 4]
    for summary in first["per_gap"]:
        assert summary["sample_count"] == 2
        assert summary["end_to_end"]["mean_ms"] == pytest.approx(60)
        assert summary["warm_reference_order_agreement"] == 1.0

    second_output = idle.base.output.with_name("second.json")
    monkeypatch.setattr(sys, "argv", [
        *sys.argv, "--output", str(second_output),
    ])
    idle.module.main()
    second = json.loads(second_output.read_text(encoding="utf-8"))
    assert schedule == [
        (sample["trial"], sample["requested_idle_seconds"])
        for sample in second["samples"]
    ]


def test_cpu_timing_and_devices(idle, monkeypatch):
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--reranker-device", "cpu"])
    idle.module.main()
    data = report(idle)
    assert "sync" not in idle.events
    assert all(
        sample["end_to_end_ms"] == pytest.approx(50)
        for sample in data["samples"]
    )
    assert data["environment"]["gpu_name"] is None
    assert idle.base.factories[0].call_args.kwargs["device"] == "cpu"
    assert idle.base.factories[2].call_args.kwargs["device"] == "cpu"


@pytest.mark.parametrize("target", ["output", "corpus"])
def test_existing_files_are_protected(idle, monkeypatch, target):
    path = getattr(idle.base, target)
    if target == "output":
        path.write_bytes(b"preserve existing report")
    original = path.read_bytes()
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--output", str(path)])
    with pytest.raises(SystemExit) as caught:
        idle.module.main()
    assert caught.value.code == 2
    assert path.read_bytes() == original
    for factory in idle.base.factories:
        factory.assert_not_called()


@pytest.mark.parametrize("arguments", [
    ["--idle-seconds", "-1"],
    ["--idle-seconds", "nan"],
    ["--idle-seconds", "inf"],
    ["--idle-seconds", "1", "1"],
    ["--trials", "0"],
    ["--warmup-queries", "0"],
    ["--query", "   "],
])
def test_invalid_cli_prevents_loading(idle, monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", [*sys.argv, *arguments])
    with pytest.raises(SystemExit) as caught:
        idle.module.main()
    assert caught.value.code == 2
    assert not idle.base.output.exists()
    for factory in idle.base.factories:
        factory.assert_not_called()


def test_invalid_corpus_prevents_loading(idle):
    idle.manifest["chunk_count"] += 1
    idle.base.corpus.write_text(json.dumps(idle.manifest), encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        idle.module.main()
    assert caught.value.code == 2
    assert not idle.base.output.exists()
    for factory in idle.base.factories:
        factory.assert_not_called()


def test_unavailable_cuda_prevents_loading(idle):
    idle.base.torch.cuda.is_available = lambda: False
    with pytest.raises(SystemExit) as caught:
        idle.module.main()
    assert caught.value.code == 2
    assert not idle.base.output.exists()
    for factory in idle.base.factories:
        factory.assert_not_called()


@pytest.mark.parametrize("device,reranking_ms", [("cuda", 40.0), ("cpu", 30.0)])
def test_stage_timing_intervals(idle, monkeypatch, device, reranking_ms):
    monkeypatch.setattr(sys, "argv", [
        *sys.argv, "--stage-timing", "--reranker-device", device,
    ])
    idle.module.main()
    data = report(idle)
    assert data["report_schema_version"] == 2
    assert data["experiment"].endswith("_stage_timing")
    assert data["settings"]["stage_timing"] is True

    measurements = [
        data["first_query_after_setup"], data["warm_reference"], *data["samples"],
    ]
    for measurement in measurements:
        assert measurement["retrieval_ms"] == pytest.approx(20.0)
        assert measurement["reranking_ms"] == pytest.approx(reranking_ms)
        assert measurement["end_to_end_ms"] == pytest.approx(
            measurement["retrieval_ms"] + measurement["reranking_ms"]
        )
    for sample in data["samples"]:
        assert sample["conditioning_retrieval_ms"] == pytest.approx(20.0)
        assert sample["conditioning_reranking_ms"] == pytest.approx(reranking_ms)
        gap = sample["requested_idle_seconds"]
        assert sample["application_idle_seconds"] == pytest.approx(
            0.4 + gap + (0.005 if gap else 0.0)
        )
    for summary in data["per_gap"]:
        assert summary["retrieval"]["mean_ms"] == pytest.approx(20.0)
        assert summary["reranking"]["mean_ms"] == pytest.approx(reranking_ms)

    query = idle.module.DEFAULT_QUERY
    searches = [
        i for i, event in enumerate(idle.events)
        if isinstance(event, tuple) and event[0] == "search"
    ]
    expected = ["clock", ("search", query), "clock", ("score", query)]
    if device == "cuda":
        expected.append("sync")
    expected.extend(["clock", "validate", "validate"])
    for index in searches:
        assert idle.events[index - 1:index - 1 + len(expected)] == expected


def test_default_report_has_no_stage_measurements(idle):
    idle.module.main()
    data = report(idle)
    assert data["report_schema_version"] == 1
    assert data["experiment"] == "controlled_application_idle_hybrid_reranking"
    assert data["settings"]["stage_timing"] is False
    for measurement in [
        data["first_query_after_setup"], data["warm_reference"], *data["samples"],
    ]:
        assert "retrieval_ms" not in measurement
        assert "reranking_ms" not in measurement
    for summary in data["per_gap"]:
        assert "retrieval" not in summary
        assert "reranking" not in summary


def test_prediction_timing_requires_stage_timing(idle, monkeypatch):
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--prediction-timing"])
    with pytest.raises(SystemExit) as caught:
        idle.module.main()
    assert caught.value.code == 2
    for factory in idle.base.factories:
        factory.assert_not_called()


@pytest.mark.parametrize("device,reranking_ms", [("cuda", 40.0), ("cpu", 30.0)])
def test_prediction_timing_report_and_restoration(
    idle, monkeypatch, device, reranking_ms,
):
    monkeypatch.setattr(sys, "argv", [
        *sys.argv, "--stage-timing", "--prediction-timing",
        "--reranker-device", device,
    ])
    backend = idle.base.scorer._model
    original_score = idle.base.scorer.score.side_effect

    def placement(device):
        return backend

    def preprocess(passages):
        return passages

    def predict(query, passages):
        backend.to(device)
        backend.preprocess(passages)
        return original_score(query, passages)

    monkeypatch.setattr(backend, "to", placement, raising=False)
    monkeypatch.setattr(backend, "preprocess", preprocess, raising=False)
    monkeypatch.setattr(backend, "predict", predict, raising=False)
    idle.base.scorer.score.side_effect = (
        lambda query, passages: backend.predict(query, passages)
    )
    idle.module.main()
    data = report(idle)
    assert data["report_schema_version"] == 3
    assert data["experiment"].endswith("_prediction_timing")
    assert data["settings"]["prediction_timing"] is True
    assert "benchmark_prediction_timing.py" in data["source_sha256"]
    assert backend.to is placement
    assert backend.preprocess is preprocess
    assert backend.predict is predict

    for measurement in [
        data["first_query_after_setup"], data["warm_reference"], *data["samples"],
    ]:
        assert measurement["retrieval_ms"] == pytest.approx(20.0)
        assert measurement["reranking_ms"] == pytest.approx(reranking_ms)
        timing = measurement["prediction"]
        assert timing["predict_ms"] == pytest.approx(30.0)
        assert timing["device_placement_ms"] == pytest.approx(0.0)
        assert timing["preprocess_ms"] == pytest.approx(0.0)
        assert timing["predict_other_ms"] == pytest.approx(30.0)
        assert timing["predict_calls"] == 1
        assert timing["device_placement_calls"] == 1
        assert timing["preprocess_calls"] == 1
    for sample in data["samples"]:
        assert sample["conditioning_prediction"]["predict_ms"] == pytest.approx(30)
        gap = sample["requested_idle_seconds"]
        assert sample["application_idle_seconds"] == pytest.approx(
            0.4 + gap + (0.005 if gap else 0.0)
        )
        assert sample["matches_warm_reference"] is True
    for summary in data["per_gap"]:
        assert summary["prediction"]["predict_ms"]["mean_ms"] == pytest.approx(30)
        assert summary["prediction"]["predict_other_ms"]["mean_ms"] == pytest.approx(30)


def test_default_report_has_no_prediction_measurements(idle):
    idle.module.main()
    data = report(idle)
    assert data["report_schema_version"] == 1
    assert data["settings"]["prediction_timing"] is False
    for measurement in [
        data["first_query_after_setup"], data["warm_reference"], *data["samples"],
    ]:
        assert "prediction" not in measurement
        assert "conditioning_prediction" not in measurement
    for summary in data["per_gap"]:
        assert "prediction" not in summary
