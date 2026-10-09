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
