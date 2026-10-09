"""Cross-encoder adapter tests without model downloads or inference."""

import sys
from types import ModuleType
from unittest.mock import Mock, sentinel

import numpy as np
import pytest

from ragforge.retrieval.cross_encoder import CrossEncoderScorer


@pytest.fixture
def fake_backend(monkeypatch):
    model = Mock()
    constructor = Mock(return_value=model)
    identity = Mock(return_value=sentinel.activation)

    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.CrossEncoder = constructor
    torch_nn = ModuleType("torch.nn")
    torch_nn.Identity = identity

    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)
    monkeypatch.setitem(sys.modules, "torch.nn", torch_nn)
    return constructor, model, identity


def test_constructor_uses_explicit_configuration(fake_backend):
    constructor, _, identity = fake_backend

    CrossEncoderScorer(
        "synthetic-model", device="cpu", batch_size=4, max_length=256
    )

    constructor.assert_called_once_with(
        "synthetic-model", device="cpu", max_length=256
    )
    identity.assert_called_once_with()


def test_prediction_preserves_pair_order_and_returns_raw_scores(fake_backend):
    _, model, _ = fake_backend
    model.predict.return_value = np.array([-2.5, 3.0], dtype=np.float32)
    scorer = CrossEncoderScorer(batch_size=4)
    passages = ["second passage", "first passage"]

    scores = scorer.score("query", passages)

    model.predict.assert_called_once_with(
        [("query", "second passage"), ("query", "first passage")],
        batch_size=4,
        show_progress_bar=False,
        activation_fn=sentinel.activation,
        apply_softmax=False,
        convert_to_numpy=True,
    )
    assert scores == [-2.5, 3.0]
    assert all(isinstance(score, float) for score in scores)
    assert passages == ["second passage", "first passage"]


def test_single_passage_output(fake_backend):
    _, model, _ = fake_backend
    model.predict.return_value = np.array([0.25], dtype=np.float64)

    assert CrossEncoderScorer().score("query", ["passage"]) == [0.25]


def test_empty_passages_skip_prediction(fake_backend):
    _, model, _ = fake_backend

    assert CrossEncoderScorer().score("query", []) == []
    model.predict.assert_not_called()


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"model_name": None}, TypeError),
        ({"model_name": " "}, ValueError),
        ({"revision": 123}, TypeError),
        ({"revision": False}, TypeError),
        ({"revision": ""}, ValueError),
        ({"revision": " "}, ValueError),
        ({"device": None}, TypeError),
        ({"device": ""}, ValueError),
        ({"batch_size": True}, TypeError),
        ({"batch_size": 1.5}, TypeError),
        ({"batch_size": 0}, ValueError),
        ({"batch_size": -1}, ValueError),
        ({"max_length": False}, TypeError),
        ({"max_length": "512"}, TypeError),
        ({"max_length": 0}, ValueError),
        ({"max_length": -1}, ValueError),
    ],
)
def test_invalid_configuration_rejected_before_model_loading(
    fake_backend, kwargs, error
):
    constructor, _, identity = fake_backend

    with pytest.raises(error):
        CrossEncoderScorer(**kwargs)

    constructor.assert_not_called()
    identity.assert_not_called()


@pytest.mark.parametrize(
    ("query", "error"),
    [
        (None, TypeError),
        (123, TypeError),
        ("", ValueError),
        (" \n", ValueError),
    ],
)
def test_invalid_query_skips_prediction(fake_backend, query, error):
    _, model, _ = fake_backend
    scorer = CrossEncoderScorer()

    with pytest.raises(error):
        scorer.score(query, ["passage"])

    model.predict.assert_not_called()


@pytest.mark.parametrize(
    ("passages", "error"),
    [
        ("passage", TypeError),
        (b"passage", TypeError),
        ([None], TypeError),
        ([123], TypeError),
        ([""], ValueError),
        ([" \t"], ValueError),
    ],
)
def test_invalid_passages_skip_prediction(fake_backend, passages, error):
    _, model, _ = fake_backend
    scorer = CrossEncoderScorer()

    with pytest.raises(error):
        scorer.score("query", passages)

    model.predict.assert_not_called()


@pytest.mark.parametrize("output", [None, [0.5]])
def test_non_numpy_output_rejected(fake_backend, output):
    _, model, _ = fake_backend
    model.predict.return_value = output

    with pytest.raises(TypeError, match="NumPy array"):
        CrossEncoderScorer().score("query", ["passage"])


@pytest.mark.parametrize(
    "output",
    [
        np.array(0.5),
        np.array([[0.5]]),
        np.array([0.5, 0.7]),
        np.array([]),
    ],
)
def test_wrong_output_shape_rejected(fake_backend, output):
    _, model, _ = fake_backend
    model.predict.return_value = output

    with pytest.raises(ValueError, match="one score per passage"):
        CrossEncoderScorer().score("query", ["passage"])


@pytest.mark.parametrize("dtype", [np.int64, np.bool_, np.complex64, object, np.str_])
def test_nonfloating_output_rejected(fake_backend, dtype):
    _, model, _ = fake_backend
    model.predict.return_value = np.array([1], dtype=dtype)

    with pytest.raises(TypeError, match="floating-point dtype"):
        CrossEncoderScorer().score("query", ["passage"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_output_rejected(fake_backend, value):
    _, model, _ = fake_backend
    model.predict.return_value = np.array([value], dtype=np.float64)

    with pytest.raises(ValueError, match="finite"):
        CrossEncoderScorer().score("query", ["passage"])


def test_prediction_exception_propagates_unchanged(fake_backend):
    _, model, _ = fake_backend
    error = RuntimeError("synthetic inference failure")
    model.predict.side_effect = error

    with pytest.raises(RuntimeError) as caught:
        CrossEncoderScorer().score("query", ["passage"])

    assert caught.value is error


def test_missing_dependency_has_install_guidance(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    with pytest.raises(ImportError, match="pip install"):
        CrossEncoderScorer()


def test_constructor_forwards_revision(fake_backend):
    constructor, _, _ = fake_backend
    CrossEncoderScorer("synthetic-model", revision="synthetic-sha", device="cuda")
    constructor.assert_called_once_with(
        "synthetic-model", revision="synthetic-sha", device="cuda", max_length=512
    )


def test_none_revision_preserves_backend_default(fake_backend):
    constructor, _, _ = fake_backend
    CrossEncoderScorer("synthetic-model", revision=None)
    constructor.assert_called_once_with(
        "synthetic-model", device="cpu", max_length=512
    )
