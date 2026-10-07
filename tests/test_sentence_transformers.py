"""Test the embedding adapter without model downloads or optional dependencies."""

import builtins
import subprocess
import sys
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pytest

from ragforge.embeddings.sentence_transformers import SentenceTransformerEmbedder


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock]:
    """Replace the optional package with a controllable fake backend."""
    model = Mock()
    model.get_sentence_embedding_dimension.return_value = 3
    model.encode.return_value = np.array([[1, 2, 3]], dtype=np.float32)
    factory = Mock(return_value=model)

    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = factory
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    return factory, model


def test_module_import_does_not_load_optional_dependencies() -> None:
    # A fresh process prevents earlier imports or fake modules hiding a problem.
    code = (
        "import sys; "
        "import ragforge.embeddings.sentence_transformers; "
        "assert 'sentence_transformers' not in sys.modules; "
        "assert 'torch' not in sys.modules"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_loads_default_model_once(fake_backend: tuple[Mock, Mock]) -> None:
    factory, model = fake_backend

    embedder = SentenceTransformerEmbedder()
    embedder.embed(["first"])
    embedder.embed(["second"])

    factory.assert_called_once_with(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=None,
    )
    model.get_sentence_embedding_dimension.assert_called_once_with()
    assert model.encode.call_count == 2
    assert embedder.dimension == 3


def test_passes_configuration_and_preserves_order(
    fake_backend: tuple[Mock, Mock],
) -> None:
    factory, model = fake_backend
    model.encode.return_value = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    embedder = SentenceTransformerEmbedder(
        "test-model",
        device="cpu",
        batch_size=8,
    )

    vectors = embedder.embed(("second document", "first document"))

    factory.assert_called_once_with("test-model", device="cpu")
    model.encode.assert_called_once_with(
        ["second document", "first document"],
        batch_size=8,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    np.testing.assert_array_equal(vectors, [[1, 0, 0], [0, 1, 0]])
    assert vectors.dtype == np.dtype(np.float32)


def test_empty_batch_skips_encoding(fake_backend: tuple[Mock, Mock]) -> None:
    _, model = fake_backend
    embedder = SentenceTransformerEmbedder()

    vectors = embedder.embed([])

    assert vectors.shape == (0, 3)
    assert vectors.dtype == np.dtype(np.float32)
    model.encode.assert_not_called()


def test_converts_float64_without_normalizing(
    fake_backend: tuple[Mock, Mock],
) -> None:
    _, model = fake_backend
    original = np.array([[3, 4, 0]], dtype=np.float64)
    model.encode.return_value = original
    embedder = SentenceTransformerEmbedder()

    vectors = embedder.embed(["document"])

    assert vectors.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(vectors, [[3, 4, 0]])
    np.testing.assert_array_equal(original, [[3, 4, 0]])


@pytest.mark.parametrize("model_name", ["", " \n\t "])
def test_blank_model_name_is_rejected_before_loading(
    fake_backend: tuple[Mock, Mock],
    model_name: str,
) -> None:
    factory, _ = fake_backend

    with pytest.raises(ValueError, match="model_name"):
        SentenceTransformerEmbedder(model_name)

    factory.assert_not_called()


@pytest.mark.parametrize("model_name", [None, 123])
def test_nonstring_model_name_is_rejected(
    fake_backend: tuple[Mock, Mock],
    model_name: object,
) -> None:
    factory, _ = fake_backend

    with pytest.raises(TypeError, match="model_name"):
        SentenceTransformerEmbedder(model_name)  # type: ignore[arg-type]

    factory.assert_not_called()


@pytest.mark.parametrize("batch_size", [0, -1])
def test_nonpositive_batch_size_is_rejected(
    fake_backend: tuple[Mock, Mock],
    batch_size: int,
) -> None:
    factory, _ = fake_backend

    with pytest.raises(ValueError, match="batch_size"):
        SentenceTransformerEmbedder(batch_size=batch_size)

    factory.assert_not_called()


@pytest.mark.parametrize("batch_size", [True, False, 1.5, "8", None])
def test_noninteger_batch_size_is_rejected(
    fake_backend: tuple[Mock, Mock],
    batch_size: object,
) -> None:
    factory, _ = fake_backend

    with pytest.raises(TypeError, match="batch_size"):
        SentenceTransformerEmbedder(batch_size=batch_size)  # type: ignore[arg-type]

    factory.assert_not_called()


@pytest.mark.parametrize("dimension", [None, 0, -1, True, False, 3.5, "3"])
def test_invalid_model_dimension_is_rejected(
    fake_backend: tuple[Mock, Mock],
    dimension: object,
) -> None:
    _, model = fake_backend
    model.get_sentence_embedding_dimension.return_value = dimension

    with pytest.raises(ValueError, match="positive integer embedding dimension"):
        SentenceTransformerEmbedder()


def test_numpy_integer_dimension_is_accepted(
    fake_backend: tuple[Mock, Mock],
) -> None:
    _, model = fake_backend
    model.get_sentence_embedding_dimension.return_value = np.int64(3)

    embedder = SentenceTransformerEmbedder()

    assert embedder.dimension == 3
    assert isinstance(embedder.dimension, int)


@pytest.mark.parametrize(
    "texts",
    ["one string", b"bytes", None, 42, ["valid", 1], ["valid", None]],
)
def test_invalid_text_inputs_skip_encoding(
    fake_backend: tuple[Mock, Mock],
    texts: object,
) -> None:
    _, model = fake_backend
    embedder = SentenceTransformerEmbedder()

    with pytest.raises(TypeError, match="string"):
        embedder.embed(texts)  # type: ignore[arg-type]

    model.encode.assert_not_called()


@pytest.mark.parametrize(
    "output",
    [
        np.array([1, 2, 3], dtype=np.float32),
        np.ones((1, 1, 3), dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
        np.ones((1, 2), dtype=np.float32),
        np.empty((0, 3), dtype=np.float32),
    ],
)
def test_invalid_output_shape_is_rejected(
    fake_backend: tuple[Mock, Mock],
    output: np.ndarray,
) -> None:
    _, model = fake_backend
    model.encode.return_value = output
    embedder = SentenceTransformerEmbedder()

    with pytest.raises(ValueError, match="output shape"):
        embedder.embed(["document"])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_output_is_rejected(
    fake_backend: tuple[Mock, Mock],
    value: float,
) -> None:
    _, model = fake_backend
    model.encode.return_value = np.array([[value, 1, 0]], dtype=np.float32)
    embedder = SentenceTransformerEmbedder()

    with pytest.raises(ValueError, match="finite"):
        embedder.embed(["document"])


def test_zero_vector_in_batch_is_rejected(
    fake_backend: tuple[Mock, Mock],
) -> None:
    _, model = fake_backend
    model.encode.return_value = np.array([[1, 0, 0], [0, 0, 0]], dtype=np.float32)
    embedder = SentenceTransformerEmbedder()

    with pytest.raises(ValueError, match="nonzero"):
        embedder.embed(["valid document", "zero document"])


def test_model_encoding_error_is_preserved(
    fake_backend: tuple[Mock, Mock],
) -> None:
    _, model = fake_backend
    error = RuntimeError("Synthetic encoding failure")
    model.encode.side_effect = error
    embedder = SentenceTransformerEmbedder()

    with pytest.raises(RuntimeError) as caught:
        embedder.embed(["document"])

    assert caught.value is error


@pytest.mark.parametrize(
    "missing_name",
    ["sentence_transformers", "synthetic_transitive_dependency"],
)
def test_missing_dependency_errors(
    monkeypatch: pytest.MonkeyPatch,
    missing_name: str,
) -> None:
    original_import = builtins.__import__
    error = ModuleNotFoundError(
        f"No module named '{missing_name}'",
        name=missing_name,
    )

    def controlled_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "sentence_transformers":
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", controlled_import)

    if missing_name == "sentence_transformers":
        with pytest.raises(ImportError, match="pip install") as caught:
            SentenceTransformerEmbedder()

        assert caught.value.__cause__ is error
    else:
        with pytest.raises(ModuleNotFoundError) as caught:
            SentenceTransformerEmbedder()

        assert caught.value is error