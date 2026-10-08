"""Sentence Transformers adapter with validated NumPy embeddings.

The optional model dependency is imported only when an adapter is
instantiated. Embeddings are returned without normalization; retrieval
implementations are responsible for their own similarity calculations.
"""

from collections.abc import Iterable
from numbers import Integral

import numpy as np
from numpy.typing import NDArray


class SentenceTransformerEmbedder:
    """Load one model and return finite, nonzero float32 embeddings."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string.")
        if not model_name.strip():
            raise ValueError("model_name must not be blank.")

        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            if exc.name != "sentence_transformers":
                raise
            raise ImportError(
                "Sentence Transformers is required for this embedder. "
                'Install it with: python -m pip install -e ".[dense]"'
            ) from exc

        self._model = SentenceTransformer(model_name, device=device)

        # Prefer the current API while supporting older dependency versions.
        get_dimension = getattr(self._model, "get_embedding_dimension", None)
        if get_dimension is None:
            get_dimension = getattr(self._model, "get_sentence_embedding_dimension", None)
        if get_dimension is None:
            raise AttributeError(
                "SentenceTransformer model does not expose an embedding dimension."
            )
        dimension = get_dimension()

        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, Integral)
            or dimension <= 0
        ):
            raise ValueError(
                "The model must report a positive integer embedding dimension."
            )

        self._dimension = int(dimension)
        self._batch_size = batch_size

    @property
    def dimension(self) -> int:
        """Return the model's validated embedding dimension."""
        return self._dimension

    def embed(self, texts: Iterable[str]) -> NDArray[np.float32]:
        """Embed texts in input order without normalizing the vectors.

        An empty batch returns an empty float32 matrix without calling
        the model's encoder. Invalid inputs are rejected before encoding.
        Backend encoding exceptions propagate unchanged.
        """
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be an iterable of strings.")

        try:
            batch = list(texts)
        except TypeError as exc:
            raise TypeError("texts must be an iterable of strings.") from exc

        if any(not isinstance(text, str) for text in batch):
            raise TypeError("Every item in texts must be a string.")

        if not batch:
            return np.empty((0, self._dimension), dtype=np.float32)

        output = self._model.encode(
            batch,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        if not isinstance(output, np.ndarray):
            raise TypeError("Model output must be a NumPy array.")

        expected_shape = (len(batch), self._dimension)
        if output.shape != expected_shape:
            raise ValueError(
                f"Model output shape must be {expected_shape}; "
                f"received {output.shape}."
            )

        if not np.issubdtype(output.dtype, np.floating):
            raise TypeError("Model output must have a floating-point dtype.")

        if not np.isfinite(output).all():
            raise ValueError("Model output must contain only finite values.")

        # A finite float64 value can overflow when converted to float32.
        with np.errstate(over="ignore", invalid="ignore"):
            vectors = output.astype(np.float32, copy=False)

        if not np.isfinite(vectors).all():
            raise ValueError(
                "Model output must remain finite after float32 conversion."
            )

        # Check the converted vectors so underflow cannot create silent zeros.
        if np.any(np.all(vectors == 0, axis=1)):
            raise ValueError("Every embedding vector must be nonzero.")

        return vectors
