"""Adapt Sentence Transformers to the shared embedding contract."""

from collections.abc import Sequence
from numbers import Integral

import numpy as np
from numpy.typing import NDArray


class SentenceTransformerEmbedder:
    """Embed text using one model loaded at construction time.

    Importing this module does not import Sentence Transformers.
    Construction requires the optional ``dense`` dependencies and may
    download model files unless they are already cached.

    Use a symmetric model that embeds documents and queries into the
    same vector space without separate task-specific prefixes.
    """

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
            raise ValueError("model_name must be nonblank.")

        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        # Keep the optional dependency out of lightweight module imports.
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            if exc.name != "sentence_transformers":
                raise
            raise ImportError(
                "Sentence Transformers is required. "
                'Install it with: python -m pip install -e ".[dense]"'
            ) from exc

        self._model = SentenceTransformer(model_name, device=device)
        
                # Prefer the current API while supporting older dependency versions.
        get_dimension = getattr(self._model, "get_embedding_dimension", None)
        if get_dimension is None:
            get_dimension = self._model.get_sentence_embedding_dimension
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
        """Return the configured model's embedding dimension."""
        return self._dimension

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Return ordered float32 vectors without normalizing them."""
        # A single string is a sequence, but it is not a batch of texts.
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings, not one string.")
        if not isinstance(texts, Sequence):
            raise TypeError("texts must be a sequence of strings.")

        batch = list(texts)
        if any(not isinstance(text, str) for text in batch):
            raise TypeError("Every input text must be a string.")

        if not batch:
            return np.empty((0, self._dimension), dtype=np.float32)

        output = self._model.encode(
            batch,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        vectors = np.asarray(output, dtype=np.float32)

        if vectors.shape != (len(batch), self._dimension):
            raise ValueError(
                "Model output shape must match the batch size "
                "and configured embedding dimension."
            )

        if not np.isfinite(vectors).all():
            raise ValueError("Model embeddings must contain only finite values.")

        # Check for all-zero rows without computing potentially unstable norms.
        if np.any(~np.any(vectors != 0, axis=1)):
            raise ValueError("Every model embedding must have a nonzero vector.")

        return vectors