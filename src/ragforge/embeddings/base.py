"""Define the contract shared by embedding implementations."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Embedder(Protocol):
    """Convert text batches into numeric vectors.

    Implementations must preserve input order and return a float32
    array with shape (number_of_texts, embedding_dimension).

    The embedding dimension must be positive and consistent across
    calls to the same configured embedder. Values must be finite.
    Nonempty input texts must produce nonzero vectors.

    Document and query embeddings must use the same compatible
    vector space. Implementations may apply model-specific prefixes
    internally.

    Empty batches must return shape (0, embedding_dimension).
    Vectors need not be normalized; the retriever handles that.
    """

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Return one embedding vector per input text."""
        ...