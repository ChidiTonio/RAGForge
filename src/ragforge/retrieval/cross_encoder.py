"""Optional Sentence Transformers cross-encoder relevance scorer."""

from collections.abc import Sequence

import numpy as np


class CrossEncoderScorer:
    """Score query-passage pairs with a single-output cross-encoder.

    Model dependencies are imported only during construction.
    Scores are raw model outputs, not relevance probabilities.
    Inputs longer than max_length tokens are truncated by the backend.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string.")
        if not model_name.strip():
            raise ValueError("model_name must not be blank.")

        if not isinstance(device, str):
            raise TypeError("device must be a string.")
        if not device.strip():
            raise ValueError("device must not be blank.")

        for name, value in (
            ("batch_size", batch_size),
            ("max_length", max_length),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero.")

        try:
            from sentence_transformers import CrossEncoder
        except ModuleNotFoundError as exc:
            if exc.name != "sentence_transformers":
                raise
            raise ImportError(
                "Sentence Transformers is required for this scorer. "
                'Install it with: python -m pip install -e ".[dense]"'
            ) from exc

        from torch.nn import Identity

        self._model = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
        )
        self._activation = Identity()
        self._batch_size = batch_size

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return finite scores in passage order without changing inputs.

        Empty passage lists skip prediction. Blank queries are rejected.
        Backend prediction exceptions propagate unchanged.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        if not query.strip():
            raise ValueError("query must not be blank.")

        if isinstance(passages, (str, bytes)):
            raise TypeError("passages must be a sequence of strings.")

        batch = list(passages)
        if any(not isinstance(passage, str) for passage in batch):
            raise TypeError("Every passage must be a string.")
        if any(not passage.strip() for passage in batch):
            raise ValueError("Every passage must contain nonblank text.")
        if not batch:
            return []

        pairs = [(query, passage) for passage in batch]
        output = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
            activation_fn=self._activation,
            apply_softmax=False,
            convert_to_numpy=True,
        )

        if not isinstance(output, np.ndarray):
            raise TypeError("Model output must be a NumPy array.")
        if output.shape != (len(batch),):
            raise ValueError(
                "Model must return one score per passage with shape "
                f"{(len(batch),)}; received {output.shape}."
            )
        if not np.issubdtype(output.dtype, np.floating):
            raise TypeError("Model output must have a floating-point dtype.")
        if not np.isfinite(output).all():
            raise ValueError("Model output must contain only finite values.")

        with np.errstate(over="ignore", invalid="ignore"):
            scores = output.astype(np.float64, copy=False)
        if not np.isfinite(scores).all():
            raise ValueError("Model scores must remain finite after conversion.")

        return scores.tolist()
