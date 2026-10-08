"""Download-free tests for fixed-candidate reranking."""

from unittest.mock import Mock

import pytest

from ragforge.ingestion.chunking import TextChunk
from ragforge.retrieval.reranking import Reranker
from ragforge.retrieval.types import SearchResult


@pytest.fixture
def candidates() -> list[SearchResult]:
    """Build real chunks with distinct source-linked metadata."""
    return [
        SearchResult(
            chunk=TextChunk(
                source="synthetic.pdf",
                page_number=index + 1,
                chunk_number=1,
                start_char=0,
                end_char=len(text),
                text=text,
            ),
            score=10.0 - index,
        )
        for index, text in enumerate(["first passage", "second passage", "third passage"])
    ]


def test_reranking_orders_scores_and_preserves_chunks(candidates):
    scorer = Mock()
    scorer.score.return_value = [0.2, 0.9, 0.5]
    original = list(candidates)

    results = Reranker(scorer=scorer).rerank("query", candidates, top_k=2)

    scorer.score.assert_called_once_with(
        "query", ["first passage", "second passage", "third passage"]
    )
    assert [result.score for result in results] == [0.9, 0.5]
    assert results[0].chunk is candidates[1].chunk
    assert results[1].chunk is candidates[2].chunk
    assert candidates == original
    assert [result.score for result in candidates] == [10.0, 9.0, 8.0]


def test_ties_preserve_candidate_order(candidates):
    scorer = Mock()
    scorer.score.return_value = [0.5, 0.5, 0.5]

    results = Reranker(scorer=scorer).rerank("query", candidates)

    assert [result.chunk for result in results] == [
        result.chunk for result in candidates
    ]


def test_generator_and_large_top_k(candidates):
    scorer = Mock()
    scorer.score.return_value = [-3, 0, -1]

    results = Reranker(scorer=scorer).rerank(
        "query", (result for result in candidates), top_k=10
    )

    assert [result.chunk for result in results] == [
        candidates[1].chunk, candidates[2].chunk, candidates[0].chunk
    ]
    assert [result.score for result in results] == [0.0, -1.0, -3.0]


@pytest.mark.parametrize("query", ["", " ", "\n\t"])
def test_blank_query_skips_scorer(query, candidates):
    scorer = Mock()

    assert Reranker(scorer=scorer).rerank(query, candidates) == []
    scorer.score.assert_not_called()


def test_empty_candidates_skip_scorer():
    scorer = Mock()

    assert Reranker(scorer=scorer).rerank("query", []) == []
    scorer.score.assert_not_called()


@pytest.mark.parametrize("top_k", [True, False, 1.5, "5", None])
def test_noninteger_top_k_is_rejected(top_k, candidates):
    scorer = Mock()

    with pytest.raises(TypeError, match="top_k must be an integer"):
        Reranker(scorer=scorer).rerank("query", candidates, top_k=top_k)

    scorer.score.assert_not_called()


@pytest.mark.parametrize("top_k", [0, -1])
def test_nonpositive_top_k_is_rejected(top_k, candidates):
    scorer = Mock()

    with pytest.raises(ValueError, match="greater than zero"):
        Reranker(scorer=scorer).rerank("query", candidates, top_k=top_k)

    scorer.score.assert_not_called()


@pytest.mark.parametrize("query", [None, 123, b"query"])
def test_nonstring_query_is_rejected(query, candidates):
    scorer = Mock()

    with pytest.raises(TypeError, match="query must be a string"):
        Reranker(scorer=scorer).rerank(query, candidates)

    scorer.score.assert_not_called()


def test_invalid_candidate_is_rejected(candidates):
    scorer = Mock()

    with pytest.raises(TypeError, match="Every candidate"):
        Reranker(scorer=scorer).rerank("query", [candidates[0], object()])

    scorer.score.assert_not_called()


@pytest.mark.parametrize("scores", [[], [0.1], [0.1, 0.2, 0.3, 0.4]])
def test_score_count_mismatch_is_rejected(scores, candidates):
    scorer = Mock()
    scorer.score.return_value = scores

    with pytest.raises(ValueError, match="Score count"):
        Reranker(scorer=scorer).rerank("query", candidates)


@pytest.mark.parametrize("bad_score", [True, False, "0.5", None, 1 + 2j])
def test_nonreal_scores_are_rejected(bad_score, candidates):
    scorer = Mock()
    scorer.score.return_value = [0.1, bad_score, 0.3]

    with pytest.raises(TypeError, match="real number"):
        Reranker(scorer=scorer).rerank("query", candidates)


@pytest.mark.parametrize(
    "bad_score", [float("nan"), float("inf"), float("-inf"), 10**1000]
)
def test_nonfinite_scores_are_rejected(bad_score, candidates):
    scorer = Mock()
    scorer.score.return_value = [0.1, bad_score, 0.3]

    with pytest.raises(ValueError, match="finite"):
        Reranker(scorer=scorer).rerank("query", candidates)


def test_backend_exception_propagates_unchanged(candidates):
    scorer = Mock()
    error = RuntimeError("synthetic scorer failure")
    scorer.score.side_effect = error

    with pytest.raises(RuntimeError) as caught:
        Reranker(scorer=scorer).rerank("query", candidates)

    assert caught.value is error
