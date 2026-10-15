"""Offline verifier plumbing tests, not real-model accuracy tests."""

import json
from unittest.mock import Mock

import pytest

from ragforge.generation.core import AnswerGenerator
from ragforge.generation.support import InvalidVerificationResponse
from ragforge.ingestion.chunking import TextChunk
from ragforge.retrieval.types import SearchResult


def hit(text):
    return SearchResult(
        chunk=TextChunk("sample.pdf", 1, 1, 0, len(text), text),
        score=0.0,
    )
def test_unsupported_numbered_section_does_not_use_fallback():
    from ragforge.retrieval.abstract import AbstractSectionRetriever

    fallback = Mock()
    reranker = Mock()
    retriever = AbstractSectionRetriever(fallback, reranker, [])
    results = retriever.search(
        "According to section 5.4, what measurable value contributions "
        "are described?",
        top_k=3,
    )
    assert results == []
    fallback.search.assert_not_called()
    reranker.rerank.assert_not_called()


def response(points):
    return json.dumps({"points": points, "abstain": False, "reason": None})


def test_unsupported_negative_is_abstained_when_verifier_rejects():
    generator_llm = Mock()
    generator_llm.complete.return_value = response([
        {
            "text": (
                "The study did not demonstrate a numerical percentage "
                "reduction in inference costs."
            ),
            "citations": ["S1"],
        }
    ])
    verifier = Mock()
    verifier.complete.return_value = '{"supported": false}'
    result = hit(
        "The article advances cost per useful output as a more meaningful "
        "metric for evaluating AI deployment."
    )
    answer = AnswerGenerator(
        generator_llm, verification_llm=verifier
    ).generate(
        "According to the abstract only, what numerical percentage "
        "reduction in inference costs did the study demonstrate?",
        [result],
    )
    assert answer.status == "abstained"
    assert answer.points == answer.citations == ()
    assert answer.evidence[0].chunk is result.chunk
    generator_llm.complete.assert_called_once()
    verifier.complete.assert_called_once()
    request = json.loads(verifier.complete.call_args.kwargs["user_prompt"])
    assert request["points"][0]["citations"] == ["S1"]
    assert request["evidence"][0]["text"] == result.chunk.text
    assert request["evidence"][0]["end_char"] == len(result.chunk.text)


def test_supported_dimensions_and_metric_are_preserved():
    points = [
        {
            "text": "The dimensions are utility, demand, supply, and monetization.",
            "citations": ["S1"],
        },
        {
            "text": "The recommended metric is cost per useful output.",
            "citations": ["S2"],
        },
    ]
    generator_llm = Mock()
    generator_llm.complete.return_value = response(points)
    verifier = Mock()
    verifier.complete.return_value = '{"supported": true}'
    answer = AnswerGenerator(
        generator_llm, verification_llm=verifier
    ).generate(
        "What are the dimensions and recommended metric?",
        [
            hit("The dimensions are utility, demand, supply, and monetization."),
            hit("The recommended metric is cost per useful output."),
        ],
    )
    assert answer.status == "answered"
    assert [point.text for point in answer.points] == [
        point["text"] for point in points
    ]
    assert [item.label for item in answer.citations] == ["S1", "S2"]
    verifier.complete.assert_called_once()


@pytest.mark.parametrize("no_evidence", [False, True])
def test_abstention_and_no_evidence_skip_verification(no_evidence):
    generator_llm = Mock()
    generator_llm.complete.return_value = json.dumps({
        "points": [],
        "abstain": True,
        "reason": "The supplied evidence does not report the requested number.",
    })
    verifier = Mock()
    answer = AnswerGenerator(
        generator_llm, verification_llm=verifier
    ).generate("What percentage?", [] if no_evidence else [hit("A metric.")])
    assert answer.status == ("no_evidence" if no_evidence else "abstained")
    verifier.complete.assert_not_called()
    assert generator_llm.complete.call_count == (0 if no_evidence else 1)


@pytest.mark.parametrize("raw", [
    None, "not JSON", "[]", "{}", '{"supported": 1}',
    '{"supported": "false"}', '{"supported": true, "extra": 0}',
])
def test_malformed_verification_is_an_error_not_abstention(raw):
    generator_llm = Mock()
    generator_llm.complete.return_value = response([
        {"text": "Revenue grew.", "citations": ["S1"]}
    ])
    verifier = Mock()
    verifier.complete.return_value = raw
    with pytest.raises(InvalidVerificationResponse):
        AnswerGenerator(
            generator_llm, verification_llm=verifier
        ).generate("What changed?", [hit("Revenue grew.")])
    generator_llm.complete.assert_called_once()
    verifier.complete.assert_called_once()


def test_verifier_backend_failure_propagates():
    generator_llm = Mock()
    generator_llm.complete.return_value = response([
        {"text": "Revenue grew.", "citations": ["S1"]}
    ])
    verifier = Mock()
    error = TimeoutError("Verifier unavailable")
    verifier.complete.side_effect = error
    with pytest.raises(TimeoutError) as caught:
        AnswerGenerator(
            generator_llm, verification_llm=verifier
        ).generate("What changed?", [hit("Revenue grew.")])
    assert caught.value is error
    generator_llm.complete.assert_called_once()
    verifier.complete.assert_called_once()


def test_verification_runs_once_after_answer_contract_retry():
    generator_llm = Mock()
    generator_llm.complete.side_effect = [
        "not JSON",
        response([{"text": "Revenue grew.", "citations": ["S1"]}]),
    ]
    verifier = Mock()
    verifier.complete.return_value = '{"supported": true}'
    answer = AnswerGenerator(
        generator_llm, verification_llm=verifier
    ).generate("What changed?", [hit("Revenue grew.")])
    assert answer.status == "answered"
    assert generator_llm.complete.call_count == 2
    verifier.complete.assert_called_once()
def test_verifier_receives_only_cited_evidence():
    generator_llm = Mock()
    generator_llm.complete.return_value = response([
        {"text": "Revenue grew.", "citations": ["S1"]}
    ])
    verifier = Mock()
    verifier.complete.return_value = '{"supported": true}'
    AnswerGenerator(generator_llm, verification_llm=verifier).generate(
        "What changed?",
        [hit("Revenue grew."), hit("Uncited contextual evidence.")],
    )
    request = json.loads(verifier.complete.call_args.kwargs["user_prompt"])
    assert [item["label"] for item in request["evidence"]] == ["S1"]
    assert request["evidence"][0]["text"] == "Revenue grew."
