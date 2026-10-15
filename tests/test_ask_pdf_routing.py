"""Exercise production retrieval wrappers with synthetic, model-free inputs."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from ragforge.ingestion.chunking import TextChunk
from ragforge.retrieval.types import SearchResult


@pytest.fixture
def module():
    from ragforge import application

    return application


@pytest.fixture
def pages():
    return {
        1: (
            "TITLE outside abstract\n"
            "Recent breakthroughs in Large Language Models provide new tools.\n"
            "Abstract evidence.\n"
            "KEYWORDS\n"
            "Keywords outside abstract.\n"
        ),
        15: (
            "Earlier material outside discussion.\n"
            "5 Discussion\n"
            "Discussion introduction.\n"
            "5.1 Strengths of existing research\n"
            "Strength evidence on page fifteen.\n"
        ),
        16: "More strength evidence. " * 12,
        18: (
            "Final strength evidence.\n"
            "5.2 Limitations of existing research studies\n"
            "Limitation evidence.\n"
            "5.3 Research gaps and future directions\n"
            "Future direction evidence.\n"
        ),
        19: (
            "Further future direction evidence.\n"
            "Author contributions\n"
            "Author material outside discussion.\n"
        ),
    }


def make_chunks(pages):
    chunks = []
    for page, text in pages.items():
        spans = [(0, len(text))]
        if page == 16:
            midpoint = len(text) // 2
            spans = [(0, midpoint + 12), (midpoint, len(text))]
        for number, (start, end) in enumerate(spans, start=1):
            chunks.append(
                TextChunk(
                    source="synthetic-profile.pdf",
                    page_number=page,
                    chunk_number=number,
                    start_char=start,
                    end_char=end,
                    text=text[start:end],
                )
            )
    return chunks


@pytest.fixture
def routing(module, pages):
    fallback = Mock()
    reranker = Mock()

    def rerank(query, candidates, *, top_k):
        return [
            SearchResult(chunk=result.chunk, score=float(100 - index))
            for index, result in enumerate(reversed(candidates))
        ][:top_k]

    reranker.rerank.side_effect = rerank
    retriever = module.PaperSectionRetriever(fallback, reranker, make_chunks(pages))
    return retriever, fallback, reranker


@pytest.mark.parametrize(
    ("candidate_k", "top_k", "expected_pool"),
    [(30, 3, 30), (30, 40, 40), (8, 3, 8)],
)
def test_wrapper_widens_pool_and_delegates(module, candidate_k, top_k, expected_pool):
    backend = Mock()
    reranker = Mock()
    wrapper = module.RerankedRetriever(backend, reranker, candidate_k=candidate_k)
    result = wrapper.search("A synthetic query", top_k=top_k)

    backend.search.assert_called_once_with("A synthetic query", top_k=expected_pool)
    reranker.rerank.assert_called_once_with(
        "A synthetic query", backend.search.return_value, top_k=top_k
    )
    assert result is reranker.rerank.return_value


def test_sections_have_exact_boundaries_and_preserve_metadata(routing, pages):
    retriever, _, _ = routing
    bounds = {
        "abstract": (
            (1, pages[1].index("Recent breakthroughs")),
            (1, pages[1].index("KEYWORDS")),
        ),
        "discussion": (
            (15, pages[15].index("5 Discussion")),
            (19, pages[19].index("Author contributions")),
        ),
        "5.1": (
            (15, pages[15].index("5.1 Strengths")),
            (18, pages[18].index("5.2 Limitations")),
        ),
        "5.2": (
            (18, pages[18].index("5.2 Limitations")),
            (18, pages[18].index("5.3 Research")),
        ),
        "5.3": (
            (18, pages[18].index("5.3 Research")),
            (19, pages[19].index("Author contributions")),
        ),
    }
    originals = make_chunks(pages)
    for section, (start, end) in bounds.items():
        expected = []
        for chunk in originals:
            if not start[0] <= chunk.page_number <= end[0]:
                continue
            left = max(
                chunk.start_char,
                start[1] if chunk.page_number == start[0] else chunk.start_char,
            )
            right = min(
                chunk.end_char,
                end[1] if chunk.page_number == end[0] else chunk.end_char,
            )
            if left < right:
                expected.append(
                    replace(
                        chunk,
                        start_char=left,
                        end_char=right,
                        text=pages[chunk.page_number][left:right],
                    )
                )
        assert retriever._sections[section] == expected
        assert expected


@pytest.mark.parametrize(
    ("query", "section"),
    [
        ("Explain section 5.1", "5.1"),
        ("Explain section 5.2", "5.2"),
        ("Explain section 5.3", "5.3"),
        ("Summarize the ABSTRACT", "abstract"),
        ("What are the strengths?", "5.1"),
        ("Explain abstractness and strengths-basedness", "5.1"),
        ("What are the LIMITATIONS?", "5.2"),
        ("What future directions are proposed?", "5.3"),
        ("What future research directions are proposed?", "5.3"),
        ("What research gaps remain?", "5.3"),
        ("Discuss 5.2 strengths and abstract", "5.2"),
        ("Abstract and future directions", "abstract"),
        ("Future directions and limitations", "5.3"),
        ("Limitations and strengths", "5.2"),
        ("Strengths in the discussion", "5.1"),
    ],
)
def test_routes_and_precedence(routing, query, section):
    retriever, fallback, reranker = routing
    results = retriever.search(query, top_k=3)

    fallback.search.assert_not_called()
    reranker.rerank.assert_called_once()
    call = reranker.rerank.call_args
    assert call.args[0] == query
    assert call.kwargs == {"top_k": 3}
    candidates = call.args[1]
    assert [item.chunk for item in candidates] == retriever._sections[section]
    assert all(item.score == 0.0 for item in candidates)
    assert [item.chunk for item in results] == [item.chunk for item in reversed(candidates)][:3]


@pytest.mark.parametrize(
    "query",
    [
        "Explain the methods",
        "What is the future of this field?",
        "Explain section 5.4",
        "Explain abstractness and strengthsfulness",
    ],
)
def test_unmatched_queries_use_fallback(routing, query):
    retriever, fallback, reranker = routing
    assert retriever.search(query, top_k=2) is fallback.search.return_value
    fallback.search.assert_called_once_with(query, top_k=2)
    reranker.rerank.assert_not_called()


@pytest.mark.parametrize("top_k", [1, 3, 5, 20])
def test_discussion_balances_ranked_subsections(routing, top_k):
    retriever, fallback, reranker = routing
    results = retriever.search("Summarize the DISCUSSION", top_k=top_k)

    fallback.search.assert_not_called()
    calls = reranker.rerank.call_args_list
    assert [call.args[0] for call in calls] == [
        "What strengths does the reviewed research show?",
        "What limitations does the reviewed research have?",
        "What future research directions are proposed?",
    ]
    pools = []
    for call, section in zip(calls, ("5.1", "5.2", "5.3")):
        assert call.kwargs == {"top_k": top_k}
        assert [item.chunk for item in call.args[1]] == retriever._sections[section]
        pools.append(list(reversed(retriever._sections[section]))[:top_k])

    expected = []
    for depth in range(max(map(len, pools))):
        for pool in pools:
            if depth < len(pool):
                expected.append(pool[depth])
    assert [item.chunk for item in results] == expected[:top_k]


@pytest.mark.parametrize("query", ["", " ", "\n\t"])
def test_blank_query_does_not_call_dependencies(routing, query):
    retriever, fallback, reranker = routing
    assert retriever.search(query) == []
    fallback.search.assert_not_called()
    reranker.rerank.assert_not_called()


@pytest.mark.parametrize(
    ("query", "top_k", "error"),
    [
        (None, 3, TypeError),
        (42, 3, TypeError),
        ("discussion", True, TypeError),
        ("discussion", 1.5, TypeError),
        ("discussion", "3", TypeError),
        ("discussion", 0, ValueError),
        ("discussion", -1, ValueError),
    ],
)
def test_invalid_search_inputs(routing, query, top_k, error):
    retriever, fallback, reranker = routing
    with pytest.raises(error):
        retriever.search(query, top_k=top_k)
    fallback.search.assert_not_called()
    reranker.rerank.assert_not_called()


@pytest.mark.parametrize(
    ("page", "marker"),
    [
        (1, "Recent breakthroughs in Large Language Models"),
        (1, "KEYWORDS"),
        (15, "5 Discussion"),
        (15, "5.1 Strengths of existing research"),
        (18, "5.2 Limitations of existing research studies"),
        (18, "5.3 Research gaps and future directions"),
        (19, "Author contributions"),
    ],
)
def test_missing_verified_markers_fail(module, pages, page, marker):
    pages[page] = pages[page].replace(marker, "REMOVED MARKER")
    with pytest.raises(ValueError, match="Verified section marker missing"):
        module.PaperSectionRetriever(Mock(), Mock(), make_chunks(pages))


def test_inconsistent_overlapping_text_fails(module, pages):
    chunks = make_chunks(pages)
    index = next(
        i for i, chunk in enumerate(chunks) if chunk.page_number == 16 and chunk.chunk_number == 2
    )
    chunks[index] = replace(chunks[index], text="!" + chunks[index].text[1:])
    with pytest.raises(ValueError, match="Overlapping chunk text disagrees"):
        module.PaperSectionRetriever(Mock(), Mock(), chunks)


def test_mismatched_chunk_offsets_fail(module, pages):
    chunks = make_chunks(pages)
    chunks[0] = replace(chunks[0], end_char=chunks[0].end_char + 1)
    with pytest.raises(ValueError, match="Unexpected chunk offsets"):
        module.PaperSectionRetriever(Mock(), Mock(), chunks)


def test_reversed_section_boundaries_fail(module, pages):
    pages[1] = "KEYWORDS\nRecent breakthroughs in Large Language Models.\n"
    with pytest.raises(ValueError, match="Invalid section boundaries"):
        module.PaperSectionRetriever(Mock(), Mock(), make_chunks(pages))
