# Fixed-candidate reranking experiment

## Scope

Development diagnostic using an unchanged private equity-research corpus:
23 chunks and 25 questions. This is not a held-out evaluation or an
end-to-end answer-quality assessment.

Corpus files, questions, and generated reports remain local and ignored
by Git. This document records aggregate results without passage text.

## Configuration

- Embeddings: sentence-transformers/all-MiniLM-L6-v2.
- Hybrid retrieval: BM25 and dense retrieval combined with RRF.
- Backend candidate depth: 20 per backend.
- RRF constant: 60.
- Final fused candidate pool: 20 per question.
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2.
- Reranker activation: identity, returning raw relevance scores.
- Reranker batch size: 8; maximum sequence length: 512.
- Device: CPU.
- Evaluation cutoff: top five.

Each measured hybrid pool was retrieved once and frozen. Both methods
were evaluated against the same candidate membership. The existing
hybrid baseline was reproduced at the reported precision.

## Results

| Metric | Hybrid | Hybrid + reranking |
|---|---:|---:|
| Hit Rate@5 | 0.920 (23/25) | 1.000 (25/25) |
| MRR@5 | 0.843 | 0.950 |

Candidate-pool hit rate was 1.000: every pool contained at least one
labeled relevant chunk.

First relevant ranks changed as follows:

| Question | Hybrid rank | Reranked rank |
|---|---:|---:|
| q005 | 7 | 1 |
| q013 | 4 | 1 |
| q015 | 6 | 1 |
| q019 | 3 | 4 |

Both hybrid top-five misses were recovered. q019 regressed in rank but
remained a hit. The other 21 questions had unchanged MRR@5 contributions;
this does not imply their complete rankings were unchanged.

## Latency

One measured pass followed one unmeasured warm-up query.

| Component | Mean | Median | p95 |
|---|---:|---:|---:|
| Hybrid retrieval | 13.6 ms | 13.4 ms | 14.9 ms |
| Added reranking | 637.9 ms | 639.6 ms | 724.5 ms |
| Combined | 651.4 ms | 653.5 ms | 739.1 ms |

These are descriptive CPU measurements across 25 questions, excluding
model loading, index construction, and answer generation. They are not
production latency guarantees.

## Decision and limitations

Keep hybrid retrieval as the default and reranking optional.

The experiment demonstrates real-backend compatibility and improved
development-set retrieval, not perfect retrieval or large-corpus
scalability. A reranker cannot recover passages missing from its pool.

Model names, package versions, and input hashes are recorded in the
local report. Model revisions are not pinned, so model-artifact
reproducibility remains incomplete.

Before promoting reranking to the default:
- Evaluate a larger, held-out question set.
- Review the competing passages for q019 without tuning labels to scores.
- Measure latency with repeated runs on the intended deployment hardware.
- Pin model revisions for reproducible model loading.

## Verification

At this milestone, Ruff passes and 301 unit tests pass, including six
download-free experiment tests.

Ordinary CI installs the base package and development dependencies.
Tests mock optional model backends; real-model experiments are run
explicitly outside the standard unit-test workflow.
