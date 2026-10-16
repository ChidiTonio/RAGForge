# RAGForge v0.1 evaluation report

Report date: 2026-10-16
Ingestion-hardening checkpoint: `afaadac`

## Summary

This report consolidates existing evaluation evidence. No new real-model
evaluation or deployment benchmark was performed to prepare this report.

- At the ingestion-hardening checkpoint, Ruff and all 670 tests were
  reported passing, including PDF-limit and Streamlit upload-guard tests.
- A historical held-out passage-retrieval experiment improved Hit Rate@5
  from 0.920 to 1.000 and MRR@5 from 0.775 to 0.980 with reranking.
- Historical manual answer checks include partial, provisional, and
  failed outcomes. Follow-up successes do not replace original outcomes.
- Deployment capacity, access control, and hosted behavior remain
  unvalidated. The intended v0.1 deployment is a restricted single-user demo.

Automated tests, passage-retrieval metrics, and manual answer checks
assess different properties and must not be treated as interchangeable.

## 1. Evidence and provenance

| Evidence | Source | Scope and checkpoint |
|---|---|---|
| Ingestion hardening | Commit `afaadac`; reported Ruff/test results | Parser limits and upload-guard behavior |
| Manual answer baseline | [Baseline](baseline-a4a8a53.md) | Header identifies `a4a8a53`; replacement-PDF checks dated 2026-10-15; exact run-code provenance is incomplete |
| Development reranking | [Experiment notes](../docs/reranking-experiment.md) | Private-corpus retrieval diagnostic, not held-out or answer evaluation |
| Held-out reranking | [JSON report](stem_education/reranking_heldout_v1.json) | Created 2026-10-08; committed in `b2d2098`; exact experiment code checkpoint is not recorded in the JSON |
| GPU idle-latency investigation | [Latency notes](../docs/performance/idle-latency.md) | Separate diagnostic environment; unresolved cause |

Commit `5eb5e3a`, dated 2026-10-15, restricted support verification to
citations and guarded unsupported section queries. The October 8
retrieval results do not establish evaluation of that later code or
of the ingestion-hardening checkpoint.

## 2. Held-out passage retrieval

### Dataset and configuration

- Dataset: `stem_education_v1_heldout`.
- Document: `Entrepreneurial STEM Education and Innovation Ecosystems.pdf`.
- Corpus: 96 chunks; 25 held-out questions.
- Labels: human-approved and frozen.
- Chunk size: 1,000; overlap: 150.
- Retrieval: BM25/dense hybrid with RRF constant 60.
- Backend candidate depth: 20; final candidate pool: 20.
- Evaluation cutoff: top five.
- Both methods rank the same fixed hybrid candidate pools.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`.
- Embedding revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Reranker revision: unpinned.
- Device: CPU; embedding/reranking batch size: 8.
- Reranker maximum sequence length: 512; activation: identity.

Recorded package versions: sentence-transformers 5.7.0,
transformers 5.19.0, torch 2.14.1+cpu, and numpy 2.4.6.
Input SHA-256 hashes are preserved in the linked JSON report.

### Results

| Metric | Hybrid | Hybrid + reranking |
|---|---:|---:|
| Hit Rate@5 | 0.920 (23/25) | 1.000 (25/25) |
| MRR@5 | 0.775 | 0.980 |

Candidate-pool hit rate was 1.000. Mean relevant-label coverage within
the candidate pools was 0.967. Having at least one relevant passage in
every pool does not mean every labeled relevant passage was retrieved.

These results support improved ranking on this small held-out dataset.
They do not establish perfect retrieval, answer correctness, large-corpus
scalability, or generalization across documents and domains.

### Descriptive latency

One measured pass followed one warm-up query.

| Component | Mean | Median | p95 |
|---|---:|---:|---:|
| Hybrid retrieval | 8.74 ms | 8.69 ms | 10.20 ms |
| Added reranking | 484.32 ms | 500.75 ms | 581.58 ms |
| Combined | 493.06 ms | 508.40 ms | 592.55 ms |

Recorded setup times were 52.34 s for embedder loading, 1.72 s for
hybrid index construction, and 2.98 s for reranker loading.

These are single-run observations, not deployment benchmarks.
Query-stage timings exclude setup and answer generation. Setup times
do not isolate downloads, cache effects, or other initialization costs.

## 3. Historical manual answer checks

The AI Tokenomics baseline records these original outcomes:

| Test | Outcome |
|---|---|
| 1–3: objective, dimensions, metric explanation | Pass, with extraction/page-verification caveats |
| 4: formula retrieval | Partial pass; components correct but division relationship omitted |
| 5: measurement limitations | Pass with a relevance caveat |
| 6: unsupported numerical claim | Provisional safe abstention; document-wide ground truth and retrieved evidence not reviewed |
| 7: value contributions | Fail; returned output categories instead of requested value types |

Clarified follow-ups for Tests 4 and 7 succeeded, but do not overwrite
the original partial pass and failure. Test 7's follow-up also duplicated
the answer.

Retain individual response times in the baseline rather than combining
original tests and follow-ups into an aggregate benchmark.
Do not derive a definitive answer pass rate from these mixed outcomes.

The README separately records live `qwen2.5:7b` CLI checks at the
October 15 checkpoint. The baseline does not record enough model,
revision, configuration, and run-code detail to assign that model
conclusively to every baseline row.

## 4. Ingestion-hardening validation

At checkpoint `afaadac`, Ruff and all 670 tests were reported passing.

Default limits for newly parsed PDFs:

- File size: 20 MiB.
- Page count: 200 pages.
- Cumulative extracted text: 1,000,000 characters.

Regression tests cover PDF-limit enforcement and the Streamlit upload
guard, including state clearing after oversized-upload rejection.
The frontend rejects oversized uploads before copying their bytes;
Streamlit also has a configured upload limit of 20.

This validates tested behavior, not a PDF sandbox, extraction timeout,
or exhaustive resistance to malicious documents. Compatible cache hits
skip parsing and do not reapply parser limits. OCR is unsupported.

## 5. Remaining release limitations

- Citation membership does not prove factual support, answer relevance,
  or quotation fidelity.
- Formula layout and chunk boundaries can degrade extracted evidence.
- The CLI's filename-specific `PaperSectionRetriever` configuration
  differs from the Web UI's `AbstractSectionRetriever`.
- Unpinned reranker artifacts limit reproducibility.
- Retrieval experiments do not validate end-to-end generation.
- The historical GPU idle-delay investigation is not a CPU demo
  benchmark and has no confirmed production fix.

## 6. Deployment checks still required

Before hosting the restricted single-user demo:

- Record host CPU, RAM, operating system, and relevant package versions.
- Provision the selected Ollama model and record its tag/artifact identity.
- Preload embedding and reranking models.
- Measure cold and warm end-to-end latency on the intended host.
- Measure memory during model loading, ingestion, and question answering.
- Account for per-session model allocation and restrict concurrent use.
- Configure and test access control separately from upload limits.
- Confirm oversized-upload handling and normal PDF use in the hosted UI.

Current evidence supports tested ingestion hardening and historical
retrieval improvements. It does not certify production readiness or
comprehensive current-release answer quality.
