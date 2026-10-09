# RAGForge hardware baseline — 2026-10-09

## Hardware and environment

- GPU: NVIDIA GeForce RTX 4050 Laptop GPU.
- GPU memory: 6141 MiB; NVIDIA driver: 581.86.
  These two values were observed separately using nvidia-smi.
- Environment: WSL2 Linux x86_64; Python 3.11.17.
- PyTorch: 2.14.1+cu130; reported CUDA runtime: 13.0.
- Sentence Transformers: 5.7.0; Transformers: 5.19.0; NumPy: 2.4.6.
- CPU benchmark threads: 6 intra-op, 12 inter-op.
- CPU model, power mode, GPU power limit, and thermal conditions
  have not been established for this summary.

## Sustained reranking performance

Report: `reranking_sustained_cpu_gpu_20261009T075745047409Z.json`

Recorded commit: `b2d20983fcc4e79a4daa86c512862c54d37862c4`.
This run must not be attributed to the later commit `432baa4`.

Workload: 25 held-out questions, 20 fixed candidates per question,
batch size 8, maximum sequence length 512, FP32, TF32 disabled.
Two full warm-up passes and five measured passes per device produced
125 timing samples per device, not 125 distinct evaluation questions.

| Measurement | CPU | CUDA |
|---|---:|---:|
| Mean latency | 414.70 ms | 37.00 ms |
| Median latency | 412.59 ms | 37.33 ms |
| p95 latency | 550.87 ms | 45.69 ms |
| Sequential queries/second | 2.41 | 27.02 |
| Hit@5, first measured pass | 1.000 | 1.000 |
| MRR@5, first measured pass | 0.980 | 0.980 |

Reported mean-latency speedup: **11.21x**.
Reported full-ranking agreement: **100%**.

Timing includes tokenization, scoring, CPU score transfer, and sorting.
This is fixed-candidate reranking, not end-to-end RAG latency.
Sequential throughput is not concurrent serving capacity.

The recorded first CUDA query took 820.66 ms, separate from warm timings.
Peak PyTorch CUDA allocation was approximately 142 MiB; reservation
approximately 202 MiB. These exclude driver and other-process memory.

## Quality and preliminary integration measurements

Report: `reranking_cuda_integration_20261009T082510_1402.json`

Evaluation: 25 held-out questions, 96 chunks, one source PDF.
Labels are recorded as human-approved and frozen.
Scope: passage retrieval, not generated-answer evaluation.

| Method | Hit@5 | MRR@5 |
|---|---:|---:|
| Hybrid retrieval | 0.920 | 0.7747 |
| Hybrid + CUDA reranking | 1.000 | 0.9800 |

Reranking recovered both hybrid top-five misses:
q014 moved from candidate rank 13 to rank 1; q021 from rank 8 to rank 1.
A regression is also retained: q007 moved from rank 1 to rank 2.

Candidate-pool hit rate was 1.000; mean relevant-label coverage was 0.9667.
Finding one relevant passage does not establish complete evidence coverage.

This integration run used one warm-up query and one measured pass.
Mean retrieval-stage latency: 7.38 ms.
Mean reranking latency: 37.25 ms.
Combined stage mean: 44.63 ms, calculated from separately measured stages.
**The combined value is not directly measured end-to-end request latency.**

## Model reproducibility

Embedding: `sentence-transformers/all-MiniLM-L6-v2`
Revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`

Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
Revision: `233902d25c440f23af6f7d6e94d2946bac0bee0a`

All six original JSON reports are preserved unchanged.
Reports retain dataset hashes and detailed results.
Earlier runs are not interchangeable benchmarks or independent evaluations.

Verify JSON integrity from this directory with:
`sha256sum -c SHA256SUMS`

## Limitations and next steps

Results are specific to this hardware and small single-document evaluation.
They do not establish general retrieval accuracy, answer correctness,
concurrent capacity, or production latency guarantees.

Next: directly measure full retrieval-request latency, record power and
thermal conditions, and evaluate a broader frozen dataset.
Do not tune on this held-out set while continuing to call it untouched.
