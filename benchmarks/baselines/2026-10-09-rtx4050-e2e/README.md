# RTX 4050 end-to-end retrieval baseline — 2026-10-09

## Scope
Direct request timing: CPU query embedding, hybrid retrieval, and
CPU or CUDA cross-encoder reranking. Index construction and model loading
are outside request timing. Validation and report generation are outside
the timed request interval.

Dataset: 96 chunks, 25 unique held-out questions.
Each device: 2 full warm-up passes, then 5 measured passes (125 requests).
Dataset order is reused on each pass; CPU was run before CUDA.

## Results
| Metric | CPU reranker | CUDA reranker |
| --- | ---: | ---: |
| Mean end-to-end latency | 416.88 ms | 44.91 ms |
| Median | 422.57 ms | 45.16 ms |
| p95 | 548.58 ms | 53.66 ms |
| First query after setup | 551.30 ms | 700.55 ms |
| Hit@5 | 1.000 | 1.000 |
| MRR@5 | 0.980 | 0.980 |
| Repeat top-five order agreement | 1.000 | 1.000 |

Mean end-to-end speedup: approximately 9.28x.
First measured pass: no CPU/CUDA top-five order differences on 25 questions.
Reciprocal mean latency is a sequential service rate, not concurrent throughput.

## Configuration
NVIDIA GeForce RTX 4050 Laptop GPU; WSL2; Python 3.11.17.
Both models use pinned revisions; reranker parameters are FP32.
Batch size 8; maximum length 512; candidate pool 20; output top-k 5.
PyTorch threads: 6 intra-op, 12 inter-op. TF32 disabled.
CUDA request timing includes explicit device synchronization.
Full environment, model revisions, and input hashes are in the JSON reports.

## Provenance and integrity
Original reports: benchmarks/runs/e2e-20261009T102948Z/.
Recorded parent commit: 8588fe0407841295b1c0ec014f2c4aa15be37172.
The benchmark script was untracked during execution; the recorded commit
alone does not identify its source. Reports retain their original Git fields.
The archived script is the working-tree source preserved after this run;
SHA256SUMS identifies that snapshot and the accompanying tests.
Subsequent validation: Ruff passed; 9 benchmark tests and 327 total tests passed.
Verify this archive with: sha256sum -c SHA256SUMS

## Limitations
One CPU-then-CUDA run sequence on a small dataset; repeated requests are
not additional independent questions. Power and thermal telemetry were
not captured. First-query timing is not full process startup latency or
a controlled idle-to-active experiment. These results do not establish
broader quality generalization or production concurrent throughput.
