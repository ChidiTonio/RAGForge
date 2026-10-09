# Retrieval latency after idle periods

## Status

The delay has been localized primarily to the CrossEncoder input-transfer
path. Its underlying cause remains unresolved. This investigation does
not establish a production fix.

## Observed environment

- NVIDIA GeForce RTX 4050
- NVIDIA driver 581.86; reported CUDA version 13.0
- PyTorch 2.14.1+cu130
- Windows/WSL

## Procedure

Compare warm retrieval with retrieval after idle gaps of 5 and 15 seconds,
using the same query, corpus, model, and retrieval configuration.

Diagnostic flags:

- `--stage-timing`: host timestamps for retrieval/reranking stages.
- `--prediction-timing`: timing for prediction, device placement, and preprocessing.
- `--prediction-detail-timing`: granular input-transfer and forward timing.
- `--cuda-pre-transfer-probe`: a timed, synchronized GPU operation before
  input transfer; this is an intervention, not a baseline measurement.

Use the scripts' `--help` output for required arguments and supported
combinations. Preserve the configuration and output report for each run.

## Findings

- Observed post-idle latency was approximately 240 ms.
- Approximately 86% of that latency was attributed to input transfer
  in the diagnostic measurements.
- The delay appeared after both 5-second and 15-second idle gaps.
- A pre-transfer GPU probe completed in approximately 1 ms, but the
  subsequent input transfer still incurred approximately 200 ms.
- Diagnostic runs preserved rankings relative to the warm reference
  (`matches_warm_reference`).

These figures summarize observed runs, not guaranteed performance.

## Interpretation

The probe result does not support a simple explanation in which any
first GPU operation absorbs the entire delay. It does not rule out
transfer-specific driver or power-state behavior.

The inspected transfer path calls `.to(target_device)` for individual
tensors without explicitly using pinned memory, preallocated GPU
buffers, or `non_blocking=True`.

Possible contributors include allocation, transfer handling, and
power-state transitions. None has been confirmed as the root cause.

## Measurement limitations

- Host-call timing is not equivalent to completed GPU execution timing.
- A transfer call may include allocation or synchronization costs;
  it does not isolate physical copy time.
- Forward host time is not completed GPU execution time.
- Prediction remainder is not isolated output-transfer time.
- Detailed timing temporarily wraps a module-level transfer reference;
  concurrent prediction in the same process is unsupported.
- The GPU probe changes scheduling and synchronization.
- Ranking agreement on these runs is not a retrieval-quality evaluation.

## Automated validation

Fake-backend tests cover detailed timing, probe accounting, and patch
restoration after failures. They validate instrumentation behavior,
not actual CUDA performance.

## Next milestone

Establish a labeled retrieval-quality baseline using recall@k, nDCG@k,
and MRR. Record the corpus, queries, relevance judgments, and retrieval
configuration. Further transfer optimization is deferred.
