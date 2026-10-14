# RAGForge — Session Handoff

## Current state
- Branch: master
- Environment: ~/venvs/ragforge-gpu
- Python: 3.11.17
- Previous checkpoint: c159666 (bounded retry regression coverage)
- Current work: point-level citation contract implemented.
- Focused generation + Ollama tests: 89 passed.
- Full suite: 448 passed on 2026-10-14.
- Live model validation of the new contract is still pending.

## Completed in this checkpoint
- Added immutable AnswerPoint and default-empty Answer.points.
- New model JSON contract: points, abstain, reason.
- Each answered point requires text and valid, distinct citation labels.
- Citation labels may be reused across different points.
- Python renders numbered points and citation markers.
- Global citations are deduplicated in first-use order.
- Existing script-facing Answer fields remain available.
- Manual abstentions remain compatible.
- Existing bounded retry behavior is preserved.
- Added 25 tests.
- No retrieval, PDF script, or Ollama adapter changes.

## Limitations
- Citation membership does not prove factual support.
- Requested point count is instructed, not parser-enforced.
- Numbering and citation markers inside model point text are discouraged
  by the prompt, not rejected by the parser.
- Quotation fidelity is not yet validated.

## Next task
Validate the new contract with the existing PDF workflow and qwen2.5:7b.
Explicitly pass --model qwen2.5:7b; script default remains qwen2.5:3b.
Check a supported answer, a multi-point summary, and an abstention.
Inspect point/citation placement and contract failures.
Do not repeat completed offline baseline checks without a code change.

## Working rules
Use the Ubuntu terminal.
Activate: source ~/venvs/ragforge-gpu/bin/activate
Stage only intended files.
Do not stage benchmarks/runs/ or .bak files.
Update this handoff at each checkpoint.

## Live smoke-test results — 2026-10-14
Checkpoint tested: 0b47d08, using qwen2.5:7b.
- Structured answers and rendered citations worked.
- Abstract answer: cited passages supported the displayed points,
  but a request for one point returned three.
- Discussion answer: three points correctly covered a strength,
  limitation, and future direction with corresponding citations.
- Future-direction point was a supported fragment rather than a
  complete explanatory sentence; no missing continuation was invented.
- Bank-account question: correctly abstained without citations and
  limited its explanation to the supplied evidence.
- No validation-failure debug messages appeared in the supplied output.
- These results supersede the earlier "live validation pending" note.
- The interactive application has been exited.

Next task:
Design explicit point-count validation and regression tests so count
mismatches can trigger the existing bounded retry. Define how the
requested count reaches the generator; do not assume arbitrary natural
language can be reliably parsed with a simple regex.
No repeat smoke test or offline baseline rerun is needed before that work.

## Explicit point-count enforcement — 2026-10-14

Implemented:
- Optional point_count on AnswerGenerator.generate and RAGPipeline.answer.
- Explicit requested count included in the LLM request.
- Supported answers with the wrong number of points trigger the existing
  bounded two-attempt validation retry.
- Valid abstention and no-evidence behavior remain allowed.
- Invalid counts are rejected before retrieval/backend calls.
- PDF CLI supports --point-count N and rejects nonpositive counts before
  PDF/model loading. In interactive mode the count applies to all questions.
- Natural-language count wording alone does not activate enforcement.

Verification:
- git diff --check passed before the CLI patch; rerun before committing.
- Focused generation/Ollama suite: 105 passed.
- Full suite after CLI integration: 464 passed (16 new generation cases).
- CLI --point-count 0 rejected before PDF/model loading.
- Live qwen2.5:7b Abstract check with --point-count 1:
  exactly one supported point.
- Live bank-account question with --point-count 1:
  abstained without citations.
- Live Discussion check with --point-count 3:
  exactly three complete sentences covering a strength, limitation,
  and future direction; each supported by its cited passage.
- No validation-failure debug messages appeared in these live outputs.
  Retry recovery is covered by automated tests, not demonstrated live.

Remaining limitations:
- Point-count validation does not enforce topic coverage, sentence quality,
  quotation fidelity, or factual support.
- The one-point Abstract answer included an unnecessary citation:
  S3 directly supported the claim; S1 did not directly support the full claim.
- CLI option behavior was manually checked; dedicated CLI regression tests
  have not yet been added.
- The PDF script rebuilds its in-memory hybrid index on each invocation.
  Cached model weights are not a persisted document index.
- Verified section routing is specific to the profiled paper, not a general
  section parser.

Next:
- Add lightweight automated CLI regression coverage.
- Decide MVP deployment target: local CLI, private web app, or public service.
- Plan persistent indexing and quotation/factual-support validation against
  that target. Do not treat count enforcement as full reliability certification.

## CLI regression coverage — 2026-10-14

- Added tests/test_ask_pdf_cli.py with eight mocked CLI cases.
- Covered zero, negative, and noninteger point counts rejected before loading.
- Covered single-question and interactive count forwarding, with and without
  an explicit count.
- Covered --help exiting without PDF/model construction.
- Focused CLI suite: 8 passed.
- Full suite: 472 passed.
- No live model run needed for this test-only change.
- This supersedes the earlier note that dedicated CLI tests were absent.

Next: decide MVP deployment target, then scope persistent indexing.
Quotation fidelity and factual-support validation remain separate work.
