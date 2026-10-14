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
