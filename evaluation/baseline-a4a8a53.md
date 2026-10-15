# RAGForge baseline evaluation

Code checkpoint: a4a8a53
PDF title / filename:
Date:

| Test | Exact question | Expected answer and source page | Actual answer / status | Evidence quality | Response time | Pass / Fail / Unclear |
|---|---|---|---|---|---|---|
| 1 — Study objective | What is the main objective of this study? | Identify, compare, and integrate concepts explaining AI tokens as units of computation, workload demand, infrastructure cost, service pricing, and economic value creation; PDF p. 10. | answered; matches the cited aim. | S1 directly supports the answer; excerpt ends mid-word after the supporting sentences. | 8.32 s | Pass |

### Replacement PDF evaluation — 2026-10-15
Document: AI_Tokenomics_and_the_Economics_of_Inference_at_Scale.pdf. User reports added printed page numbers. Citation-to-viewer page mapping not yet visually verified.
| Test | Exact question | Expected answer and source page | Actual answer / status | Evidence quality | Response time | Pass / Fail / Unclear |
|---|---|---|---|---|---|---|
| 1 rerun — Study objective | What is the main objective of this study? | Identify, compare, and integrate concepts explaining AI tokens as units of computation, workload demand, infrastructure cost, service pricing, and economic value creation; cited PDF p. 10. | answered; matches the cited aim. | S1 directly supports the answer; chunk ends mid-word at "thre"; page mapping pending visual check. | 10.06 s | Pass for answer/evidence; page check pending |
| 2 — Exact count of dimensions | How many interdependent dimensions of AI tokenomics does the study identify, and what are they? | Four: token utility, token demand, token supply, and token monetization; abstract, cited PDF p. 1. | Correct count and all four names; status not included in pasted result. | S1 and S2 directly support count and names; 3 chunks supplied, 2 cited; truncated chunk starts; page mapping pending visual check. | 4.16 s | Pass for answer/evidence; page check pending |

Test 2 verification — 2026-10-15: PASS. Latest run: status answered; 3.79 s; four dimensions correctly identified and named. S1 and S2 directly support the answer. User visually confirmed the S2 sentence on PDF viewer page 1, also printed page 1. S1 page 18 not separately verified. Earlier run timings remain separate.
| 3 — Metric explanation | Why does the study consider cost per token an incomplete measure, and what alternative metric does it propose? | Token volume does not necessarily indicate usefulness, reliability, or monetizable outcomes; alternative: cost per useful output. | answered; both parts correct. | S1 supports the explanation; S2 supports the alternative. User verified S2 heading on viewer page 32, printed page 32. Formula visible in PDF but absent from supplied S2 text. | 9.03 s | Pass |
| 4 — Formula retrieval | What is the formula for cost per monetizable output, including its numerator and denominator? | Total AI Operating Cost divided by Number of Outputs that Generate Measurable Value; section 5.4, user-reported p. 33. | answered; lists both correct components but omits labels and division relationship. | S1 retrieves both components on PDF p. 33; fraction layout flattened into text. | 8.24 s | Partial pass: components correct; full formula incomplete |

Test 4 follow-up diagnostic: Question: "According to section 5.4, how is cost per monetizable output calculated? Explicitly state what is divided by what." Status: answered. Response time: 6.23 s. Answer correctly states Total AI Operating Cost / Number of Outputs that Generate Measurable Value, citing S1 on PDF p. 33. PASS for this follow-up; original Test 4 remains partial pass. Fraction layout remains flattened in evidence.
| 5 — Measurement limitations | According to the study, why can cost per useful output and cost per monetizable output be difficult to calculate? | Assessing usefulness or monetizability may require human judgement, task-specific evaluation, or business outcome measurement. | answered; all three reasons correct. | S1 directly supports the answer on PDF p. 33. Extra statement from S3 p. 8 refers to a different limitation and is potentially confusing. | 4.83 s | Pass — minor relevance issue |
| 6 — Unsupported numerical claim | What exact percentage reduction in total AI operating cost does the study report from adopting cost per monetizable output? | Return a supported percentage if reported; otherwise avoid inventing one. Document-wide ground truth not yet established. | abstained; verifier could not verify the proposed answer against its cited evidence. | 3 evidence chunks supplied; passages and rejected candidate not reviewed. | 3.97 s | Provisional pass — safe abstention; retrieval correctness unverified |

Test 6 observation: No cited passages displayed for the abstained response, although the UI reports 3 evidence chunks supplied. Retrieved evidence and rejected candidate could not be inspected. Result remains provisional; document-wide absence of the requested percentage is not confirmed.
| 7 — Supported list retrieval | According to section 5.4, which four types of measurable value can outputs contribute to when assessing cost per monetizable output? | Revenue, productivity, cost savings, and risk reduction; established from p. 33. | answered; incorrectly returned raw token output, usable output, validated output, and monetizable output. | S2 p. 7 supports output categories, not the requested value types; its definition also contains relevant value contributions. | 4.45 s | Fail — cited but nonresponsive answer |

Test 7 follow-up diagnostic: Asked to complete the section 5.4 statement and list contributions rather than output categories. Status: answered. Response time: 4.69 s. Correctly returned revenue, productivity, cost savings, or risk reduction, supported by S1 on PDF p. 33. PASS with duplicate-output caveat: identical answer listed twice. Original Test 7 remains FAIL.

## Baseline findings summary
- Tests 1–3: Pass; chunk-boundary and formula-extraction concerns noted.
- Test 4: Partial pass — formula test.
- Test 5: Pass with answer-relevance caveat — measurement limitations.
- Test 6: Provisional pass for safe abstention; no cited passages displayed, so retrieved evidence could not be reviewed.
- Test 7: Fail — returned output categories instead of the requested value contributions.
- Test 7 follow-up: Pass with duplicate-output caveat; correct contributions supported by PDF page 33.
- Priority: Investigate answer relevance in generation and verification, then duplicate-answer handling.
- Preserve original test outcomes; follow-up successes do not replace original failures.
