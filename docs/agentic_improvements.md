# Agentic Improvements — Design Notes

This document describes five improvements made to the SLR Agent pipeline to make it more autonomous, more trustworthy, and less dependent on human intervention for routine quality checks. Each improvement targets a specific gap between a pipeline that produces output and one that can reason about the quality of its own output.

---

## Motivation

The original pipeline was a well-structured seven-stage sequence with HITL gates at each stage. It was correct and auditable, but it relied on the human reviewer to catch several categories of problems that an automated pass could catch first:

- **Provenance opacity**: extracted fields were grounded to source text, but there was no way to know *how* a value was matched — verbatim quote, paraphrase, or LLM inference.
- **Manual grounding trigger**: fields that failed fuzzy matching were quarantined and only received LLM re-verification if the human explicitly clicked a button at Gate 5.
- **Silent evidence gaps**: the synthesis stage identified what the evidence *said* but never surfaced what it *didn't answer* — the reviewer had to infer open questions from reading the narrative.
- **Hallucinated citations**: the manuscript writer was asked not to hallucinate PMIDs but had no structural mechanism preventing it; citations were written from parametric memory.
- **No self-critique before Gate 7**: the human at Gate 7 received a draft and a rubric score, but no adversarial assessment of whether the draft was internally consistent or whether earlier pipeline stages needed to be redone.

The five improvements below address each of these gaps in order, from the most foundational (data layer) to the highest-level (manuscript critique).

---

## Improvement 1: Provenance type on `Span`

**Where:** `slr_agent/db.py`, `slr_agent/grounding.py`

**What changed:** The `Span` TypedDict, which records where an extracted value was found in the source text, now carries a `provenance_type` field:

| Value | Meaning |
|---|---|
| `"direct"` | Exact substring match — the value appears verbatim in the source |
| `"paraphrased"` | Fuzzy `token_set_ratio` match — the value is a paraphrase of the source |
| `"inferred"` | LLM-confirmed — the value was quarantined by fuzzy matching but confirmed by a second-pass LLM call (see Improvement 2) |

**Why it matters:** The audit trail previously only said whether a value was grounded or quarantined. With `provenance_type`, downstream stages and human reviewers can distinguish a verbatim quote (high confidence, GRADE-relevant) from a paraphrase (lower confidence, worth a second look) from an LLM inference (confirmed but without a character span). This distinction is particularly important for GRADE evidence quality assessments, where the directness of evidence affects the certainty rating.

---

## Improvement 2: Automatic LLM grounding before quarantine

**Where:** `slr_agent/subgraphs/extraction.py`

**What changed:** After fuzzy grounding runs in `_extract_node`, any fields that failed the fuzzy threshold now automatically go through a second-pass LLM grounding call (`_auto_llm_ground`) before being written to the quarantine table. The LLM is asked: *"Does the source text support this value, even if phrased differently?"* Fields confirmed by the LLM (`supported: true`) are promoted into `extracted_data` with `confidence=80.0` and `provenance_type="inferred"`. Fields the LLM also cannot confirm are quarantined as before.

The manual "LLM Ground" button at Gate 5 has been removed — auto-grounding replaces it for all papers automatically.

**Why it matters:** Fuzzy matching is conservative by design: it rejects valid extractions that are heavily paraphrased, abbreviated, or numerically expressed differently from the source (e.g., `"96"` vs `"ninety-six"`, `"MI"` vs `"myocardial infarction"`). Previously, every such case required a human to notice the quarantine flag and click a button. Auto-grounding converts this from a reactive human task to a proactive automated step, reducing Gate 5 review burden while maintaining the audit trail. The LLM call is wrapped in a try/except so grounding failures are non-fatal — a failure leaves the field quarantined rather than aborting the extraction run.

---

## Improvement 3: Structured open questions from synthesis

**Where:** `slr_agent/subgraphs/synthesis.py`, `slr_agent/orchestrator.py`

**What changed:** The synthesis LLM prompt now requests a third output field alongside `claims` and `narrative`: `unresolved_questions`. Each question has the shape:

```json
{
  "question": "Does the effect persist beyond 12 months?",
  "relevant_pmids": ["12345678"],
  "importance": "high"
}
```

`importance` is constrained to `["high", "medium", "low"]`. The questions are returned in state as `unresolved_questions` and included in the Gate 6 emit data so the reviewer sees them before approving the manuscript stage.

**Why it matters:** A synthesis that only reports what studies found tells you nothing about what they failed to investigate. Surfacing open questions at Gate 6 serves two purposes: (1) it gives the human reviewer an explicit checklist of gaps to look for in the draft, and (2) it provides the adversarial reviewer (Improvement 5) with signal about what the manuscript should and should not claim. Questions with `importance: "high"` that go unanswered in the manuscript are exactly the kind of finding that should trigger a MAJOR or FATAL critique.

---

## Improvement 4: Two-pass citation anchoring in the manuscript

**Where:** `slr_agent/subgraphs/manuscript.py`

**What changed:** Manuscript drafting is now structurally split into two passes:

**Pass 1 — Writer:** Each section is drafted as pure prose. The writer prompt explicitly forbids inline citations: *"Do NOT include inline citations (no PMID numbers, no author-year, no brackets). Citations will be added in a separate verification pass."* This eliminates the structural incentive to hallucinate PMIDs.

**Pass 2 — Verifier (`_verify_citations_node`):** After all sections are drafted, the verifier reads the synthesis file to extract grounded claims (lines of the form `- claim text [PMID1, PMID2]`). It sends the draft and the claim list to the LLM and asks it to identify which section of the manuscript each claim appears in. It then injects `(PMID: X, Y)` markers at the first matching line per claim.

All PMID anchors are sourced from the synthesis grounding pass — a claim with no supporting PMIDs cannot receive a citation tag, regardless of what the writer wrote.

**Why it matters:** Citation hallucination in AI-generated scientific text is a known failure mode. The structural fix — separating prose generation from citation anchoring — ensures that every inline PMID in the final manuscript traces directly to a grounded synthesis claim. The verifier is non-blocking: on LLM failure, the draft is written without citations rather than aborting the node.

**Known limitations:** The verifier only sees the first 6000 characters of the draft due to LLM context budget. Claims appearing in sections past that point fall back to section-header injection. If two distinct claims map to the same sentence, only the first is anchored inline; the second uses the section fallback.

---

## Improvement 5: Adversarial reviewer with stage-rerun trigger

**Where:** `slr_agent/subgraphs/manuscript.py`, `slr_agent/orchestrator.py`

**What changed:** A fourth pass runs after the citation verifier and before Gate 7: `_adversarial_review_node`. It reads the draft and the synthesis with `think=True` (Gemma 4's extended reasoning mode) and returns a structured critique:

```json
{
  "issues": [
    {
      "severity": "FATAL",
      "section": "Methods",
      "issue": "Inclusion criteria were not applied consistently — two included papers are case reports.",
      "suggestion": "Re-screen abstracts with corrected criteria.",
      "rerun_stage": "screening"
    }
  ]
}
```

**Severity levels:**

| Severity | Meaning | Action |
|---|---|---|
| `FATAL` | The manuscript is wrong or misleading; a prior stage needs to be redone | Triggers automatic rerun of the named stage before Gate 7 |
| `MAJOR` | Significant flaw requiring human attention before submission | Surfaced at Gate 7 |
| `MINOR` | Style, clarity, or minor accuracy issue | Surfaced at Gate 7 |

**Stage-rerun logic:** When FATAL issues name a valid `rerun_stage` (`screening`, `extraction`, or `synthesis`), the orchestrator reruns only that stage — not its implicit downstream stages. The rerun is bounded to one attempt. If the redrafted manuscript still has FATAL issues, they surface at Gate 7 alongside the reviewer's assessment so the human can take action.

**Why it matters:** Without a self-critique pass, Gate 7 asks the human to assess a draft with no prior adversarial context. The human may not notice that the Methods section describes inclusion criteria inconsistent with the papers in the Results table, or that the Abstract claims an effect size not present in any extracted data. The adversarial reviewer makes these inconsistencies explicit and, for the most severe cases, automatically corrects them before the human sees the draft. This shifts the human's role at Gate 7 from "find the problems" to "approve or redirect the fixes" — a significant reduction in cognitive load for high-stakes reviews.

The reviewer is non-blocking: on LLM failure or schema non-compliance, it returns `{"issues": []}` rather than aborting the draft node.

---

## How the improvements work together

The five improvements form a chain from data quality to manuscript quality:

```
Extraction
  └── Fuzzy grounding
        ├── "direct" / "paraphrased" provenance  [Improvement 1]
        └── Auto LLM grounding before quarantine [Improvement 2]

Synthesis
  └── Returns unresolved_questions              [Improvement 3]
        └── Surfaced at Gate 6

Manuscript (four passes)
  ├── Pass 1: Writer — prose only, no citations
  ├── Pass 2: Verifier — anchor PMIDs from synthesis claims [Improvement 4]
  ├── Pass 3: Adversarial reviewer — FATAL/MAJOR/MINOR      [Improvement 5]
  │         └── FATAL → auto-rerun named prior stage
  └── Gate 7: Human sees draft + rubric + adversarial issues
```

The provenance type (Improvement 1) feeds the audit trail. Auto-grounding (Improvement 2) ensures that by the time the synthesis and manuscript stages run, extraction data is as complete as the evidence allows. Open questions (Improvement 3) give the adversarial reviewer signal about what the manuscript should address. Citation verification (Improvement 4) ensures those claims are traceable. The adversarial reviewer (Improvement 5) closes the loop by checking whether the manuscript accurately represents the evidence — and escalating if it does not.

---

## Relationship to the MITL critic roadmap

These improvements are separate from but complementary to the MITL (Machine-in-the-Loop) critic described in `docs/mitl_critic_roadmap.md`. The MITL critic operates at the screening and extraction gates (Stages 3 and 5), auto-approving high-confidence LLM decisions to reduce human review volume. The agentic improvements documented here operate later in the pipeline (extraction data quality, synthesis output, manuscript generation) and focus on correctness and completeness rather than throughput. Both sets of improvements share the principle that the system should surface specific, structured concerns to humans rather than raw LLM output.
