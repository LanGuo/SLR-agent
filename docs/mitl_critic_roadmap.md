# MITL Critic — Design and Iterative Rollout Plan

A **Machine-in-the-Loop (MITL) critic** is a second LLM pass that reviews the primary LLM's output before it reaches the human reviewer. The goal is progressive automation: start with the critic as a silent observer, graduate it to a pre-filter, eventually reach a state where humans only see escalations.

This document covers design decisions, data strategy, and a phased rollout plan rooted in the current pipeline's architecture.

---

## What the critic is (and is not)

The critic is **not** a re-run of the primary LLM on the same task. It is a distinct evaluation pass that asks:

> *"Given the input, the primary LLM's output, and the structured reasoning behind it — is this output correct, consistent, and trustworthy enough to proceed without a human?"*

It produces a structured verdict: `approve` / `flag` / `reject`, a confidence score, and a list of specific concerns. That verdict gates whether the result goes straight through, gets surfaced for human review, or is re-run.

---

## Why this codebase is unusually well-positioned for MITL

Three properties of the current design make the critic tractable today — before any training data exists:

1. **Criterion scores are structured ground truth.** Every screening decision already comes with per-criterion verdicts (`yes`/`no`/`unclear`). The critic can check internal consistency — does the overall `decision` actually follow from the scores? This requires zero learned data.

2. **Grounding scores are objective signals.** The fuzzy-match grounder already assigns a numeric confidence to every extracted field. Low grounding scores are a direct critic input.

3. **`hitl_trace.jsonl` captures human corrections.** Every time a human overrides an LLM output, the before/after diff is written to disk. This is a supervised dataset accumulating with every run, even before any critic exists.

---

## Design decisions

### 1. Same model vs. stronger model as critic

| Option | Tradeoff |
|---|---|
| Same model (`gemma4:e4b`) | No extra compute, but the critic may repeat the same errors as the primary — correlated failures |
| Larger model (`gemma4:26b`, `gemma4:31b`) | Better catch rate, especially for nuanced cases; more RAM |
| Different model family (remote API) | Best independence, but adds network dependency and cost |

**Recommended default:** Use `gemma4:26b` or `gemma4:31b` as critic when available, fall back to the same model otherwise. The `--critic-model` CLI flag should be independent of `--model`.

**Key principle:** The critic's error rate is only useful if it is *independent* of the primary. Two runs of the same model on the same input will often agree even when both are wrong. Model diversity is more valuable than model strength.

### 2. Critic sees thinking chains or not

The primary model's `thinking` chain (from Gemma 4 thinking mode) is saved in `llm_trace.jsonl`. Should the critic see it?

- **Pro:** Richer context — the critic can spot flawed reasoning, not just wrong outputs
- **Con:** The thinking chain anchors the critic toward the primary's conclusion (sycophancy / confirmation bias)

**Recommendation:** Do not pass the thinking chain to the critic by default. The critic should form an independent view. Pass it only as a second step — "here is how the model reasoned; does this change your assessment?" — to detect *reasoning errors* specifically.

### 3. Hard escalation rules vs. soft confidence thresholds

Some cases should always reach a human regardless of critic confidence:

| Hard escalation trigger | Rationale |
|---|---|
| Decision is `uncertain` (screening) | The primary already flagged doubt |
| Extraction has quarantined fields the LLM grounder also rejected | Double-failure signal |
| Critic and primary disagree | Disagreement is itself a flag |
| Review is for a high-stakes clinical question (configurable) | Risk tolerance |

Soft thresholds govern the rest: `auto_approve_threshold` (default 0.85) and `escalate_threshold` (default 0.60). Between the two thresholds, the output is surfaced to a human with the critic's concerns pre-populated.

### 4. What the critic verdict looks like

```json
{
  "verdict": "flag",
  "confidence": 0.71,
  "concerns": [
    {
      "field": "criterion_scores[2]",
      "issue": "Criterion 'Adults with hypertension' scored 'yes' but abstract describes pediatric patients",
      "severity": "high"
    }
  ],
  "suggested_decision": "exclude",
  "rationale": "Exclusion criterion EX1 is met despite being scored 'no'"
}
```

`verdict` drives routing: `approve` → auto-proceed, `flag` → surface to human with pre-populated concerns, `reject` → re-run primary LLM with critic feedback injected.

### 5. Reject-and-retry loop

When the critic rejects, the primary can be re-run with the critic's concerns as an additional user message. This is the same pattern used today for JSON parse errors. Max retries should be bounded (default 2) to avoid infinite loops.

```
primary output → critic → reject
     ↑                        │
     └── retry with concerns ─┘
                              │
                           (if still rejected after max retries)
                              ↓
                         escalate to human
```

### 6. Per-stage critic scope

The critic's job is different at each gate:

| Stage | What the critic checks |
|---|---|
| **1 — PICO** | Query completeness (all 4 PICO components present), synonym coverage, no field tags |
| **3A — Criteria** | Criteria specificity (no vague catch-alls), coverage of known study design exclusions, no contradiction between inclusion and exclusion lists |
| **3B — Screening** | Decision consistent with criterion scores; scores consistent with abstract text; no obvious misread of population/intervention |
| **5 — Extraction** | Cross-field consistency (e.g. sample size matches result), GRADE certainty plausible given risk-of-bias, no field values that are pure synonyms of the question |
| **7 — Manuscript** | Rubric criteria actually met (not just claimed), citations match claims, PRISMA flow numbers are internally consistent |

The critic at Stage 3B is the highest-value target and the easiest to bootstrap, because the criterion scores already encode the primary model's structured reasoning.

---

## Data strategy — what to collect and how to use it

### Data available today (zero runs needed)

| Source | Signal |
|---|---|
| Criterion scores | Internal consistency check — no training data needed |
| Grounding scores | Numeric confidence on every extracted field |
| `hitl_trace.jsonl` | Human corrections — supervised labels on LLM outputs |

### Data to collect prospectively

**Run the critic in shadow mode before deploying it autonomously.** This generates a disagreement log — rows where the critic would have acted differently than the human did. That log is the most important dataset.

```
shadow_critic_log:
  run_id, stage, pmid, primary_decision, critic_verdict,
  human_decision, critic_was_right (bool), human_correction_text
```

**Human confidence labels.** At HITL gates, ask the reviewer to rate their own confidence (`certain` / `borderline`). Borderline human decisions where the critic agreed are weak labels; borderline cases where they disagreed are the highest-value training examples.

**Time-on-task proxy.** Fast approvals at HITL gates (< 5 seconds per paper) are a weak signal that the decision was obvious. Slow approvals (> 60 seconds, multiple overrides) signal hard cases. Log HITL interaction timestamps in `hitl_trace.jsonl`.

### Using the data

| Dataset | Use |
|---|---|
| Shadow critic disagreements | Identify systematic failure modes — where does the critic err and why? |
| Human corrections at Stage 3B | Few-shot examples injected into the critic prompt ("here are cases where the initial decision was wrong") |
| HITL time-on-task | Active learning signal — sample hard cases preferentially for human review |
| Accumulated criterion score patterns | Rule mining — e.g. "papers where IN1=no but decision=include are always overridden" |

**Important:** Do not fine-tune the model on this data until you have > 500 examples with high inter-reviewer agreement. Before that point, few-shot prompting with the top-10 most instructive correction examples outperforms fine-tuning on a small noisy set.

---

## Iterative rollout plan

### Phase 0 — Instrumentation (now → ~10 runs)

Goal: Ensure the data pipeline is in place before building anything else.

- [ ] `hitl_trace.jsonl` already logs before/after diffs and action — verify diffs are complete
- [ ] Add `human_confidence` field to HITL trace (optional UI radio: Certain / Borderline)
- [ ] Add `hitl_duration_ms` to HITL trace (timestamp on gate open and gate close)
- [ ] Add `critic_shadow` table to SQLite: stores critic outputs alongside human decisions for comparison

No changes to pipeline behaviour. The critic does not exist yet.

---

### Phase 1 — Shadow critic (10–50 runs)

Goal: Understand critic accuracy before trusting it with anything.

**What to build:**
- `slr_agent/critic.py` — `CriticClient` wrapping `LLMClient`, one `evaluate(stage, data)` method per stage
- Critic runs after each primary LLM call, result written to `critic_shadow` table
- Shadow mode: zero effect on pipeline behaviour; results visible in `slr status <run_id> --critic`

**What to measure:**
- Agreement rate with human decisions by stage (target > 85% before Phase 2)
- False positive rate (critic flags a correct decision) — tolerate up to 20%
- False negative rate (critic approves an incorrect decision) — tolerate up to 5%
- Escalation rate (what % of papers would reach a human in Phase 2)

**Exit criteria for Phase 1:**
- ≥ 30 runs completed with HITL
- Agreement rate ≥ 85% at Stage 3B (screening) and Stage 5 (extraction)
- No systematic failure pattern not yet addressed

---

### Phase 2 — Critic as pre-filter (50–200 runs)

Goal: Reduce human review load at the two highest-volume gates (Stage 3B and Stage 5) by auto-approving high-confidence cases.

**What changes:**
- `CheckpointBroker` gains a new handler: `CriticHandler(critic, human_handler, auto_approve_threshold=0.85)`
- If `critic.verdict == "approve" and critic.confidence >= threshold` → auto-approve, no human shown
- Otherwise → hand off to human handler as today, but with critic's concerns pre-populated in the UI
- All critic decisions (including auto-approvals) written to `hitl_trace.jsonl` with `action: "critic_approved"`

**Target auto-approval rates (conservative):**
- Stage 3B screening: 50–60% (clear includes and excludes with unanimous criterion scores)
- Stage 5 extraction: 30–40% (papers with no quarantined fields and high GRADE certainty)
- Stage 1, 3A, 7: not automated in this phase

**Safety net:** Weekly human audit of a random 10% sample of auto-approved decisions. Escalate threshold automatically if audit error rate > 3%.

---

### Phase 3 — Critic-first review (200+ runs)

Goal: Flip the interaction model. Human reviews the *critic's summary of concerns*, not the raw LLM output.

**What changes:**
- For flagged papers, the HITL UI shows: critic concerns at the top, criterion scorecard, abstract — not the raw JSON
- Human resolves specific concerns ("I agree" / "I disagree and here's why"), rather than reviewing everything
- Human workload estimated at ~10–20% of Phase 2 volume

**New capability — reject-and-retry:**
- When critic confidence < `escalate_threshold`, re-run primary LLM with critic concerns as additional context before surfacing to human
- If retry output passes critic, auto-approve; if not, escalate with both the original and retry outputs shown

**Target auto-approval rates:**
- Stage 3B: 75–80%
- Stage 5: 60–70%
- Stage 3A (criteria): 50% (only for revisions the critic deems minor)

---

### Phase 4 — Async human audit (mature, high-trust)

Goal: Full SLR pipeline runs autonomously; human reviews audit samples and edge-case escalations asynchronously.

**What changes:**
- Default `hitl_mode` becomes `"critic"` (new mode alongside `"cli"` / `"ui"` / `"none"`)
- `--hitl critic` runs the pipeline fully automated with critic in autonomous mode
- All decisions written to audit log; human reviews a statistically powered sample (n ≈ 30–50 papers per run) via the existing HITL UI
- Hard escalation rules remain in place (uncertain decisions, critic-primary disagreements > 20% of papers in a run)

**Quality monitoring:**
- Per-run critic calibration report: `slr status <run_id> --audit`
- Auto-pause and alert if: escalation rate spikes, inter-run agreement drops, or audit error rate exceeds threshold

---

## The consistency check — zero training data required

The simplest critic for Stage 3B can be deployed in Phase 1 **without any training data**:

```
Given:
  - The decision: {decision}
  - The criterion scores: {criterion_scores}

Check:
  1. Is there any exclusion criterion with met=yes? If so, decision must be "exclude".
  2. Is there any inclusion criterion with met=no? If so, decision must be "exclude".
  3. Are study designs specified and none met=yes? If so, decision must be "exclude".
  4. Does the stated decision match the logical consequence of the scores?

Return: verdict (approve/flag/reject), concerns list, suggested_decision.
```

This is deterministic rule-checking, not LLM reasoning. It can be implemented in `_derive_decision` (already done) and used as a fast pre-flight before the full LLM critic call.

---

## Key risks and mitigations

| Risk | Mitigation |
|---|---|
| Critic and primary are correlated (same model) | Use a larger or different model as critic; measure agreement rate on known-incorrect cases |
| Critic is too conservative (high escalation rate) | Tune `auto_approve_threshold` per stage from empirical calibration data |
| Auto-approved errors accumulate silently | Mandatory audit sampling; per-run quality report |
| Critic sycophancy (agrees with primary to avoid conflict) | Do not show primary reasoning chain to critic; evaluate critic on disagreement quality not agreement rate |
| Cold start — no data for few-shot examples | Start with rule-based consistency check (requires no data); add few-shot examples after 10 runs |
| Criterion scores themselves are wrong | Critic should independently re-score at least one criterion per paper as a consistency probe |

---

## Implementation order

Given the current codebase, the recommended implementation sequence:

1. **`hitl_trace.jsonl` augmentation** — add `duration_ms`, `human_confidence` (Phase 0)
2. **`CriticClient`** — wraps `LLMClient`, per-stage `evaluate()` methods, shadow mode only (Phase 1)
3. **`critic_shadow` SQLite table** — stores critic outputs for comparison analysis (Phase 1)
4. **`CriticHandler`** — `CheckpointBroker` handler that routes based on verdict (Phase 2)
5. **UI critic concern panel** — pre-populated concerns at HITL gate, resolve per-concern (Phase 3)
6. **`--hitl critic` CLI mode** — autonomous critic with async audit (Phase 4)

Each phase is independently useful and does not require the next phase to work. The shadow mode in Phase 1 is the most important investment — it de-risks every subsequent phase by providing empirical calibration before any autonomous action is taken.
