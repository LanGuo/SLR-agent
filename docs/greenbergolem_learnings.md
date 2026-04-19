# Learnings from greenbergolem — Improvement Roadmap

Analysis of https://github.com/jmandel/greenbergolem, a citation network analysis and evidence auditing platform by Josh Mandel. Same problem space (biomedical literature analysis with LLMs), different lens (auditing how claims propagate through citation networks vs. synthesizing what evidence says). Five concrete improvements identified for the SLR pipeline.

---

## Improvement 1: SHA-256 LLM call caching

**Problem:** If an extraction or screening run is interrupted halfway through (network drop, OOM, user Ctrl-C), the entire stage re-runs from the beginning. For a 200-paper run, this wastes significant time and Ollama compute.

**Greenbergolem's approach:** Every subagent invocation is keyed by SHA-256 hash of its inputs (prompt + model). If the cache file exists, the result is returned immediately. Concurrent tasks use the same cache — two workers that happen to get the same paper don't both call the LLM.

**Our adaptation:** Wrap `LLMClient.chat()` with an optional disk cache keyed on `sha256(model + messages_json + schema_json)`. Cache files live in `outputs/<run_id>/.llm_cache/`. The cache is run-scoped (not shared across runs) to avoid stale data from model changes.

**Impact:** Interrupted runs resume from the exact paper they stopped at. Re-runs after human edits at Gate 5 don't re-extract already-processed papers.

---

## Improvement 2: Unified judgment schema across all LLM decisions

**Problem:** Every LLM decision point in the pipeline has its own ad-hoc structure:
- Screening: `{decision, reason, criterion_scores}`
- Extraction: `{field: value}` + separate grounding spans
- GRADE: `{certainty, risk_of_bias, ..., rationale}`
- Adversarial review: `{issues: [{severity, section, issue, suggestion}]}`

There is no common envelope for confidence, justification, and provenance. The audit trail is fragmented across different JSON shapes.

**Greenbergolem's approach:** A `makeJudgmentSchema` factory produces a standard envelope for every LLM decision: `{value, confidence: 0–1, justification, evidenceSpans: [{text, source}], provenance: {agent, model, timestamp}}`.

**Our adaptation:** Add a `JudgmentEnvelope` TypedDict wrapping every LLM decision. Screening, GRADE, and adversarial review decisions gain a `confidence` float and `evidence_spans` list. The `llm_trace.jsonl` already captures all calls; the envelope makes confidence and spans part of the structured output rather than buried in free-text rationale.

**Impact:** Consistent audit trail. Downstream code (Gate UIs, synthesis, adversarial reviewer) can read confidence directly rather than inferring it from prose. Enables future filtering like "auto-approve screening decisions with confidence > 0.9."

---

## Improvement 3: Citation network layer for circular citation detection

**Problem:** GRADE evidence quality assessment treats each included paper as independently contributing evidence. In practice, many papers in a corpus cite each other — if 15 papers all support a claim but 14 of them are citing the same original study, the effective evidence base is 1, not 15. GRADE's five dimensions don't capture this.

**Greenbergolem's approach:** Citation edges are the primary data structure. Every citation site (section, paragraph, stance) is labeled individually, then aggregated. Authority scores weight papers by supportive in-degree + PageRank. Distortion patterns (echo chambers, circular clusters, retraction propagation) are computed from the graph.

**Our adaptation:** After corpus assembly (end of Stage 2), build a lightweight citation graph:
- For each included paper, parse reference lists from full-text XML (already fetched in Stage 4)
- Identify which references resolve to other papers in the corpus (cross-citation edges)
- Compute: cluster coefficient, echo-chamber ratio (what fraction of supportive citations are within-corpus), and whether a single paper accounts for >50% of citations
- Surface a `citation_network_summary` in the Stage 6 synthesis emit and in the GRADE rationale

This is strictly additive — no existing stages change behavior. The network summary is advisory to the reviewer and the adversarial reviewer.

**Impact:** Catches the most common form of evidence inflation in medical literature. Adds a dimension to the audit trail that no existing SLR tool provides.

---

## Improvement 4: Per-result-instance extraction (occurrence-first)

**Problem:** We extract 8 fields per paper in a single LLM pass. Papers that report multiple subgroups, timepoints, or outcomes collapse everything into one record. A trial with 12-month and 24-month endpoints produces one `follow_up_duration` value, losing half the data.

**Greenbergolem's approach:** Label each individual citation *occurrence* (section, paragraph, context) before aggregating to paper-level edges. Fine-grained location enables detecting within-paper stance variability ("mostly supportive, one critical mention in the limitations").

**Our adaptation:** Change the extraction schema from one record per paper to one record per *result instance*. The LLM first segments the paper into result instances (e.g., "primary outcome at 12 months," "secondary outcome at 24 months," "subgroup: diabetic patients"), then extracts the 8 fields for each instance. Multiple `ExtractionRecord`s per paper are stored and shown at Gate 5. Synthesis and GRADE aggregate across instances.

**Impact:** Significantly more complete data for multi-arm or multi-timepoint trials. Enables synthesis claims like "effect size at 12 months: X; at 24 months: Y" rather than a single blurred value.

---

## Improvement 5: Taxonomy-driven GRADE dimension generation

**Problem:** GRADE scoring uses a fixed prompt with five hardcoded dimensions. Different SLR domains (clinical trials, basic science, epidemiology, diagnostic accuracy) have different relevant dimensions. Diagnostic accuracy reviews need sensitivity/specificity considerations; basic science reviews need replication and model organism considerations that aren't in the standard GRADE rubric.

**Greenbergolem's approach:** Evidence classes and analysis dimensions are defined in a taxonomy. The resolver auto-generates assessment specifications from the taxonomy structure — add a new evidence class and new metrics appear automatically.

**Our adaptation:** Replace the hardcoded GRADE prompt with a configurable `grade_taxonomy.json` that defines: which dimensions to assess, their allowed values, and the prompt text for each. The default taxonomy implements standard GRADE (certainty, risk_of_bias, inconsistency, indirectness, imprecision). Users can extend or override for their domain. The GRADE LLM call iterates over the taxonomy dimensions rather than using a hardcoded schema.

**Impact:** Makes the pipeline adaptable to non-clinical SLR domains without code changes. Aligns with the existing template system (users already customize the manuscript template).

---

## Implementation priority

| # | Improvement | Effort | ROI | Breaking changes |
|---|---|---|---|---|
| 1 | LLM call caching | Low | High | No |
| 2 | Unified judgment schema | Medium | Medium | Minor (schema additions) |
| 3 | Citation network layer | Medium | High | No (additive) |
| 4 | Per-result-instance extraction | High | High | Yes (DB schema, Gate 5 UI) |
| 5 | Taxonomy-driven GRADE | Medium | Medium | No (opt-in) |

Recommended order: 1 → 3 → 2 → 5 → 4 (cache first for immediate pain relief, citation network second as it's additive and novel, then schema consistency, then GRADE flexibility, then the breaking extraction refactor last).
