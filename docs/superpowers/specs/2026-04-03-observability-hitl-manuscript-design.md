# Observability, HITL Stage Gates, and Manuscript Template Design

## Goal

Make the SLR pipeline transparent and interactive: show intermediate results at every stage, let users review and edit data at six checkpoint gates (via CLI or Gradio UI), save all intermediate outputs to disk, and generate manuscripts driven by a user-supplied template (JSON schema or PDF) with a rubric-guided HITL revision loop.

## Architecture

Three new components are introduced alongside the existing LangGraph pipeline:

**`CheckpointBroker`** — coordinates pause/resume between the pipeline thread and the human. Has two handler implementations: `CLIHandler` (prints formatted data, offers Approve/Edit/Skip prompt, opens `$EDITOR` for JSON edits) and `UIHandler` (pushes checkpoint data to a queue, blocks on a `threading.Event` until Gradio calls `resume(edited_data)`). The pipeline calls `broker.pause(stage, data) → edited_data` at stages 1, 2, 3, 5, 6, 7.

**`ProgressEmitter`** — a lightweight event sink called by each pipeline stage on completion. Fans out to: CLI `click.echo` (formatted), Gradio live log queue, and disk write to `outputs/<run_id>/stage_N_<name>.json`. Decoupled from both the broker and the pipeline graph.

**SQLite checkpointer fix** — `langgraph-checkpoint-sqlite` is a separate package not currently installed. Install it; update import in `orchestrator.py` from `langgraph.checkpoint.sqlite` to `langgraph_checkpoint_sqlite`. This is a prerequisite: without it `slr resume` is broken.

### CLI flags

```
slr run "..." --hitl cli         # terminal editing at each gate (default)
slr run "..." --hitl ui          # auto-launch Gradio, block until browser approves
slr run "..." --no-checkpoints   # fully automated, no gates
slr run "..." --template <file>  # manuscript template (JSON schema or PDF)
slr run "..." --max-results N
slr run "..." --no-fulltext
```

### Component boundaries

```
RunSession
├── CheckpointBroker(handler=CLIHandler|UIHandler)
├── ProgressEmitter(output_dir, cli_echo, gradio_queue)
└── LangGraph pipeline
    ├── pico_node      → emitter.emit(1, data) → broker.pause(1, data)
    ├── search_node    → emitter.emit(2, data) → broker.pause(2, data)
    ├── screening_node → emitter.emit(3, data) → broker.pause(3, data)
    ├── fulltext_node  → emitter.emit(4, data)   [no gate]
    ├── extraction_node→ emitter.emit(5, data) → broker.pause(5, data)
    ├── synthesis_node → emitter.emit(6, data) → broker.pause(6, data)
    └── manuscript_node→ emitter.emit(7, data) → broker.pause(7, data)
```

---

## Intermediate Results Saved to Disk

Every stage writes its structured output immediately on completion (before the checkpoint gate). Files are written by `ProgressEmitter`.

```
outputs/<run_id>/
  stage_1_pico.json          # PICO fields, query strings, detected language
  stage_2_search.json        # papers per query, dedup stats, total retrieved
  stage_3_screening.json     # per-paper: decision, reason, inclusion criteria used
  stage_4_fulltext.json      # n_fetched, n_unavailable (written even if stage skipped)
  stage_5_extraction.json    # per-paper: extracted fields, quarantined fields, GRADE
  stage_6_synthesis.json     # claims + supporting_pmids, narrative, quarantined claims
  stage_7_rubric.json        # rubric used (auto-generated or user-provided) + scores
  stage_7_draft_v1.md        # first manuscript draft
  stage_7_draft_v2.md        # revision 2 (if revision pass triggered), etc.
  <run_id>_manuscript.md     # final approved manuscript
  <run_id>_manuscript.docx   # Word export (if pandoc available)
  <run_id>_prisma.md         # PRISMA flow diagram (Mermaid)
```

---

## HITL Gate Panels

Gates fire at stages 1, 2, 3, 5, 6, 7. Stage 4 (full-text fetch) emits progress but has no gate.

### Stage 1 — PICO

**Shows:** Population, Intervention, Comparator, Outcome fields; generated PubMed query strings; detected source language.

**Editable:** All PICO fields; add/edit/remove individual query strings; override output language.

### Stage 2 — Search Results

**Shows:** Papers retrieved per query (title, PMID, source); deduplication count; total retrieved.

**Editable:**
- Exclude individual papers before screening (sets `screening_decision="excluded_manual"`)
- Remove a query and its results
- **Add paper by PMID/DOI** — fetch metadata from PubMed, add to DB
- **Add paper by title** — PubMed title search, user picks from results list
- **Upload PDF** — extract text with PyMuPDF; metadata filled from PDF or entered manually; added with `source="manual"`

Papers added at stage 2 proceed through screening normally.

### Stage 3 — Screening

**Shows:** Each paper with title, abstract snippet, AI decision (include/exclude), reason; summary counts.

**Editable:**
- Flip include↔exclude per paper (user override takes precedence over AI decision)
- Bulk-approve all AI decisions
- **Add paper** (same PMID/DOI/title/PDF options as stage 2); papers added here are marked `include` directly, bypassing AI screening

### Stage 5 — Extraction

**Shows:** Per-paper extracted fields (sample size, intervention, outcomes, effect sizes, GRADE), quarantined fields with quarantine reason.

**Editable:**
- Edit any extracted field value (edit is stored with `source="human_override"`)
- Un-quarantine a field (promotes it to grounded with human confirmation)

### Stage 6 — Synthesis

**Shows:** List of grounded claims with supporting PMIDs; narrative paragraph; quarantined claims.

**Editable:**
- Edit or delete individual claims
- Edit narrative paragraph
- Delete quarantined claims or promote to grounded

### Stage 7 — Manuscript (HITL revision loop)

**Shows:** Full draft rendered as markdown; rubric criteria with per-criterion score (`met`/`partial`/`not met`) and explanation; revision history.

**Editable:**
- Edit rubric criteria (add/remove/reword)
- Trigger LLM revision pass (rewrites `partial`/`not met` sections)
- Edit draft directly in text area
- Upload or paste a new rubric/journal guidelines to replace current rubric
- Approve and finalize

Each revision produces a new numbered draft file (`stage_7_draft_vN.md`). The loop repeats until the user approves.

---

## Manuscript Template + Rubric System

### Template formats

**JSON schema** — directly specifies sections, instructions, and optionally criteria:
```json
{
  "sections": [
    {
      "name": "Methods",
      "instructions": "Describe search strategy, eligibility criteria, data extraction.",
      "rubric_criteria": ["Specifies inclusion/exclusion criteria", "Names all databases searched"]
    }
  ],
  "style_notes": "Use passive voice. Max 6000 words."
}
```
Parsed structurally — no LLM call needed to extract structure.

**PDF (reference paper)** — PyMuPDF extracts text; LLM analyzes to infer: section structure and headings, writing style and conventions, depth of detail per section, citation density. Generates rubric criteria from this analysis.

**No template** — built-in PRISMA 2020 default structure (Abstract, Introduction, Methods, Results, Discussion, Conclusions, PRISMA flow appendix) and corresponding default rubric.

### Normalized internal representation

All three formats normalize to:
```json
{
  "sections": [
    {
      "name": "string",
      "instructions": "string",
      "rubric_criteria": ["string"]
    }
  ],
  "style_notes": "string"
}
```
Saved to `stage_7_rubric.json` (criteria only, scores added after evaluation).

### Two-pass generation

**Pass 1 — Draft:** LLM writes each section guided by template section instructions and PRISMA checklist context. Sections are written individually and concatenated.

**Pass 2 — Rubric-guided HITL revision loop:**
1. LLM scores the draft against each rubric criterion (`met`/`partial`/`not met` + one-sentence explanation)
2. Stage 7 gate shows scored rubric alongside draft — gaps are visible
3. User can:
   - **Approve** — accept draft as-is
   - **Edit rubric** — add/remove/reword criteria, then re-score
   - **Trigger revision** — LLM rewrites sections scoring `partial` or `not met`, referencing specific failing criteria
4. After revision, scores are recalculated and the loop repeats
5. Each iteration saves a new versioned draft file

---

## New Files

| File | Purpose |
|------|---------|
| `slr_agent/broker.py` | `CheckpointBroker`, `CLIHandler`, `UIHandler` |
| `slr_agent/emitter.py` | `ProgressEmitter` |
| `slr_agent/template.py` | Template parsing (JSON schema + PDF), rubric generation, rubric scoring |
| `slr_agent/ui/panels/search.py` | Stage 2 Gradio panel |
| `slr_agent/ui/panels/screening.py` | Stage 3 Gradio panel |
| `slr_agent/ui/panels/extraction.py` | Stage 5 Gradio panel |
| `slr_agent/ui/panels/synthesis.py` | Stage 6 Gradio panel |
| `slr_agent/ui/panels/manuscript.py` | Stage 7 Gradio panel (revision loop) |

### Modified files

| File | Change |
|------|--------|
| `slr_agent/orchestrator.py` | Inject broker + emitter; call `broker.pause()` and `emitter.emit()` at each stage |
| `slr_agent/cli.py` | Add `--hitl`, `--template` flags; construct broker/emitter; print stage summaries |
| `slr_agent/ui/app.py` | Wire UIHandler to Gradio; add live log panel; add template upload |
| `slr_agent/ui/panels/pico.py` | Minor: return edited dict to broker (was standalone) |
| `slr_agent/subgraphs/manuscript.py` | Two-pass generation; accept template/rubric from state |
| `slr_agent/config.py` | Add `template_path`, `hitl_mode` to `RunConfig` |
| `pyproject.toml` | Add `langgraph-checkpoint-sqlite` dependency |

---

## Testing

- Unit tests for `CheckpointBroker` (mock handler, verify pause/resume data round-trip)
- Unit tests for `ProgressEmitter` (verify files written to correct paths)
- Unit tests for `template.py` (JSON parse, PDF parse mock, rubric generation mock, rubric scoring)
- Unit tests for manual paper add (PMID lookup mock, PDF upload mock)
- Integration test: full pipeline with `CLIHandler` in auto-approve mode (no user interaction needed)
- Existing 32 unit tests must continue to pass
