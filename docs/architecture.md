# SLR Agent — Architecture

## Overview

The pipeline is a **LangGraph state machine** with seven stage subgraphs wired through a central orchestrator. Each stage writes its results to SQLite and emits a JSON file to disk. At configured stages, the orchestrator pauses for human review before continuing.

Three cross-cutting components sit alongside the pipeline:

- **`CheckpointBroker`** — coordinates pause/resume between the pipeline thread and the human (CLI prompts or Gradio UI)
- **`ProgressEmitter`** — fans out stage events to CLI echo, Gradio live log queue, and `outputs/<run_id>/stage_N_<name>.json`
- **`Database`** — SQLite store for `PaperRecord` (one row per paper per run) and the quarantine table

---

## Layer Diagram

```
┌─────────────────────────────────────────────────────┐
│                   Interface Layer                    │
│   Gradio UI (review panels, progress, export)        │
│   CLI  (slr run / slr resume / slr status / export)  │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│            Orchestrator Graph (LangGraph)            │
│   Routes between subgraphs                          │
│   Manages checkpoint gates (broker.pause)           │
│   Emits progress events (emitter.emit / .log)       │
│   Persists full state to SQLite checkpointer        │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│                  Stage Subgraphs                     │
│                                                     │
│  ① PICO        ② Search      ③ Screening            │
│  ④ Full-text*  ⑤ Extraction  ⑥ Synthesis            │
│  ⑦ Manuscript                                       │
│                                                     │
│  * optional — controlled by fetch_fulltext config   │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│                   Infrastructure                     │
│  Ollama (gemma3:12b default SQLite (checkpointer    │
│  / gemma4:27b recommended)                         │
│  PubMed Entrez API          + paper records)        │
│  bioRxiv REST API           Pandoc (→ .docx)        │
│  PyMuPDF (PDF parsing)      rapidfuzz (grounding)   │
└─────────────────────────────────────────────────────┘
```

---

## Component Boundaries

```
slr run "..."
│
├── ProgressEmitter(output_dir, run_id, echo=click.echo)
│     emit(stage, data)  →  disk write  +  CLI echo  +  Gradio queue
│     log(message)       →  CLI echo    +  Gradio queue
│
├── CheckpointBroker(handler=CLIHandler | UIHandler | NoOpHandler)
│     pause(stage, name, data) → blocks until human approves → returns edited_data
│
└── create_orchestrator(db, llm, config, broker, emitter)
      │
      ├── pico_node       → pico_sg.invoke(...)   → emitter.emit(1) → broker.pause(1)
      ├── search_node     → search_sg.invoke(...) → emitter.emit(2) → broker.pause(2)
      ├── screening_node  → LLM criteria → broker.pause(3, "screening_criteria")
      │                     → screening_sg.invoke(...) → emitter.emit(3) → broker.pause(3)
      ├── fulltext_node   → fulltext_sg.invoke(...)→ emitter.emit(4)   [no gate]
      ├── extraction_node → extraction_sg.invoke(...)→ emitter.emit(5) → broker.pause(5)
      ├── synthesis_node  → synthesis_sg.invoke(...)→ emitter.emit(6)  → broker.pause(6)
      └── manuscript_node → manuscript_sg.invoke(...)→ emitter.emit(7) → broker.pause(7)
                            (revision loop if stage 7 in checkpoint_stages)
```

---

## Pipeline Flow

```
Question ──▶ ① PICO
                │  gate: edit PICO fields, query strings, search config
                ▼
             ② Search (PubMed + bioRxiv)
                │  gate: exclude papers, add by PMID/PDF
                ▼
             ③ Screening
                │  gate A: edit inclusion/exclusion criteria (before LLM screens)
                │  gate B: override per-paper include/exclude decisions
                ▼
             ┌── fetch_fulltext enabled? ──▶ ④ Full-text fetch
             │                                     │ (no gate)
             └─────────────────────────────────────┘
                │
                ▼
             ⑤ Data Extraction + GRADE
                │  gate: edit extracted fields, un-quarantine
                ▼
             ⑥ Evidence Synthesis
                │  gate: edit narrative, claims
                ▼
             ⑦ Manuscript
                │  gate: revision loop (approve / edit rubric / trigger LLM revision)
                ▼
             outputs/<run_id>/*.md + *.docx
```

---

## State Design

### OrchestratorState (LangGraph graph state)

Intentionally slim — holds routing signals and summary counts only. All bulk evidence lives in SQLite.

```python
class OrchestratorState(TypedDict):
    run_id: str
    raw_question: str
    config: RunConfig
    pico: PICOResult | None
    search_counts: SearchCounts | None
    screening_counts: ScreeningCounts | None
    fulltext_counts: FulltextCounts | None
    extraction_counts: ExtractionCounts | None
    synthesis_path: str | None
    manuscript_path: str | None
    template: dict | None
    manuscript_draft_version: int
    date_from: str | None          # search date range, e.g. "2000-01-01"
    date_to: str | None            # search date range, e.g. "2026-12-31"
    search_sources: list[str]      # ["pubmed"] | ["pubmed", "biorxiv"]
    max_results: int
    screening_criteria: dict | None  # {inclusion_criteria, exclusion_criteria, study_designs}
    current_stage: str
    checkpoint_pending: bool
```

### PICOResult

```python
class PICOResult(TypedDict):
    population: str
    intervention: str
    comparator: str
    outcome: str
    query_strings: list[str]   # editable at Stage 1 gate
    source_language: str       # ISO 639-1, e.g. "fr"
    search_language: str       # always "en"
    output_language: str       # same as source_language
```

### RunConfig

```python
class RunConfig(TypedDict, total=False):
    checkpoint_stages: list[int]   # default [1, 2, 3, 5, 6, 7]
    fetch_fulltext: bool
    output_format: Literal["markdown", "word", "both"]
    pubmed_api_key: str | None
    max_results: int               # per source, default 500
    search_sources: list[str]      # ["pubmed"] | ["pubmed", "biorxiv"]
    date_from: str | None          # "YYYY-MM-DD", default "2000-01-01"
    date_to: str | None            # "YYYY-MM-DD", default today
    template_path: str | None      # JSON schema or PDF reference paper
    hitl_mode: str | None          # "cli" | "ui" | "none"
```

### PaperRecord (SQLite only)

```python
class PaperRecord(TypedDict):
    pmid: str              # "99999" for PubMed; "biorxiv:10.1101/..." for bioRxiv
    run_id: str
    title: str
    abstract: str
    fulltext: str | None
    source: Literal["abstract", "fulltext"]
    screening_decision: Literal["include", "exclude", "uncertain", "excluded_manual"]
    screening_reason: str
    extracted_data: dict
    grade_score: GRADEScore
    provenance: list[Span]
    quarantined_fields: list[QuarantinedField]
```

---

## HITL Mechanism

### CheckpointBroker

The broker sits between the pipeline thread and the human. The pipeline calls `broker.pause(stage, name, data)` and blocks. The broker delegates to its handler:

- **`NoOpHandler`** — returns data unchanged (fully automated)
- **`CLIHandler`** — prints formatted data, prompts Approve/Edit/Skip, uses `click.prompt` for field-by-field inline editing
- **`UIHandler`** — pushes data to a `Queue`, blocks on a `threading.Event` until Gradio calls `ui_handler.resume(edited_data)`

```
pipeline thread          broker                  human
     │                     │                       │
     │── pause(3, data) ──▶│                       │
     │                     │── push to queue ──────▶│
     │   (blocks)          │                       │  [reviews in Gradio / CLI]
     │                     │◀── resume(edited) ────│
     │◀── returns edited ──│                       │
     │                     │                       │
```

### Stage 3 — Two-gate screening

Stage 3 has two broker calls: one before screening (criteria review) and one after (per-paper decision review):

1. LLM generates inclusion/exclusion criteria from PICO
2. `broker.pause(3, "screening_criteria", criteria)` — user edits criteria
3. Criteria saved to `stage_3_screening_criteria.json`
4. Screening subgraph runs with user-edited criteria
5. `broker.pause(3, "screening", decisions)` — user reviews per-paper decisions

### Stage 7 — Revision loop

Stage 7 runs in a loop while the user keeps requesting revisions:

1. Manuscript subgraph generates draft + rubric scores
2. `broker.pause(7, "manuscript", {draft, rubric, draft_version})` — user reviews
3. If `action == "revise"`: re-run manuscript subgraph → new versioned draft → loop
4. If `action == "approve"`: exit loop, finalize

---

## Grounding Layer

Every LLM-extracted value is fuzzy-matched against its source text using **rapidfuzz** `token_sort_ratio` with threshold 85.

```
extracted_value ─── fuzzy_match ──▶ source_text
                          ├── score ≥ 85 → Span(char_start, char_end, text) stored as provenance
                          └── score < 85 → QuarantinedField written to quarantine table
```

**Quarantine behaviour:**
- Quarantined fields are stored, not dropped — the paper is not discarded
- They appear in HITL gates for manual resolution: accept / edit / discard
- PRISMA flow diagram includes quarantine counts as a data quality signal
- Full quarantine table queryable via SQLite or `slr status <run_id>`

**Synthesis grounding (Stage 6):** Gemma 4 is asked to cite the PMIDs that support each synthesised claim. Claims with zero citations are quarantined.

---

## Search Design

### PubMed (Entrez)

- One Entrez `esearch` call per query string in `pico.query_strings`
- `mindate` / `maxdate` / `datetype=pdat` for date range filtering
- `efetch` in batches of 200 to retrieve titles + abstracts
- Rate limiting: 3 req/s without API key, 10 req/s with key

#### Result cap and relevance ordering

PubMed returns results in **relevance order** by default (best matches first within each query). The cap is distributed evenly across query strings to keep the total within `max_results`:

```
per_query_cap = max_results // len(query_strings)
```

For example, `max_results=50` with 4 query strings → 12–13 results per query. After merging (deduplicating by PMID), a final slice enforces the hard cap:

```
all_pmids = list(ordered_union_of_all_queries)[:max_results]
```

This means the cap is applied to the **total** retrieved, not per-query, and the most relevant papers from each query strand are preferred over less relevant ones. There is no cross-query re-ranking — papers are ordered by which query retrieved them first, then by PubMed relevance within that query.

### bioRxiv

- Date-range API: `https://api.biorxiv.org/details/biorxiv/{date_from}/{date_to}/0/json`
- No keyword search API — fetches recent preprints by date range, capped at `max_results`
- PICO relevance filtering happens in Stage 3 screening (LLM-based), not at retrieval time
- Failures (network, timeout) are silently skipped; pipeline continues with PubMed results only

#### Combined volume

With both sources enabled, the total paper count entering screening can be up to `2 × max_results` (PubMed cap + bioRxiv cap independently). Set `search_sources: ["pubmed"]` to stay within `max_results` total.

### Search configuration (user-editable at Stage 1 gate)

`search_sources`, `max_results`, `date_from`, `date_to` are all exposed at the Stage 1 PICO gate. User edits flow into state and are used by the search subgraph at Stage 2.

---

## Manuscript Template System

Three template formats all normalize to the same internal representation:

```json
{
  "sections": [
    {
      "name": "Methods",
      "instructions": "Describe search strategy and eligibility criteria.",
      "rubric_criteria": ["Specifies inclusion/exclusion criteria", "Names all databases searched"]
    }
  ],
  "style_notes": "Use passive voice. Max 6000 words."
}
```

| Format | How parsed |
|---|---|
| JSON schema | Direct parse — no LLM call |
| PDF (reference paper) | PyMuPDF text extraction → Gemma 4 infers sections, style, rubric criteria |
| None | Built-in PRISMA 2020 default (`DEFAULT_PRISMA_TEMPLATE`) |

**Two-pass generation:**
1. **Draft pass** — Gemma 4 writes each section guided by its `instructions`
2. **Rubric pass** — Gemma 4 scores each criterion (`met` / `partial` / `not met` + explanation)

The Stage 7 HITL gate shows the scored rubric alongside the draft. Triggering a revision re-runs Pass 1 targeting only `partial` / `not met` sections.

---

## ProgressEmitter

`ProgressEmitter` is the event sink that decouples the pipeline from its consumers. Every stage calls `emitter.emit(stage, data)` on completion and `emitter.log(message)` for progress updates.

Each call fans out to:

1. **Disk** — `outputs/<run_id>/stage_N_<name>.json` (always)
2. **CLI** — `click.echo` (if `echo` provided)
3. **Gradio** — put to `Queue` polled by `app.load(..., every=1)` (if Gradio is running)

The `name` parameter overrides the default stage name for custom filenames (e.g., `stage_3_screening_criteria.json` vs `stage_3_screening.json`).

---

## SQLite Checkpointer

LangGraph's `SqliteSaver` is used as the graph checkpointer. After every node completion, full graph state is persisted to `slr_runs.db`. This enables `slr resume <run_id>` to re-enter at the exact node where the run paused or failed.

**Important:** `SqliteSaver` must be constructed as `SqliteSaver(conn)` with an already-open `sqlite3.connect()` connection. `SqliteSaver.from_conn_string()` is a context manager (yields, doesn't return) and cannot be used directly outside a `with` block.

---

## Testing Strategy

| Layer | Tool | Ollama needed |
|---|---|---|
| Unit tests | pytest + MockLLM | No |
| Integration tests | pytest + MockLLM | No |
| E2E smoke test | pytest -m e2e | Yes |

`MockLLM` matches prompts by substring (case-sensitive `in` check) and returns pre-registered structured responses. Tests register keys matching the start of each LLM prompt.

Grounding regression test: injects a `PaperRecord` with a deliberately hallucinated extracted field (no matching span) and verifies the grounding node quarantines it rather than passing it through.
