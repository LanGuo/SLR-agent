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
    search_sources: list[str]      # ["pubmed"] | ["pubmed", "biorxiv"] | ["pubmed", "biorxiv", "arxiv"]
    max_results: int
    screening_criteria: dict | None  # {inclusion_criteria, exclusion_criteria, study_designs}
    unresolved_questions: list[dict] | None  # open questions from synthesis (importance: high/medium/low)
    citation_network: dict | None  # CitationNetworkSummary.to_dict(); None if fulltext not fetched
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
    checkpoint_stages: list[int]   # default [1, 2, 3, 5, 7]
    fetch_fulltext: bool
    output_format: Literal["markdown", "word", "both"]
    pubmed_api_key: str | None
    max_results: int               # total cap (PubMed: distributed across queries), default 500
    model: str                     # Ollama model tag, default "gemma4:e4b"
    search_sources: list[str]      # ["pubmed"] | ["pubmed", "biorxiv"] | ["pubmed", "biorxiv", "arxiv"]
    date_from: str | None          # "YYYY-MM-DD", default "2000-01-01"
    date_to: str | None            # "YYYY-MM-DD", default today
    template_path: str | None      # JSON schema or PDF reference paper
    hitl_mode: str | None          # "cli" | "ui" | "none"
    screening_batch_size: int      # abstracts per LLM call during screening, default 3
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
    criterion_scores: list[dict]   # per-criterion scores; each: {criterion, type, met, note}
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

### Stage 2 — Search gate

After retrieval, all papers are surfaced at `broker.pause(2, "search", data)`:

- Set `excluded=true` on any paper to remove it before LLM screening (`screening_decision` → `excluded_manual`)
- Add a paper by PMID by including an entry with `manual_add=true` — the pipeline fetches its abstract from PubMed

### Stage 3 — Two-gate screening

Stage 3 has two broker calls: one before screening (criteria review) and one after (per-paper decision review):

1. LLM generates inclusion/exclusion criteria from PICO
2. `broker.pause(3, "screening_criteria", criteria)` — user edits criteria
3. Criteria saved to `stage_3_screening_criteria.json`
4. Screening subgraph runs with user-edited criteria
5. `broker.pause(3, "screening", decisions)` — user reviews per-paper decisions

**Screening prompt — strict exclusion rules:**
In addition to the user-edited criteria, every screening prompt enforces 5 strict rules:
1. The intervention must be the PRIMARY focus — incidental mentions → EXCLUDE
2. Outcomes must include observed endpoint data — surrogate-only studies → EXCLUDE
3. Protocol/design papers with no reported results → EXCLUDE
4. Narrative reviews and non-systematic guideline summaries → EXCLUDE
5. "uncertain" only when the abstract genuinely lacks information; default to EXCLUDE

**Batch size and retry:** Controlled by `RunConfig["screening_batch_size"]` (default 3). After each batch LLM call, any PMID that did not receive a decision is retried individually. This guarantees full coverage even when the LLM truncates its output.

**Criterion scoring:** For each paper the LLM returns a `criterion_scores` list — one entry per criterion — with `met` (`yes`/`no`/`unclear`) and a `note` quoting the relevant abstract span. The overall `decision` is then **derived algorithmically** from the scores (not from the LLM's free-form verdict):
- Any exclusion criterion `met=yes` → `exclude`
- Any inclusion criterion `met=no` → `exclude`
- Study designs specified and none matched → `exclude`
- Any `unclear` score (when not already excluded) → `uncertain`
- All criteria met → `include`

The LLM's `decision` field is used only as a fallback when no criterion scores are returned.

**Gate B — Screening HITL panel (UI mode):** Filter radio (All / Include / Uncertain / Exclude), non-interactive dataframe of decisions, expandable detail panel showing full title, abstract (up to 2,000 characters), AI reason, and criterion scorecard. The scorecard uses `[IN]`/`[EX]`/`[SD]` type labels and `✓`/`✗`/`?` met indicators with per-criterion notes. Per-paper ✓ Include / ? Uncertain / ✗ Exclude buttons override decisions before final approval.

### Stage 7 — Revision loop

Stage 7 runs in a loop while the user keeps requesting revisions:

1. Manuscript subgraph generates draft + rubric scores
2. `broker.pause(7, "manuscript", {draft, rubric, draft_version})` — user reviews
3. User has three editing modes (all combinable before approving):
   - **Direct edit** — the draft code block is fully editable; user changes markdown inline
   - **Section rewrite** — user names a section (e.g. `Methods`) and types a freeform instruction; the LLM rewrites just that section body in-place (no broker round-trip, updates the draft display immediately)
   - **Full revise** — `action == "revise"`: re-runs the manuscript subgraph → new versioned draft → loops back to step 2
4. `action == "approve"`: exit loop. If the draft was directly edited or section-rewritten, the `edited_draft` is written to disk and re-exported to `.docx` before finalising.

---

## Grounding Layer

Every LLM-extracted value is verified against its source text before being stored. The goal is to catch hallucinations — values the LLM invented rather than derived from the paper.

### Decision flow

```
extracted_value
    │
    ├── len < 20 chars? ──▶ exact substring search
    │                           ├── found → grounded (confidence=100)
    │                           └── not found → quarantined (confidence=0)
    │
    └── len ≥ 20 chars? ──▶ token_set_ratio vs full source text
                                ├── score ≥ threshold → locate span in sentence chunks → grounded
                                └── score < threshold → quarantined
```

### Metric: `token_set_ratio`

`token_set_ratio` (rapidfuzz) sorts both strings into token sets before comparing. This handles word-order differences and paraphrasing — common when extracting from abstracts, which summarise full-text findings in different phrasing. `partial_ratio` (character-order sensitive) was used previously and incorrectly quarantined correct paraphrased extractions.

### Source-adaptive threshold

Abstracts are paraphrased summaries; full text is verbatim. A single threshold produces too many false quarantines on abstract-sourced papers:

| Source | Threshold |
|---|---|
| `abstract` | 75 |
| `fulltext` | 85 |

The threshold can be overridden by passing `threshold=N` to `ExtractionGrounder()`.

### Short value handling

Fuzzy matching on strings shorter than 20 characters (e.g. sample sizes like `"96"`, durations like `"4 weeks"`) is unreliable — the score is dominated by token overlap noise. These are handled with exact case-insensitive substring search instead.

### Span location

When a value passes the threshold check, its location in the source text is recorded as a `Span(pmid, source, char_start, char_end, text, provenance_type)` for provenance. Span location uses **sentence-level chunking** (windows of 3 sentences scored with `token_set_ratio`) rather than a character-by-character sliding window, which was O(n) fuzzy calls for a 6000-character source text.

The `provenance_type` field records *how* each extracted value was matched to its source:

| Value | Meaning |
|---|---|
| `"direct"` | Verbatim substring match — the extracted value appears exactly in the source text (case-insensitive). Used for short values (< 20 chars). |
| `"paraphrased"` | Fuzzy token match — the value was matched via `token_set_ratio` against a sentence window. Used for longer values that may differ in word order or wording. |
| `"inferred"` | LLM-confirmed — the span was located by a second-pass LLM grounding call (set by auto-grounding in Stage 5 extraction, not by the fuzzy grounder). |

### Quarantine behaviour

- Quarantined fields are stored, not dropped — the paper is not discarded
- `extracted_data` contains only grounded fields; quarantined fields are in `quarantined_fields`
- They appear at the Stage 5 HITL gate for manual resolution
- PRISMA flow diagram includes quarantine counts as a data quality signal
- Full quarantine table queryable via SQLite or `slr status <run_id>`

### Auto LLM grounding (Stage 5, automatic)

Fuzzy matching fails on valid extractions that are heavily paraphrased, abbreviated, or expressed differently (e.g. `"96"` vs `"ninety-six"`, `"MI"` vs `"myocardial infarction"`). After fuzzy grounding runs inside `extraction.py`, any quarantined fields automatically get a second-chance LLM grounding pass before they are written to the quarantine table (`_auto_llm_ground`):

1. For each quarantined field, the LLM is asked: *"Does the source text support this value, even if phrased differently?"*
2. Fields confirmed (`supported: true`) are promoted into `extracted_data` with `confidence=80.0` and `span=None` (LLM does not return character offsets; `provenance_type="inferred"` will be set once span construction is added)
3. Fields not confirmed are written to the quarantine table as before

This is intentionally separate from re-extraction — the extracted value itself is not changed, only its quarantine status. The manual "LLM Ground" button at Gate 5 has been removed; auto-grounding replaces it for all papers.

### Synthesis grounding (Stage 6)

A separate LLM-based grounding step: Gemma is asked to cite the PMIDs that support each synthesised claim. Claims with zero citations are quarantined. This is semantic rather than lexical — appropriate for synthesised statements that combine evidence across papers.

### Unresolved questions (Stage 6)

In addition to `claims` and `narrative`, the synthesis LLM now returns an `unresolved_questions` list. Each entry has the shape:

```json
{"question": "...", "relevant_pmids": ["..."], "importance": "high"}
```

where `importance` is one of `"high"`, `"medium"`, or `"low"`. These are open questions the available evidence does not resolve.

The questions are stored in state as `synthesis_questions` and emitted at Gate 6 (`stage_6_synthesis.json`) under the key `unresolved_questions`. The reviewer can inspect these gaps before approving the manuscript stage. The `.md` synthesis file written to disk is not changed by this feature.

---

## Search Design

### PubMed (Entrez)

- One Entrez `esearch` call per query string in `pico.query_strings`
- `mindate` / `maxdate` / `datetype=pdat` for date range filtering
- `efetch` in batches of 200 to retrieve titles + abstracts
- Rate limiting: 3 req/s without API key, 10 req/s with key

#### Result cap and relevance ordering

PubMed returns results in **relevance order** by default (best matches first within each query). The per-query cap uses **2× headroom** to compensate for overlap between PICO-derived query strings (which frequently retrieve the same papers):

```
per_query_cap = max_results // len(query_strings) * 2
```

After merging (deduplicating by PMID), a final slice enforces the hard cap:

```
all_pmids = list(ordered_union_of_all_queries)[:max_results]
```

The cap is applied to the **total** retrieved, not per-query. Without the 2× factor, heavily overlapping queries (e.g. three variations of "aspirin cardiovascular") would yield far fewer unique PMIDs than `max_results` after deduplication.

**`n_pubmed` vs `n_retrieved`:** `n_pubmed` in `SearchCounts` counts papers actually stored (from `efetch` `PubmedArticle` records). This is lower than the esearch PMID count because some PMIDs correspond to letters, errata, and editorial notes that have no `Abstract` element in the XML. `n_retrieved = n_pubmed + n_biorxiv`.

### bioRxiv

- Date-range API: `https://api.biorxiv.org/details/biorxiv/{date_from}/{date_to}/0/json`
- No keyword search API — fetches recent preprints by date range, capped at `max_results`
- PICO relevance filtering happens in Stage 3 screening (LLM-based), not at retrieval time
- Failures (network, timeout, non-200 HTTP) emit a `RuntimeWarning` and the pipeline continues with PubMed results only

### arXiv (opt-in)

- Atom API: `https://export.arxiv.org/api/query?search_query=...`
- Keyword search only — PubMed field tags (`[MeSH]`, `[tiab]`, `[Title/Abstract]`) are stripped from query strings before submission
- PMIDs are stored as `arxiv:{arxiv_id}` proxies so they flow through the same paper store as PubMed records
- Rate limit: 3 req/s enforced via `time.sleep(3)` between requests
- Failures are non-fatal — the pipeline continues with PubMed (+ bioRxiv) results
- Enable by adding `"arxiv"` to `search_sources` at the Stage 1 gate

#### Combined volume

With all three sources enabled, the total paper count entering screening can be up to `3 × max_results`. Set `search_sources: ["pubmed"]` to stay within `max_results` total.

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

**Four-pass generation:**
1. **Writer pass (Pass 1)** — the LLM writes each section guided by its `instructions` as pure prose. Section prompts explicitly forbid inline citations: no PMID numbers, no author-year references, no brackets. This eliminates LLM citation hallucination at source.
2. **Citation verifier pass (Pass 2)** — `_verify_citations_node` reads grounded claims from the synthesis file (lines matching `- claim text [PMID1, PMID2]`), asks the LLM to identify which section of the draft each claim appears in, then injects `(PMID: X, Y)` markers. All PMID anchors come from the synthesis grounding pass, not from the LLM's parametric memory. If no synthesis file exists or contains no grounded claims, the draft is returned unchanged.
3. **Adversarial reviewer pass (Pass 3)** — `_adversarial_review_node` runs with `think=True` for careful reasoning and returns a structured list of issues with three severity levels:
   - `FATAL`: the manuscript is wrong or misleading in a way requiring a prior pipeline stage to be rerun. The `rerun_stage` field names the stage (`screening` | `extraction` | `synthesis`). The orchestrator handles this automatically (bounded to one retry) before Gate 7.
   - `MAJOR`: a significant flaw the human must address before submission.
   - `MINOR`: a style, clarity, or minor accuracy issue.
   All issues are stored in `adversarial_review` in state and surfaced in Gate 7 checkpoint data so the human reviewer sees them. On LLM failure, the node degrades gracefully and returns `{"issues": []}`.
4. **Rubric pass (Pass 4)** — the LLM scores each criterion (`met` / `partial` / `not met` + explanation)

The Stage 7 HITL gate shows the scored rubric, adversarial review issues, and the draft. Triggering a revision re-runs Pass 1 targeting only `partial` / `not met` sections.

### Hallucination prevention

Each section prompt includes a `search_context` block containing only factual pipeline data, with explicit instructions not to invent any details:

- Exact databases searched, date range, and query strings (from run state)
- Exact PRISMA flow counts (retrieved, screened, excluded, included)
- Accurate pipeline description: AI-assisted, GRADE (not Cochrane RoB 2.0 or NOS), narrative synthesis (no meta-analysis, no forest plots), no named human reviewers, no reference management software, no supplementary tables

This prevents the most common manuscript hallucinations: invented database names, fake reviewer names, wrong risk-of-bias tools, and bracketed placeholders like `[Table 1]`.

### Pre-generated tables

Study characteristics and GRADE evidence tables are generated directly from the SQLite paper records (no LLM), then appended verbatim to the matching manuscript sections:

- `_build_study_table(papers)` → appended to any section whose name contains "study characteristics" or "characteristics of included studies"
- `_build_grade_table(papers)` → appended to any section whose name contains "risk of bias", "quality assessment", or "grade assessment"

This ensures tables contain exact extracted values rather than LLM paraphrases.

---

## ProgressEmitter

`ProgressEmitter` is the event sink that decouples the pipeline from its consumers. Every stage calls `emitter.emit(stage, data)` on completion and `emitter.log(message)` for progress updates.

Each call fans out to:

1. **Disk** — `outputs/<run_id>/stage_N_<name>.json` (always)
2. **CLI** — `click.echo` (if `echo` provided)
3. **Gradio** — put to `Queue` polled by `gr.Timer(N).tick()` (if Gradio is running; Gradio 6 removed `app.load(every=N)`)

The `name` parameter overrides the default stage name for custom filenames (e.g., `stage_3_screening_criteria.json` vs `stage_3_screening.json`).

---

## SQLite Checkpointer

LangGraph's `SqliteSaver` is used as the graph checkpointer. After every node completion, full graph state is persisted to `slr_runs.db`. This enables `slr resume <run_id>` to re-enter at the exact node where the run paused or failed.

**Important:** `SqliteSaver` must be constructed as `SqliteSaver(conn)` with an already-open `sqlite3.connect()` connection. `SqliteSaver.from_conn_string()` is a context manager (yields, doesn't return) and cannot be used directly outside a `with` block.

---

## Gradio UI Panel Design

`build_app_with_handler(ui_handler, run_id, llm)` builds the HITL review UI. A single `gr.Timer(1).tick()` polls `UIHandler.get_pending()` and routes to one of four stage-specific panel groups based on `cp["stage"]`:

| Stage(s) | Panel | Components |
|---|---|---|
| 1, 3, 6 | Generic | Editable `gr.Code(language="json")` block + Approve button |
| 2 | Search | `gr.Dataframe` with Exclude bool column; PMID add textbox; Approve button |
| 5 | Extraction | `gr.Dataframe` with Exclude bool column; row-click shows extracted and quarantined fields in two `gr.Code` panels (LLM grounding now runs automatically before Gate 5; the manual LLM Ground button has been removed) |
| 7 | Manuscript | Editable draft (`gr.Code`, interactive); section-rewrite accordion (section name + instruction → LLM rewrites section in-place); rubric scores; Approve / Full Revise buttons |

**Implementation notes:**
- `gr.Group` is used for visibility toggling (not `gr.Column`) — Gradio 6 requires `gr.Group` for `visible` updates to propagate correctly; panels use `elem_classes="checkpoint-panel"` with CSS `height: auto` to prevent flex-stretch layout issues
- `gr.Timer(N).tick()` replaces the removed `app.load(every=N)` from Gradio 5
- `gr.Dataframe` uses `column_count=(N, "fixed")` (the `col_count` parameter was deprecated in Gradio 6); `max_height=250` keeps tables scrollable without pushing buttons off-screen
- `poll_checkpoint` returns 15 values matching its `outputs=` list (3 added for the Stage 2 search panel)
- `papers_state = gr.State([])` holds full paper dicts between the timer tick and the row-select handler, avoiding redundant DB reads
- Section rewrite runs entirely inside the Gradio handler (calls `_llm.chat()` directly) — no broker round-trip required; the updated draft is passed back via `edited_draft` only on Approve

---

## Trajectory Logging (TraceWriter)

Every run writes two append-only JSONL trace files alongside the stage output files:

```
outputs/<run_id>/
  llm_trace.jsonl    — one entry per Ollama call
  hitl_trace.jsonl   — one entry per HITL gate interaction
```

### `llm_trace.jsonl`

Each line is a JSON object recording the full context of one `LLMClient.chat()` call:

```json
{
  "ts": 1712345678.123,
  "model": "gemma4:e4b",
  "think": true,
  "attempt": 1,
  "n_messages": 1,
  "messages": [{"role": "user", "content": "You are screening abstracts..."}],
  "schema_keys": ["decision", "pmid", "reason"],
  "thinking": "The abstract describes a randomised trial. The population...",
  "response": "{\"decisions\": [...]}",
  "latency_s": 4.217,
  "prompt_tokens": 312,
  "completion_tokens": 89,
  "error": null
}
```

Fields:
- `messages` — the full prompt including PICO context and batch text
- `thinking` — Gemma 4's reasoning chain (only present when `think=True`; `null` otherwise)
- `schema_keys` — sorted list of required output fields (avoids duplicating the full schema in every entry)
- `error` — non-null only on failed attempts (e.g. `"JSONDecodeError: ..."`); a successful retry after a failed attempt produces two entries for the same logical call
- `prompt_tokens` / `completion_tokens` — from Ollama's `prompt_eval_count` / `eval_count` response fields; `null` if not reported

### `hitl_trace.jsonl`

Each line records one broker gate interaction (called even in `NoOpHandler` mode so automated runs are traceable):

```json
{
  "ts": 1712345890.456,
  "stage": 3,
  "stage_name": "screening",
  "action": "approve",
  "n_changes": 2,
  "diff": {
    "papers": {
      "before": [{"pmid": "99999", "decision": "uncertain", ...}],
      "after":  [{"pmid": "99999", "decision": "include",   ...}]
    }
  },
  "before": { ... },
  "after":  { ... }
}
```

Fields:
- `diff` — only keys whose values changed between what the pipeline produced and what the user returned; the `action` key is excluded from the diff
- `n_changes` — number of changed keys (0 = pure approve with no edits)
- `before` / `after` — full snapshots for complete auditability

### Wiring

`TraceWriter` is instantiated once per run in `_build_orchestrator` (in `cli.py`) using `emitter.run_dir` as the target directory. The same instance is injected into both `LLMClient` and `CheckpointBroker`:

```python
trace_writer = TraceWriter(emitter.run_dir)
llm = LLMClient(model=..., trace_writer=trace_writer)
broker._trace = trace_writer
```

`TraceWriter` does not hold open file handles — each `write_*` call opens, appends one line, and closes. This is safe under the GIL for single-process runs.

### Use cases

| Task | Which file | How |
|---|---|---|
| Replay a screening decision | `llm_trace.jsonl` | Find entries where `schema_keys` contains `"decision"` |
| Inspect Gemma 4 reasoning chain | `llm_trace.jsonl` | Filter `"think": true`, read `"thinking"` field |
| See what a user changed at a gate | `hitl_trace.jsonl` | Filter by `"stage"`, read `"diff"` |
| Measure LLM latency per stage | `llm_trace.jsonl` | Aggregate `"latency_s"` by prompt content |
| Identify JSON retry failures | `llm_trace.jsonl` | Filter `"error": {"$ne": null}` |
| Token usage per run | `llm_trace.jsonl` | Sum `"prompt_tokens"` + `"completion_tokens"` |

---

## LLM Call Caching

`LLMCache` (`slr_agent/cache.py`) is a disk-backed cache for Ollama call results. It is instantiated once per run in `_build_orchestrator` and passed to `LLMClient` via `cache=cache`.

**Key design decisions:**

- **Run-scoped:** Cache files live at `outputs/<run_id>/.llm_cache/<hash>.json`. A new run always starts with an empty cache; stale results from a different model or prompt version cannot leak in.
- **SHA-256 keyed:** The cache key is `sha256(json({model, messages, schema, think}))` with `sort_keys=True` so key order doesn't affect the hash. Images (`images=` in messages) are included in the hash when present.
- **Atomic writes:** `put()` writes to a temp file in the same directory then renames with `os.replace` (POSIX-atomic). Crashed writes leave no partial file.
- **Graceful read failures:** `get()` catches `json.JSONDecodeError`, `KeyError`, and `OSError` and returns `None` (cache miss), so a corrupt file from a previous crash is silently skipped rather than crashing the pipeline.
- **Cache hit path:** If the cache returns a result, `LLMClient.chat()` returns it immediately — no Ollama call, no trace write. This makes re-runs of interrupted pipelines fast.
- **No new dependencies:** Uses only `hashlib`, `json`, `os`, `tempfile` from stdlib.

---

## Citation Network Layer

`slr_agent/citation_network.py` builds a lightweight within-corpus citation graph after full-text fetch (Stage 4). It detects two evidence-inflation patterns common in medical literature that GRADE's five dimensions do not capture.

### How it works

After `fulltext_node` completes, included papers' PubMed XML reference lists are parsed with `_extract_cited_pmids_from_xml`. Each `<ArticleId IdType="pubmed">` element is extracted and intersected with the corpus PMID set. Self-citations are excluded. Two metrics are computed:

| Metric | What it measures |
|---|---|
| `echo_chamber_ratio` | Fraction of corpus papers that cite ≥1 other corpus paper |
| `dominant_count / n` | Whether one paper is cited by >50% of the corpus |

### Warning conditions

A `warning` string is set in `CitationNetworkSummary` when:
- **Dominant paper:** one PMID is cited by >50% of included papers — indicates many papers may derive from the same original source, inflating apparent evidence volume
- **Echo-chamber ratio:** >50% of included papers cross-cite each other — indicates circular citation patterns

### Pipeline integration

- `fulltext_node` calls `build_citation_network(included_papers)` and stores `cn_summary.to_dict()` in `OrchestratorState["citation_network"]`
- When `fetch_fulltext=False`, `synthesis_node` computes the network lazily (all papers have `fulltext=None`, so the summary is empty with no warning)
- The summary is included in the Gate 6 emit (`stage_6_synthesis.json`) under `citation_network` so the reviewer sees it before approving synthesis
- If a warning is present, it is passed to `_adversarial_review_node` as a `CITATION NETWORK ALERT` block in the prompt

---

## Testing Strategy

| Layer | Tool | Ollama needed |
|---|---|---|
| Unit tests | pytest + MockLLM | No |
| Integration tests | pytest + MockLLM | No |
| E2E smoke test | pytest -m e2e | Yes |

`MockLLM` matches prompts by substring (case-sensitive `in` check) and returns pre-registered structured responses. Tests register keys matching the start of each LLM prompt.

Grounding regression test: injects a `PaperRecord` with a deliberately hallucinated extracted field (no matching span) and verifies the grounding node quarantines it rather than passing it through.
