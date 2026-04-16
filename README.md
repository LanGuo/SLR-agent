# SLR Agent

A general-purpose Systematic Literature Review (SLR) agent that takes any research question as input, runs a PRISMA-compliant pipeline, and produces a structured manuscript.

Runs entirely locally using LLM of your choice via **Ollama** (default, **Gemma 4 E4B** 9.6 GB). Orchestrated with **LangGraph** hierarchical subgraphs for explicit state machine control, resumability, and full audit trails.

---

## Quick Start

### 1. Install prerequisites

**Python 3.11+**

**Ollama** — download from [ollama.com](https://ollama.com) or on Mac:
```bash
brew install ollama
```

**Pull the model** — the pipeline uses `gemma4:e4b` by default (9.6 GB, fits in 16 GB unified memory):
```bash
ollama pull gemma4:e4b   # default — Gemma 4 E4B, 9.6 GB, 128K context
ollama pull gemma4:26b   # higher quality — MoE, 18 GB, 256K context (needs 24 GB+ RAM)
ollama pull gemma4:31b   # best quality — dense 31B, 20 GB, 256K context (needs 24 GB+ RAM)
```

**Gemma 4 model selection guide (Mac Mini M4):**

| Model | Size | RAM needed | Quality |
|---|---|---|---|
| `gemma4:e4b` | 9.6 GB | 16 GB | Good — default |
| `gemma4:26b` | 18 GB | 24 GB | Better — MoE, only 3.8B active params at inference |
| `gemma4:31b` | 20 GB | 24 GB+ | Best — dense 31B |

**Start the Ollama server** (must be running before `slr run`):
```bash
ollama serve             # runs in foreground; open a new terminal for slr commands
# On Mac you can also start the Ollama desktop app instead
```

Verify it's up:
```bash
curl http://localhost:11434   # should return "Ollama is running"
```

**Pandoc** (optional, for Word .docx export):
```bash
brew install pandoc
```

### 2. Install and run

```bash
# Install
pip install -e .

# Run a review (CLI HITL mode — review each stage in your terminal)
slr run "Does aspirin reduce blood pressure in adults with hypertension?" --hitl cli

# Run with Gradio UI review panels
slr run "Does aspirin reduce blood pressure?" --hitl ui
# Then open http://localhost:7860

# Fully automated (no checkpoints)
slr run "Does aspirin reduce blood pressure?" --no-checkpoints

# With a custom manuscript template
slr run "..." --template path/to/template.json

# Resume a paused run
slr resume <run_id>

# Check run status and quarantine report
slr status <run_id>

# Export manuscript from a completed run
slr export <run_id>
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--hitl cli\|ui` | `cli` | Review mode: terminal prompts or Gradio browser panels |
| `--no-checkpoints` | off | Skip all gates, run fully automated |
| `--no-fulltext` | off | Skip full-text PDF fetching (faster) |
| `--max-results N` | 500 | Total paper cap across all queries; distributed evenly across query strings |
| `--model TAG` | `gemma4:e4b` | Ollama model tag (e.g. `gemma4:26b`, `gemma4:31b`) |
| `--api-key KEY` | env `PUBMED_API_KEY` | PubMed API key (10 req/s vs 3 req/s without) |
| `--template FILE` | PRISMA default | Manuscript template: JSON schema or PDF reference paper |
| `--screening-batch-size N` | `3` | Abstracts per LLM call during screening; smaller = more reliable JSON |

---

## Pipeline Stages

The pipeline has 7 stages. Each produces structured output saved to `outputs/<run_id>/`. Stages 1, 2, 3, 5, and 7 have human-in-the-loop (HITL) gates by default.

```
① PICO Formulation     — Translates question → PICO + PubMed queries
② Search               — PubMed Entrez + bioRxiv + arXiv (opt-in); date range; dedup
③ Screening            — LLM screens abstracts with explicit criteria
④ Full-text (optional) — Fetches and screens PMC PDFs
⑤ Data Extraction      — Structured extraction per paper + GRADE scoring
⑥ Evidence Synthesis   — Narrative synthesis + PRISMA flow + open questions
⑦ Manuscript           — Writer → citation verifier → adversarial reviewer → rubric revision loop
```

At each HITL gate you can edit, override, or add data before the pipeline continues. At the **Stage 1 gate** you can also edit search configuration (sources, date range, max results) — this is where you enable arXiv by adding `"arxiv"` to the sources list. At the **Stage 2 gate** you can exclude papers by PMID or add papers manually; excluded papers are pinned and skipped by the LLM screener. At the **Stage 3 gate** you can first review AI-generated inclusion/exclusion criteria, then review per-paper screening decisions — each paper shows a **criterion scorecard** (`✓`/`✗`/`?` per criterion with a supporting note from the abstract) plus per-paper Include/Uncertain/Exclude override buttons. At the **Stage 5 gate** you can exclude papers and mark quarantined fields for LLM re-verification. At the **Stage 6 gate** you see the synthesis preview plus any **unresolved questions** the LLM identified (each tagged `high`/`medium`/`low` importance) — these surface evidence gaps before the manuscript is written. At the **Stage 7 gate** you see the draft alongside its rubric score and the **adversarial reviewer's** FATAL/MAJOR/MINOR findings; FATAL issues that name a prior stage trigger an automatic one-shot rerun before the gate opens. You can also edit the draft directly, rewrite any section with a custom LLM prompt, or trigger a full LLM revision pass.

---

## Outputs

All outputs land under `outputs/<run_id>/`:

```
outputs/<run_id>/
  stage_1_pico.json              # PICO fields, query strings, detected language
  stage_2_search.json            # papers per query, dedup stats, total retrieved
  stage_3_screening_criteria.json # inclusion/exclusion criteria used for screening
  stage_3_screening.json         # per-paper decisions + criterion scores
  stage_4_fulltext.json          # full-text fetch stats
  stage_5_extraction.json        # per-paper extracted fields + GRADE + quarantined
  stage_6_synthesis.json         # claims, supporting PMIDs, narrative, unresolved_questions
  stage_7_rubric.json            # rubric scores (met/partial/not met)
  stage_7_draft_v1.md            # first manuscript draft
  stage_7_draft_v2.md            # revision 2 (if triggered), etc.
  <run_id>_manuscript.md         # final approved manuscript
  <run_id>_manuscript.docx       # Word export (requires Pandoc)
  <run_id>_prisma.md             # PRISMA 2020 flow diagram (Mermaid)
  llm_trace.jsonl                # every Ollama call: prompt, thinking, response, latency, tokens
  hitl_trace.jsonl               # every HITL gate: before/after diff, user action
```

---

## Technology Stack

| Component | Technology |
|---|---|
| LLM runtime | Ollama — Gemma 4 E4B (default, 9.6 GB); `gemma4:26b` MoE (18 GB) or `gemma4:31b` (20 GB) for best results |
| Orchestration | LangGraph hierarchical subgraphs + SQLite checkpointer |
| Search | PubMed Entrez (Biopython) + bioRxiv REST API + arXiv Atom API (httpx) |
| PDF parsing | PyMuPDF |
| Fuzzy matching / grounding | rapidfuzz |
| UI | Gradio |
| Word export | Pandoc (Markdown → .docx) |
| State & paper store | SQLite (`slr_runs.db`) |
| Language | Python 3.11+ |
| PRISMA diagram | Mermaid |

---

## Grounding & Audit Trail

Every LLM-extracted value is **fuzzy-matched against its source text** (abstract or full-text) using `token_set_ratio` (rapidfuzz). The threshold is source-adaptive: 75 for abstracts (paraphrased summaries), 85 for full text (verbatim). Short values under 20 characters use exact substring search instead.

On match, the exact character span is stored as provenance with a `provenance_type`:

| Type | Meaning |
|---|---|
| `direct` | Exact substring — value appears verbatim in source |
| `paraphrased` | Fuzzy match — value is a paraphrase of the source |
| `inferred` | LLM-confirmed — failed fuzzy matching but a second-pass LLM call confirmed the source supports the value |

Fields that fail fuzzy matching go through **automatic LLM grounding** before being quarantined: the model is asked whether the source text supports the value even if phrased differently. Confirmed fields are promoted with `confidence=80` and `provenance_type="inferred"`. Only fields the LLM also cannot confirm are quarantined.

- `slr status <run_id>` shows quarantined field counts
- Quarantined items appear in HITL gates for manual resolution (accept / edit / discard)
- Full quarantine table in SQLite for audit

Synthesis claims are grounded separately: the LLM must cite the PMIDs that support each claim. Claims with zero citations are quarantined.

---

## Multi-language Support

Input can be in any language the model supports. The pipeline:
1. Detects the source language
2. Searches PubMed in English
3. Outputs the manuscript in the detected source language

---

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full architecture, state design, HITL mechanism, grounding design, and component boundaries.

---

## Testing

```bash
# Unit + integration tests (no Ollama required)
pytest tests/ --ignore=tests/e2e

# End-to-end smoke test (requires Ollama + internet)
pytest tests/e2e -m e2e
```

Tests use `MockLLM` — a canned-response stub that matches prompts by substring. No Ollama needed for CI.

---

## Project Structure

```
slr_agent/
  orchestrator.py     # LangGraph graph wiring; HITL gate logic
  state.py            # OrchestratorState, PICOResult, SearchCounts, etc.
  config.py           # RunConfig TypedDict + DEFAULT_CONFIG
  broker.py           # CheckpointBroker, CLIHandler, UIHandler, NoOpHandler
  emitter.py          # ProgressEmitter (disk + CLI + Gradio fan-out)
  trace.py            # TraceWriter — llm_trace.jsonl + hitl_trace.jsonl per run
  db.py               # SQLite paper store (PaperRecord, GRADEScore)
  grounding.py        # ExtractionGrounder (rapidfuzz span matching)
  template.py         # Manuscript template loading (JSON / PDF / default PRISMA)
  llm.py              # LLMClient (Ollama) + MockLLM
  cli.py              # Click CLI (slr run/resume/status/export)
  prisma.py           # PRISMA flow diagram generation
  export.py           # Pandoc .docx export
  subgraphs/
    pico.py           # Stage 1: PICO formulation + query generation
    search.py         # Stage 2: PubMed + bioRxiv + arXiv search
    screening.py      # Stage 3: abstract screening with criteria
    fulltext.py       # Stage 4: PMC PDF fetch + screen
    extraction.py     # Stage 5: structured data extraction + GRADE
    synthesis.py      # Stage 6: narrative synthesis
    manuscript.py     # Stage 7: writer → citation verifier → adversarial reviewer → rubric
  ui/
    app.py            # Gradio app factory (build_app_with_handler)
    panels/           # Per-stage review panels (pico, search, screening,
                      #   extraction, synthesis, manuscript)
```
