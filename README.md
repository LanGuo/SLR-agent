# SLR Agent

A general-purpose Systematic Literature Review (SLR) agent that takes any research question as input, runs a PRISMA-compliant pipeline, and produces a structured manuscript. Built for the Kaggle Gemma 4 Good Hackathon.

Runs entirely locally on a Mac Mini using **Gemma 4 26B MoE via Ollama**. Orchestrated with **LangGraph** hierarchical subgraphs for explicit state machine control, resumability, and full audit trails.

---

## Quick Start

### 1. Install prerequisites

**Python 3.11+**

**Ollama** — download from [ollama.com](https://ollama.com) or on Mac:
```bash
brew install ollama
```

**Pull the model** — the pipeline uses `gemma3:12b` by default (~8GB); swap to `gemma4:27b` (~18GB) for higher quality:
```bash
ollama pull gemma3:12b   # default, fits in 8GB VRAM
# or
ollama pull gemma4:27b   # recommended for best results, needs ~18GB RAM
```

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
| `--max-results N` | 500 | PubMed search cap per query |
| `--api-key KEY` | env `PUBMED_API_KEY` | PubMed API key (10 req/s vs 3 req/s without) |
| `--template FILE` | PRISMA default | Manuscript template: JSON schema or PDF reference paper |

---

## Pipeline Stages

The pipeline has 7 stages. Each produces structured output saved to `outputs/<run_id>/`. Stages 1, 3, 5, 6, and 7 have human-in-the-loop (HITL) gates by default.

```
① PICO Formulation     — Translates question → PICO + PubMed queries
② Search               — PubMed Entrez + bioRxiv; date range; dedup
③ Screening            — LLM screens abstracts with explicit criteria
④ Full-text (optional) — Fetches and screens PMC PDFs
⑤ Data Extraction      — Structured extraction per paper + GRADE scoring
⑥ Evidence Synthesis   — Narrative synthesis + PRISMA flow
⑦ Manuscript           — Sectioned draft + rubric-guided revision loop
```

At each HITL gate you can edit, override, or add data before the pipeline continues. At the **Stage 1 gate** you can also edit search configuration (sources, date range, max results). At the **Stage 3 gate** you can first review AI-generated inclusion/exclusion criteria, then review per-paper screening decisions.

---

## Outputs

All outputs land under `outputs/<run_id>/`:

```
outputs/<run_id>/
  stage_1_pico.json              # PICO fields, query strings, detected language
  stage_2_search.json            # papers per query, dedup stats, total retrieved
  stage_3_screening_criteria.json # inclusion/exclusion criteria used for screening
  stage_3_screening.json         # per-paper decisions
  stage_4_fulltext.json          # full-text fetch stats
  stage_5_extraction.json        # per-paper extracted fields + GRADE + quarantined
  stage_6_synthesis.json         # claims, supporting PMIDs, narrative
  stage_7_rubric.json            # rubric scores (met/partial/not met)
  stage_7_draft_v1.md            # first manuscript draft
  stage_7_draft_v2.md            # revision 2 (if triggered), etc.
  <run_id>_manuscript.md         # final approved manuscript
  <run_id>_manuscript.docx       # Word export (requires Pandoc)
  <run_id>_prisma.md             # PRISMA 2020 flow diagram (Mermaid)
```

---

## Technology Stack

| Component | Technology |
|---|---|
| LLM runtime | Ollama — Gemma 4 26B MoE (4-bit, ~18GB RAM) |
| Orchestration | LangGraph hierarchical subgraphs + SQLite checkpointer |
| Search | PubMed Entrez (Biopython) + bioRxiv REST API (httpx) |
| PDF parsing | PyMuPDF |
| Fuzzy matching / grounding | rapidfuzz |
| UI | Gradio |
| Word export | Pandoc (Markdown → .docx) |
| State & paper store | SQLite (`slr_runs.db`) |
| Language | Python 3.11+ |
| PRISMA diagram | Mermaid |

Gemma 4 26B MoE fits in ~18GB RAM at 4-bit precision, running comfortably on a Mac Mini with 24GB unified memory.

---

## Grounding & Audit Trail

Every LLM-extracted value is **fuzzy-matched against its source text** (abstract or full-text, via rapidfuzz token_sort_ratio ≥ 85). On match, the exact character span is stored as provenance. On failure, the field is **quarantined** — kept but flagged, not dropped.

- `slr status <run_id>` shows quarantined field counts
- Quarantined items appear in HITL gates for manual resolution (accept / edit / discard)
- Full quarantine table in SQLite for audit

Synthesis claims are grounded separately: Gemma 4 must cite the PMIDs that support each claim. Claims with zero citations are quarantined.

---

## Multi-language Support

Input can be in any language Gemma 4 supports (35+). The pipeline:
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
  db.py               # SQLite paper store (PaperRecord, GRADEScore)
  grounding.py        # ExtractionGrounder (rapidfuzz span matching)
  template.py         # Manuscript template loading (JSON / PDF / default PRISMA)
  llm.py              # LLMClient (Ollama) + MockLLM
  cli.py              # Click CLI (slr run/resume/status/export)
  prisma.py           # PRISMA flow diagram generation
  export.py           # Pandoc .docx export
  subgraphs/
    pico.py           # Stage 1: PICO formulation + query generation
    search.py         # Stage 2: PubMed + bioRxiv search
    screening.py      # Stage 3: abstract screening with criteria
    fulltext.py       # Stage 4: PMC PDF fetch + screen
    extraction.py     # Stage 5: structured data extraction + GRADE
    synthesis.py      # Stage 6: narrative synthesis
    manuscript.py     # Stage 7: two-pass draft + rubric scoring
  ui/
    app.py            # Gradio app factory (build_app_with_handler)
    panels/           # Per-stage review panels (pico, search, screening,
                      #   extraction, synthesis, manuscript)
```
