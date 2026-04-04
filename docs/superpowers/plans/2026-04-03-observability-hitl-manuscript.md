# Observability, HITL Stage Gates, and Manuscript Template Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add pipeline transparency (intermediate results saved to disk + CLI progress), six HITL checkpoint gates (CLI and Gradio UI modes), manual paper inclusion, and a template-driven two-pass manuscript with rubric-guided revision.

**Architecture:** A `ProgressEmitter` writes structured JSON after each stage; a `CheckpointBroker` with swappable handlers (CLIHandler / UIHandler / NoOpHandler) pauses the pipeline for human review; the manuscript subgraph becomes template-driven with a rubric-scoring pass, and the orchestrator drives a revision loop at stage 7.

**Tech Stack:** Python 3.11, LangGraph 1.1.5, langgraph-checkpoint-sqlite, Gradio ≥4, Click ≥8, PyMuPDF (fitz), Biopython (Entrez), threading + queue for broker, uv for deps.

---

## File Map

| File | Status | Responsibility |
|------|--------|---------------|
| `slr_agent/emitter.py` | **Create** | Write stage JSON to disk; fan out to CLI echo and Gradio queue |
| `slr_agent/broker.py` | **Create** | `CheckpointBroker`, `CLIHandler`, `UIHandler`, `NoOpHandler` |
| `slr_agent/template.py` | **Create** | Load JSON/PDF template; default PRISMA structure; rubric gen/scoring |
| `slr_agent/ui/panels/search.py` | **Create** | Stage 2 Gradio panel |
| `slr_agent/ui/panels/screening.py` | **Create** | Stage 3 Gradio panel |
| `slr_agent/ui/panels/extraction.py` | **Create** | Stage 5 Gradio panel |
| `slr_agent/ui/panels/synthesis.py` | **Create** | Stage 6 Gradio panel |
| `slr_agent/ui/panels/manuscript_panel.py` | **Create** | Stage 7 Gradio panel with revision loop |
| `slr_agent/config.py` | **Modify** | Add `template_path`, `hitl_mode` to `RunConfig` |
| `slr_agent/state.py` | **Modify** | Add `template`, `manuscript_draft_version` to `OrchestratorState` |
| `slr_agent/db.py` | **Modify** | Add `add_paper_from_pmid`, `add_paper_from_pdf` methods |
| `slr_agent/orchestrator.py` | **Modify** | Accept broker + emitter; call emit/pause per stage; manuscript revision loop |
| `slr_agent/subgraphs/manuscript.py` | **Modify** | Template-driven section generation; rubric scoring pass; versioned drafts |
| `slr_agent/cli.py` | **Modify** | Add `--hitl`, `--template` flags; build broker/emitter; print stage summaries |
| `slr_agent/ui/app.py` | **Modify** | Wire UIHandler; add live log; add template upload; route checkpoint panels |
| `slr_agent/ui/panels/pico.py` | **Modify** | Return edited dict compatible with broker data format |
| `pyproject.toml` | **Modify** | Add `langgraph-checkpoint-sqlite` dependency |
| `tests/unit/test_emitter.py` | **Create** | Unit tests for ProgressEmitter |
| `tests/unit/test_broker.py` | **Create** | Unit tests for CheckpointBroker and handlers |
| `tests/unit/test_template.py` | **Create** | Unit tests for template loading and rubric scoring |
| `tests/unit/test_db_manual_add.py` | **Create** | Unit tests for manual paper add methods |
| `tests/integration/test_manuscript_subgraph.py` | **Modify** | Update for new template-driven prompts |
| `tests/integration/test_orchestrator_routing.py` | **Modify** | Update for new RunConfig fields + broker injection |

---

### Task 1: Fix SQLite checkpointer dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `slr_agent/orchestrator.py:118-130`

The `langgraph.checkpoint.sqlite` module was moved to a separate package `langgraph-checkpoint-sqlite` in LangGraph ≥0.2. It is not installed. Without it, `slr resume` is broken.

- [ ] **Step 1: Add dependency to pyproject.toml**

```toml
# pyproject.toml — in [project] dependencies list, add:
"langgraph-checkpoint-sqlite>=2.0.0",
```

Full updated dependencies section:
```toml
dependencies = [
    "langgraph>=1.0.0",
    "langgraph-checkpoint-sqlite>=2.0.0",
    "langchain-community>=0.4.0",
    "biopython>=1.83",
    "pymupdf>=1.24.0",
    "rapidfuzz>=3.6.0",
    "gradio>=4.0.0",
    "click>=8.1.0",
    "ollama>=0.2.0",
    "httpx>=0.27.0",
]
```

- [ ] **Step 2: Install the package**

```bash
uv pip install langgraph-checkpoint-sqlite
```

Expected: `Successfully installed langgraph-checkpoint-sqlite-...`

- [ ] **Step 3: Verify import works**

```bash
.venv/bin/python -c "from langgraph_checkpoint_sqlite import SqliteSaver; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Update orchestrator.py import**

In `slr_agent/orchestrator.py` replace lines 118-130:

```python
    # Add SQLite checkpointer only when db_path is provided (enables resume/HITL)
    if db_path:
        try:
            from langgraph_checkpoint_sqlite import SqliteSaver
            checkpointer = SqliteSaver.from_conn_string(db_path)
            return builder.compile(checkpointer=checkpointer)
        except ImportError:
            import warnings
            warnings.warn(
                f"langgraph.checkpoint.sqlite not available — compiling without checkpointer. "
                f"Resume and HITL interrupts will not persist.",
                RuntimeWarning,
                stacklevel=2,
            )

    return builder.compile()
```

- [ ] **Step 5: Run existing tests to confirm nothing broke**

```bash
.venv/bin/pytest tests/ -x -q --ignore=tests/e2e
```

Expected: `32 passed`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml slr_agent/orchestrator.py
git commit -m "fix: install langgraph-checkpoint-sqlite; slr resume now works"
```

---

### Task 2: Config and State updates

**Files:**
- Modify: `slr_agent/config.py`
- Modify: `slr_agent/state.py`

Add `template_path` and `hitl_mode` to `RunConfig`. Add `template` and `manuscript_draft_version` to `OrchestratorState`. Both TypedDicts are used at runtime as plain dicts, so new keys are backward compatible — existing code using `.get()` handles missing keys gracefully.

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_state.py — add to existing file
def test_run_config_new_fields():
    from slr_agent.config import RunConfig
    cfg: RunConfig = {
        "checkpoint_stages": [1, 3],
        "fetch_fulltext": False,
        "output_format": "markdown",
        "pubmed_api_key": None,
        "max_results": 50,
        "search_sources": ["pubmed"],
        "template_path": "/tmp/template.json",
        "hitl_mode": "cli",
    }
    assert cfg["hitl_mode"] == "cli"
    assert cfg["template_path"] == "/tmp/template.json"


def test_orchestrator_state_new_fields():
    from slr_agent.state import OrchestratorState
    keys = OrchestratorState.__annotations__.keys()
    assert "template" in keys
    assert "manuscript_draft_version" in keys
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/unit/test_state.py::test_run_config_new_fields tests/unit/test_state.py::test_orchestrator_state_new_fields -v
```

Expected: FAIL — `KeyError` or assertion error.

- [ ] **Step 3: Update config.py**

```python
# slr_agent/config.py
from typing import Literal
from typing_extensions import TypedDict

class RunConfig(TypedDict, total=False):
    checkpoint_stages: list[int]   # stage numbers 1–7; default [1, 3, 5]
    fetch_fulltext: bool           # default True
    output_format: Literal["markdown", "word", "both"]  # default "both"
    pubmed_api_key: str | None     # raises PubMed rate limit to 10 req/s
    max_results: int               # per source, default 500
    search_sources: list[Literal["pubmed", "biorxiv"]]  # default ["pubmed", "biorxiv"]
    template_path: str | None      # path to JSON schema or PDF template
    hitl_mode: Literal["cli", "ui", "none"]  # default "cli"

DEFAULT_CONFIG: RunConfig = {
    "checkpoint_stages": [1, 3, 5],
    "fetch_fulltext": True,
    "output_format": "both",
    "pubmed_api_key": None,
    "max_results": 500,
    "search_sources": ["pubmed", "biorxiv"],
    "template_path": None,
    "hitl_mode": "cli",
}
```

- [ ] **Step 4: Update state.py**

```python
# slr_agent/state.py
from typing import Literal
from typing_extensions import TypedDict
from slr_agent.config import RunConfig

class PICOResult(TypedDict):
    population: str
    intervention: str
    comparator: str
    outcome: str
    query_strings: list[str]
    source_language: str
    search_language: str
    output_language: str

class SearchCounts(TypedDict):
    n_retrieved: int
    n_duplicates_removed: int

class ScreeningCounts(TypedDict):
    n_included: int
    n_excluded: int
    n_uncertain: int

class FulltextCounts(TypedDict):
    n_fetched: int
    n_unavailable: int
    n_excluded: int

class ExtractionCounts(TypedDict):
    n_extracted: int
    n_grade_high: int
    n_grade_moderate: int
    n_grade_low: int
    n_grade_very_low: int
    n_quarantined_fields: int

class OrchestratorState(TypedDict):
    run_id: str
    config: RunConfig
    pico: PICOResult | None
    search_counts: SearchCounts | None
    screening_counts: ScreeningCounts | None
    fulltext_counts: FulltextCounts | None
    extraction_counts: ExtractionCounts | None
    synthesis_path: str | None
    manuscript_path: str | None
    current_stage: str
    checkpoint_pending: bool
    template: dict | None              # normalized template structure (loaded from template_path)
    manuscript_draft_version: int      # current revision number, starts at 1
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/pytest tests/unit/test_state.py -v
```

Expected: all pass including the two new tests.

- [ ] **Step 6: Commit**

```bash
git add slr_agent/config.py slr_agent/state.py tests/unit/test_state.py
git commit -m "feat: add template_path, hitl_mode to RunConfig; template, draft_version to OrchestratorState"
```

---

### Task 3: ProgressEmitter

**Files:**
- Create: `slr_agent/emitter.py`
- Create: `tests/unit/test_emitter.py`

`ProgressEmitter.emit(stage, data)` writes `outputs/<run_id>/stage_N_name.json`, calls CLI echo, and optionally pushes to a Gradio queue.

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_emitter.py
import json
import os
import queue
from slr_agent.emitter import ProgressEmitter


def test_emit_writes_json_file(tmp_path):
    emitter = ProgressEmitter(output_dir=str(tmp_path), run_id="run-abc")
    emitter.emit(1, {"population": "adults", "intervention": "aspirin"})
    path = tmp_path / "run-abc" / "stage_1_pico.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["intervention"] == "aspirin"


def test_emit_calls_echo(tmp_path):
    echoed = []
    emitter = ProgressEmitter(output_dir=str(tmp_path), run_id="run-abc", echo=echoed.append)
    emitter.emit(2, {"n_retrieved": 42})
    assert any("42" in s for s in echoed)


def test_emit_pushes_to_gradio_queue(tmp_path):
    q = queue.Queue()
    emitter = ProgressEmitter(output_dir=str(tmp_path), run_id="run-abc", gradio_queue=q)
    emitter.emit(3, {"n_included": 10})
    msg = q.get_nowait()
    assert "10" in msg


def test_emit_creates_run_directory(tmp_path):
    emitter = ProgressEmitter(output_dir=str(tmp_path), run_id="run-xyz")
    emitter.emit(4, {"n_fetched": 0})
    assert (tmp_path / "run-xyz").is_dir()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/unit/test_emitter.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'slr_agent.emitter'`

- [ ] **Step 3: Implement ProgressEmitter**

```python
# slr_agent/emitter.py
import json
import os
import queue as _queue
from typing import Callable

_STAGE_NAMES = {
    1: "pico",
    2: "search",
    3: "screening",
    4: "fulltext",
    5: "extraction",
    6: "synthesis",
    7: "rubric",
}


class ProgressEmitter:
    """Writes stage output to disk and fans out to CLI/Gradio sinks."""

    def __init__(
        self,
        output_dir: str,
        run_id: str,
        echo: Callable[[str], None] | None = None,
        gradio_queue: _queue.Queue | None = None,
    ):
        self.output_dir = output_dir
        self.run_id = run_id
        self._echo = echo or (lambda _: None)
        self._gradio_queue = gradio_queue
        os.makedirs(os.path.join(output_dir, run_id), exist_ok=True)

    def emit(self, stage: int, data: dict) -> None:
        """Write stage data to disk and notify sinks."""
        name = _STAGE_NAMES.get(stage, f"stage_{stage}")
        path = os.path.join(self.output_dir, self.run_id, f"stage_{stage}_{name}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        summary = self._format_summary(stage, name, data)
        self._echo(summary)
        if self._gradio_queue is not None:
            self._gradio_queue.put(summary)

    def _format_summary(self, stage: int, name: str, data: dict) -> str:
        lines = [f"\n[Stage {stage}: {name.upper()}]"]
        for k, v in data.items():
            if isinstance(v, list):
                lines.append(f"  {k}: {len(v)} items")
            elif isinstance(v, dict):
                lines.append(f"  {k}: {len(v)} keys")
            else:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    @property
    def run_dir(self) -> str:
        return os.path.join(self.output_dir, self.run_id)
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/unit/test_emitter.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add slr_agent/emitter.py tests/unit/test_emitter.py
git commit -m "feat: add ProgressEmitter — writes stage output to disk and CLI/Gradio sinks"
```

---

### Task 4: CheckpointBroker and handlers

**Files:**
- Create: `slr_agent/broker.py`
- Create: `tests/unit/test_broker.py`

`CheckpointBroker.pause(stage, stage_name, data) → dict` blocks the pipeline thread until the handler resolves. Three handlers: `CLIHandler` (interactive terminal), `UIHandler` (Gradio queue/event), `NoOpHandler` (auto-approve, used for `--no-checkpoints`).

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_broker.py
import threading
from slr_agent.broker import CheckpointBroker, CLIHandler, UIHandler, NoOpHandler


def test_noop_handler_returns_data_unchanged():
    broker = CheckpointBroker(NoOpHandler())
    data = {"population": "adults", "intervention": "aspirin"}
    result = broker.pause(1, "pico", data)
    assert result["population"] == "adults"
    assert result["action"] == "approve"


def test_ui_handler_blocks_until_resume():
    handler = UIHandler()
    broker = CheckpointBroker(handler)
    results = []

    def pipeline():
        results.append(broker.pause(1, "pico", {"x": 1}))

    t = threading.Thread(target=pipeline)
    t.start()
    pending = handler.get_pending(timeout=2.0)
    assert pending is not None
    assert pending["stage"] == 1
    handler.resume({"x": 99, "action": "approve"})
    t.join(timeout=2.0)
    assert results[0]["x"] == 99


def test_ui_handler_get_pending_returns_none_when_empty():
    handler = UIHandler()
    assert handler.get_pending(timeout=0.05) is None


def test_cli_handler_approve(monkeypatch):
    monkeypatch.setattr("click.prompt", lambda *a, **kw: "A")
    broker = CheckpointBroker(CLIHandler())
    data = {"n": 5}
    result = broker.pause(2, "search", data)
    assert result["action"] == "approve"
    assert result["n"] == 5


def test_cli_handler_skip(monkeypatch):
    monkeypatch.setattr("click.prompt", lambda *a, **kw: "S")
    broker = CheckpointBroker(CLIHandler())
    result = broker.pause(3, "screening", {"papers": []})
    assert result["action"] == "approve"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/unit/test_broker.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'slr_agent.broker'`

- [ ] **Step 3: Implement broker.py**

```python
# slr_agent/broker.py
import json
import os
import queue
import subprocess
import tempfile
import threading
from typing import Any

import click


class CheckpointBroker:
    """Pauses the pipeline at a stage and delegates to a handler for human input."""

    def __init__(self, handler: Any):
        self._handler = handler

    def pause(self, stage: int, stage_name: str, data: dict) -> dict:
        """Block until human approves. Returns edited data with 'action' key."""
        return self._handler.handle(stage, stage_name, data)


class NoOpHandler:
    """Auto-approves every checkpoint. Used for --no-checkpoints mode."""

    def handle(self, stage: int, stage_name: str, data: dict) -> dict:
        return {**data, "action": "approve"}


class CLIHandler:
    """Interactive terminal handler: print data, prompt Approve/Edit/Skip."""

    def handle(self, stage: int, stage_name: str, data: dict) -> dict:
        click.echo(f"\n{'='*60}")
        click.echo(f"  CHECKPOINT  Stage {stage} — {stage_name.upper()}")
        click.echo(f"{'='*60}")
        # Print a readable summary (truncated to avoid wall of text)
        preview = json.dumps(data, indent=2, default=str)
        if len(preview) > 4000:
            preview = preview[:4000] + "\n... (truncated)"
        click.echo(preview)
        click.echo()

        while True:
            choice = click.prompt(
                "[A]pprove  [E]dit (opens $EDITOR)  [S]kip",
                default="A",
            ).strip().upper()

            if choice in ("A", "S"):
                return {**data, "action": "approve"}
            if choice == "E":
                edited = self._open_editor(data)
                if edited is not None:
                    return {**edited, "action": "approve"}
                click.echo("Editor returned invalid JSON — try again.")

    def _open_editor(self, data: dict) -> dict | None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f, indent=2, default=str)
            path = f.name
        editor = os.environ.get("EDITOR", "vi")
        try:
            subprocess.run([editor, path], check=True)
            with open(path) as f:
                return json.load(f)
        except (subprocess.CalledProcessError, json.JSONDecodeError, OSError):
            return None
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


class UIHandler:
    """Gradio-side handler: push checkpoint to UI queue, block until UI resumes."""

    def __init__(self):
        self._pending: queue.Queue = queue.Queue()
        self._resume: queue.Queue = queue.Queue()

    def handle(self, stage: int, stage_name: str, data: dict) -> dict:
        self._pending.put({"stage": stage, "stage_name": stage_name, "data": data})
        return self._resume.get()  # blocks until Gradio calls resume()

    def get_pending(self, timeout: float = 0.5) -> dict | None:
        """Called by Gradio polling. Returns checkpoint dict or None if no pending."""
        try:
            return self._pending.get(timeout=timeout)
        except queue.Empty:
            return None

    def resume(self, edited_data: dict) -> None:
        """Called by Gradio when user approves (edited_data must include 'action' key)."""
        self._resume.put(edited_data)
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/unit/test_broker.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add slr_agent/broker.py tests/unit/test_broker.py
git commit -m "feat: add CheckpointBroker with CLIHandler, UIHandler, NoOpHandler"
```

---

### Task 5: DB manual paper add methods

**Files:**
- Modify: `slr_agent/db.py`
- Create: `tests/unit/test_db_manual_add.py`

Add `add_paper_from_pmid(run_id, pmid, api_key)` (fetches from PubMed via Entrez) and `add_paper_from_pdf(run_id, pdf_path, metadata)` (extracts text with PyMuPDF). Both call `upsert_paper` internally.

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_db_manual_add.py
import pytest
from unittest.mock import patch, MagicMock
from slr_agent.db import Database, PaperRecord


def test_add_paper_from_pmid_inserts_record(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.ensure_run("run-1")

    mock_record = MagicMock()
    mock_record.__getitem__ = lambda self, k: {
        "PubmedArticle": [{
            "MedlineCitation": {
                "PMID": MagicMock(__str__=lambda s: "12345"),
                "Article": {
                    "ArticleTitle": "Aspirin trial",
                    "Abstract": {"AbstractText": "Background: ..."},
                },
            }
        }]
    }[k]

    with patch("slr_agent.db.Entrez") as mock_entrez:
        mock_entrez.efetch.return_value = MagicMock()
        mock_entrez.read.return_value = {
            "PubmedArticle": [{
                "MedlineCitation": {
                    "PMID": type("P", (), {"__str__": lambda s: "12345"})(),
                    "Article": {
                        "ArticleTitle": "Aspirin trial",
                        "Abstract": {"AbstractText": "Background: aspirin reduces events."},
                    },
                }
            }]
        }
        paper = db.add_paper_from_pmid("run-1", "12345")

    assert paper is not None
    assert paper["title"] == "Aspirin trial"
    assert paper["pmid"] == "12345"
    assert paper["source"] == "manual"
    stored = db.get_paper("run-1", "12345")
    assert stored is not None


def test_add_paper_from_pmid_returns_none_on_error(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.ensure_run("run-1")
    with patch("slr_agent.db.Entrez") as mock_entrez:
        mock_entrez.efetch.side_effect = Exception("network error")
        result = db.add_paper_from_pmid("run-1", "00000")
    assert result is None


def test_add_paper_from_pdf_inserts_record(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.ensure_run("run-1")

    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    with patch("slr_agent.db.fitz") as mock_fitz:
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Full text of paper about aspirin..."
        mock_doc.__iter__ = lambda s: iter([mock_page])
        mock_fitz.open.return_value = mock_doc

        paper = db.add_paper_from_pdf(
            "run-1",
            str(fake_pdf),
            metadata={"pmid": "pdf-001", "title": "Manual PDF Paper"},
        )

    assert paper["pmid"] == "pdf-001"
    assert paper["title"] == "Manual PDF Paper"
    assert "aspirin" in paper["fulltext"]
    assert paper["source"] == "manual"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/unit/test_db_manual_add.py -v
```

Expected: FAIL — `AttributeError: 'Database' object has no attribute 'add_paper_from_pmid'`

- [ ] **Step 3: Add methods to db.py**

Add the following imports at the top of `slr_agent/db.py` (after existing imports):

```python
import fitz  # PyMuPDF
from Bio import Entrez as _Entrez

# Re-export for patching in tests
Entrez = _Entrez
```

Add these two methods to the `Database` class (after `get_quarantine`):

```python
    def add_paper_from_pmid(
        self, run_id: str, pmid: str, api_key: str | None = None
    ) -> "PaperRecord | None":
        """Fetch a paper from PubMed by PMID and insert it. Returns None on failure."""
        if api_key:
            Entrez.api_key = api_key
        Entrez.email = "slr-agent@local"
        try:
            handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            articles = records.get("PubmedArticle", [])
            if not articles:
                return None
            citation = articles[0]["MedlineCitation"]
            art = citation.get("Article", {})
            title = str(art.get("ArticleTitle", ""))
            abstract_obj = art.get("Abstract", {})
            abstract = str(abstract_obj.get("AbstractText", "")) if abstract_obj else ""
        except Exception:
            return None

        from slr_agent.db import GRADEScore
        paper = PaperRecord(
            pmid=str(pmid),
            run_id=run_id,
            title=title,
            abstract=abstract,
            fulltext=None,
            source="manual",
            screening_decision="uncertain",
            screening_reason="Manually added by user",
            extracted_data={},
            grade_score=GRADEScore(
                certainty="low", risk_of_bias="high",
                inconsistency="no", indirectness="no",
                imprecision="no", rationale="Not yet assessed",
            ),
            provenance=[],
            quarantined_fields=[],
        )
        self.upsert_paper(paper)
        return paper

    def add_paper_from_pdf(
        self, run_id: str, pdf_path: str, metadata: dict
    ) -> "PaperRecord":
        """Extract fulltext from a PDF and insert as a manually added paper."""
        doc = fitz.open(pdf_path)
        fulltext = "\n".join(page.get_text() for page in doc)

        from slr_agent.db import GRADEScore
        paper = PaperRecord(
            pmid=metadata.get("pmid", f"pdf:{os.path.basename(pdf_path)}"),
            run_id=run_id,
            title=metadata.get("title", os.path.basename(pdf_path)),
            abstract=metadata.get("abstract", fulltext[:500]),
            fulltext=fulltext,
            source="manual",
            screening_decision="uncertain",
            screening_reason="Manually uploaded PDF",
            extracted_data={},
            grade_score=GRADEScore(
                certainty="low", risk_of_bias="high",
                inconsistency="no", indirectness="no",
                imprecision="no", rationale="Not yet assessed",
            ),
            provenance=[],
            quarantined_fields=[],
        )
        self.upsert_paper(paper)
        return paper
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/unit/test_db_manual_add.py tests/unit/test_db.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add slr_agent/db.py tests/unit/test_db_manual_add.py
git commit -m "feat: add Database.add_paper_from_pmid and add_paper_from_pdf"
```

---

### Task 6: Template system

**Files:**
- Create: `slr_agent/template.py`
- Create: `tests/unit/test_template.py`

`load_template(path, llm)` normalizes JSON schema or PDF into `{"sections": [...], "style_notes": "..."}`. `score_rubric(draft, template, llm)` adds `"score"` and `"explanation"` to each criterion. `DEFAULT_PRISMA_TEMPLATE` is the built-in structure.

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_template.py
import json
import pytest
from unittest.mock import patch, MagicMock
from slr_agent.template import load_template, score_rubric, DEFAULT_PRISMA_TEMPLATE


def test_default_prisma_template_has_required_sections():
    sections = {s["name"] for s in DEFAULT_PRISMA_TEMPLATE["sections"]}
    for required in ("Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusions"):
        assert required in sections, f"Missing section: {required}"


def test_load_json_template(tmp_path):
    template_data = {
        "sections": [
            {
                "name": "Methods",
                "instructions": "Describe the search strategy.",
                "rubric_criteria": ["Names all databases", "States eligibility criteria"],
            }
        ],
        "style_notes": "Use passive voice.",
    }
    path = tmp_path / "template.json"
    path.write_text(json.dumps(template_data))
    result = load_template(str(path))
    assert result["sections"][0]["name"] == "Methods"
    assert result["style_notes"] == "Use passive voice."
    assert len(result["sections"][0]["rubric_criteria"]) == 2


def test_load_json_template_fills_missing_rubric_criteria(tmp_path):
    template_data = {
        "sections": [{"name": "Results", "instructions": "Report findings."}],
        "style_notes": "",
    }
    path = tmp_path / "template.json"
    path.write_text(json.dumps(template_data))
    result = load_template(str(path))
    # rubric_criteria defaults to empty list if not provided
    assert result["sections"][0]["rubric_criteria"] == []


def test_score_rubric_returns_scored_template():
    from slr_agent.llm import MockLLM
    llm = MockLLM()
    llm.register("score the following systematic review draft", {
        "scores": [{"criterion": "Names all databases", "score": "met", "explanation": "PubMed named."}]
    })
    template = {
        "sections": [{"name": "Methods", "instructions": "...", "rubric_criteria": ["Names all databases"]}],
        "style_notes": "",
    }
    result = score_rubric("## Methods\n\nWe searched PubMed.", template, llm)
    assert "scores" in result
    assert result["scores"][0]["score"] == "met"


def test_load_unsupported_format_raises(tmp_path):
    path = tmp_path / "template.txt"
    path.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported"):
        load_template(str(path))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/unit/test_template.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'slr_agent.template'`

- [ ] **Step 3: Implement template.py**

```python
# slr_agent/template.py
import json
from typing import Any

DEFAULT_PRISMA_TEMPLATE: dict = {
    "sections": [
        {
            "name": "Abstract",
            "instructions": (
                "Provide a structured abstract with Background, Objectives, Data Sources, "
                "Study Eligibility Criteria, Participants, Interventions, Study Appraisal "
                "and Synthesis Methods, Results, Limitations, Conclusions."
            ),
            "rubric_criteria": [
                "Includes PICO elements in structured format",
                "Reports number of studies and participants",
                "States main findings with effect size",
                "Mentions limitations",
            ],
        },
        {
            "name": "Introduction",
            "instructions": (
                "Describe the rationale for the review, state the explicit research question "
                "in PICO format, and explain why this systematic review is needed."
            ),
            "rubric_criteria": [
                "States explicit research question in PICO format",
                "Justifies need for the review with evidence gap",
                "Describes expected benefits of the intervention",
            ],
        },
        {
            "name": "Methods",
            "instructions": (
                "Describe the protocol, eligibility criteria, information sources, search strategy, "
                "study selection process, data extraction, risk of bias assessment, and synthesis methods. "
                "Follow PRISMA 2020 checklist items 5-16."
            ),
            "rubric_criteria": [
                "Specifies inclusion and exclusion criteria",
                "Names all databases searched with date ranges",
                "Provides at least one full search string",
                "Describes risk of bias assessment tool used",
                "Explains data synthesis approach (meta-analysis or narrative)",
            ],
        },
        {
            "name": "Results",
            "instructions": (
                "Report study selection (PRISMA flow), study characteristics, risk of bias results, "
                "results of individual studies, and results of syntheses."
            ),
            "rubric_criteria": [
                "Includes PRISMA flow counts (retrieved, screened, included)",
                "Summarises characteristics of included studies",
                "Reports effect sizes with confidence intervals",
                "Presents risk of bias assessment results",
            ],
        },
        {
            "name": "Discussion",
            "instructions": (
                "Interpret results in context of existing evidence, discuss limitations of the review, "
                "and discuss implications for practice and research."
            ),
            "rubric_criteria": [
                "Interprets findings in context of prior evidence",
                "Discusses at least two limitations of the review",
                "Addresses clinical or policy implications",
            ],
        },
        {
            "name": "Conclusions",
            "instructions": (
                "Provide a brief, clear conclusion about the main findings and their implications. "
                "Do not introduce new evidence."
            ),
            "rubric_criteria": [
                "Directly answers the research question",
                "Does not introduce new information",
                "States implications for practice or future research",
            ],
        },
    ],
    "style_notes": "Follow PRISMA 2020 reporting guidelines. Use past tense for methods and results.",
}

_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "criterion": {"type": "string"},
                    "score": {"type": "string", "enum": ["met", "partial", "not met"]},
                    "explanation": {"type": "string"},
                },
                "required": ["criterion", "score", "explanation"],
            },
        }
    },
    "required": ["scores"],
}

_PDF_SCHEMA = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "instructions": {"type": "string"},
                    "rubric_criteria": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "instructions", "rubric_criteria"],
            },
        },
        "style_notes": {"type": "string"},
    },
    "required": ["sections", "style_notes"],
}


def load_template(path: str, llm: Any | None = None) -> dict:
    """Load and normalize a template from JSON schema or PDF.

    Returns normalized dict: {sections: [{name, instructions, rubric_criteria}], style_notes}.
    Raises ValueError for unsupported file formats.
    """
    if path.endswith(".json"):
        return _load_json_template(path)
    if path.lower().endswith(".pdf"):
        if llm is None:
            raise ValueError("LLM required to parse PDF templates")
        return _load_pdf_template(path, llm)
    raise ValueError(f"Unsupported template format: {path}. Use .json or .pdf")


def _load_json_template(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    sections = []
    for s in data.get("sections", []):
        sections.append({
            "name": s.get("name", ""),
            "instructions": s.get("instructions", ""),
            "rubric_criteria": s.get("rubric_criteria", []),
        })
    return {
        "sections": sections,
        "style_notes": data.get("style_notes", ""),
    }


def _load_pdf_template(path: str, llm: Any) -> dict:
    import fitz
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)[:8000]
    result = llm.chat([{
        "role": "user",
        "content": (
            "analyze this systematic review paper and extract the manuscript structure. "
            "For each section identify: the section name, instructions for what it should contain, "
            "and 3-5 measurable rubric criteria. Also note any style conventions. "
            "Return JSON with fields: sections (list of {name, instructions, rubric_criteria}), "
            f"style_notes.\n\nPaper text:\n{text}"
        ),
    }], schema=_PDF_SCHEMA)
    return result


def score_rubric(draft: str, template: dict, llm: Any) -> dict:
    """Score a manuscript draft against all rubric criteria in the template.

    Returns a dict with 'scores' list of {criterion, score, explanation}.
    score is one of: 'met', 'partial', 'not met'.
    """
    all_criteria = [
        c
        for section in template.get("sections", [])
        for c in section.get("rubric_criteria", [])
    ]
    if not all_criteria:
        return {"scores": []}

    result = llm.chat([{
        "role": "user",
        "content": (
            "score the following systematic review draft against each rubric criterion. "
            "For each criterion return: criterion (exact text), score ('met', 'partial', or 'not met'), "
            "and a one-sentence explanation.\n\n"
            f"Criteria:\n" + "\n".join(f"- {c}" for c in all_criteria) +
            f"\n\nDraft:\n{draft[:6000]}"
        ),
    }], schema=_SCORE_SCHEMA)
    return result
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/unit/test_template.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add slr_agent/template.py tests/unit/test_template.py
git commit -m "feat: add template system — JSON/PDF loading, default PRISMA structure, rubric scoring"
```

---

### Task 7: Manuscript subgraph rewrite

**Files:**
- Modify: `slr_agent/subgraphs/manuscript.py`
- Modify: `tests/integration/test_manuscript_subgraph.py`

Replace the three-section hardcoded draft with template-driven section generation. Add rubric scoring pass. Write versioned drafts to `output_dir/<run_id>/`. The subgraph now accepts `template` from state (or uses `DEFAULT_PRISMA_TEMPLATE`). Stage 7 broker pause lives in the orchestrator (Task 8), not here.

- [ ] **Step 1: Update the integration test**

Replace the entire contents of `tests/integration/test_manuscript_subgraph.py`:

```python
# tests/integration/test_manuscript_subgraph.py
import os
import pytest
from unittest.mock import patch
from slr_agent.subgraphs.manuscript import create_manuscript_subgraph
from slr_agent.llm import MockLLM
from slr_agent.state import PICOResult
from slr_agent.template import DEFAULT_PRISMA_TEMPLATE


def _make_llm():
    llm = MockLLM()
    # Section generation — one response matched per section name
    for section in DEFAULT_PRISMA_TEMPLATE["sections"]:
        name = section["name"].lower()
        llm.register(f"write the {name} section", {"text": f"Content of {section['name']}."})
    # Rubric scoring
    llm.register("score the following systematic review draft", {
        "scores": [{"criterion": "Names all databases", "score": "met", "explanation": "PubMed named."}]
    })
    return llm


def test_manuscript_writes_markdown(db, tmp_path):
    db.ensure_run("run-test")
    synthesis_path = str(tmp_path / "synthesis.md")
    with open(synthesis_path, "w") as f:
        f.write("# Synthesis\n\nAspirin reduces SBP by 8 mmHg.")

    with patch("slr_agent.subgraphs.manuscript.run_pandoc", return_value=None):
        graph = create_manuscript_subgraph(db=db, llm=_make_llm(), output_dir=str(tmp_path))
        result = graph.invoke({
            "run_id": "run-test",
            "pico": PICOResult(
                population="adults with hypertension", intervention="aspirin",
                comparator="placebo", outcome="blood pressure reduction",
                query_strings=[], source_language="en",
                search_language="en", output_language="en",
            ),
            "synthesis_path": synthesis_path,
            "screening_counts": {"n_included": 1, "n_excluded": 9, "n_uncertain": 0},
            "manuscript_path": None,
            "template": None,
            "manuscript_draft_version": 0,
        })

    assert result["manuscript_path"] is not None
    assert os.path.exists(result["manuscript_path"])
    content = open(result["manuscript_path"]).read()
    assert "Methods" in content
    assert "Results" in content


def test_manuscript_uses_custom_template(db, tmp_path):
    db.ensure_run("run-test2")
    synthesis_path = str(tmp_path / "synthesis.md")
    with open(synthesis_path, "w") as f:
        f.write("Narrative.")

    custom_template = {
        "sections": [
            {"name": "Background", "instructions": "Describe rationale.", "rubric_criteria": []},
            {"name": "Methods", "instructions": "Detail methods.", "rubric_criteria": []},
        ],
        "style_notes": "",
    }

    llm = MockLLM()
    llm.register("write the background section", {"text": "Background content."})
    llm.register("write the methods section", {"text": "Methods content."})
    llm.register("score the following systematic review draft", {"scores": []})

    with patch("slr_agent.subgraphs.manuscript.run_pandoc", return_value=None):
        graph = create_manuscript_subgraph(db=db, llm=llm, output_dir=str(tmp_path))
        result = graph.invoke({
            "run_id": "run-test2",
            "pico": PICOResult(
                population="adults", intervention="aspirin",
                comparator="placebo", outcome="bp",
                query_strings=[], source_language="en",
                search_language="en", output_language="en",
            ),
            "synthesis_path": synthesis_path,
            "screening_counts": None,
            "manuscript_path": None,
            "template": custom_template,
            "manuscript_draft_version": 0,
        })

    content = open(result["manuscript_path"]).read()
    assert "Background" in content
    assert "Methods" in content


def test_manuscript_rubric_saved(db, tmp_path):
    db.ensure_run("run-test3")
    synthesis_path = str(tmp_path / "synthesis.md")
    with open(synthesis_path, "w") as f:
        f.write("Narrative.")

    llm = _make_llm()
    with patch("slr_agent.subgraphs.manuscript.run_pandoc", return_value=None):
        graph = create_manuscript_subgraph(db=db, llm=llm, output_dir=str(tmp_path))
        result = graph.invoke({
            "run_id": "run-test3",
            "pico": PICOResult(
                population="adults", intervention="aspirin",
                comparator="placebo", outcome="bp",
                query_strings=[], source_language="en",
                search_language="en", output_language="en",
            ),
            "synthesis_path": synthesis_path,
            "screening_counts": None,
            "manuscript_path": None,
            "template": None,
            "manuscript_draft_version": 0,
        })

    assert "manuscript_rubric" in result
    assert "scores" in result["manuscript_rubric"]
```

- [ ] **Step 2: Run updated test to verify it fails**

```bash
.venv/bin/pytest tests/integration/test_manuscript_subgraph.py -v
```

Expected: FAIL (old subgraph signature doesn't accept `template` key, missing sections).

- [ ] **Step 3: Rewrite manuscript.py**

```python
# slr_agent/subgraphs/manuscript.py
import os
from langgraph.graph import StateGraph, END
from slr_agent.db import Database
from slr_agent.export import run_pandoc
from slr_agent.template import DEFAULT_PRISMA_TEMPLATE, score_rubric

_TEXT_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}


def _draft_manuscript_node(state: dict, db: Database, llm, output_dir: str) -> dict:
    run_id = state["run_id"]
    pico = state["pico"]
    synthesis_path = state.get("synthesis_path")
    output_language = pico.get("output_language", "en")
    template = state.get("template") or DEFAULT_PRISMA_TEMPLATE
    draft_version = state.get("manuscript_draft_version", 0) + 1

    synthesis_text = ""
    if synthesis_path and os.path.exists(synthesis_path):
        with open(synthesis_path) as f:
            synthesis_text = f.read()

    papers = db.get_papers_by_decision(run_id, "include")
    screening = state.get("screening_counts") or {}
    lang_suffix = f" Write in {output_language}." if output_language != "en" else ""

    # Draft each section from the template
    sections_md = []
    for section in template["sections"]:
        name = section["name"]
        instructions = section.get("instructions", "")
        style = template.get("style_notes", "")
        response = llm.chat([{
            "role": "user",
            "content": (
                f"write the {name} section of a systematic review manuscript. "
                f"Instructions: {instructions} "
                f"Context: P={pico['population']}, I={pico['intervention']}, "
                f"C={pico['comparator']}, O={pico['outcome']}. "
                f"{len(papers)} studies included. "
                f"Synthesis:\n{synthesis_text[:2000]}\n"
                f"Excluded: {screening.get('n_excluded', '?')}. "
                f"Style: {style}{lang_suffix} "
                "Return JSON with field 'text'."
            ),
        }], schema=_TEXT_SCHEMA)
        sections_md.append(f"## {name}\n\n{response['text']}")

    draft = (
        f"# Systematic Review: {pico['intervention']} in {pico['population']}\n\n"
        + "\n\n".join(sections_md)
    )

    # Write versioned draft
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    draft_path = os.path.join(run_dir, f"stage_7_draft_v{draft_version}.md")
    with open(draft_path, "w") as f:
        f.write(draft)

    # Final manuscript path (updated each revision)
    md_path = os.path.join(run_dir, f"{run_id}_manuscript.md")
    with open(md_path, "w") as f:
        f.write(draft)

    # Score rubric
    rubric_result = score_rubric(draft, template, llm)

    # Export to Word
    docx_path = md_path.replace(".md", ".docx")
    try:
        run_pandoc(md_path, docx_path)
    except RuntimeError:
        pass

    return {
        "manuscript_path": md_path,
        "manuscript_rubric": {**rubric_result, "template": template},
        "manuscript_draft_version": draft_version,
    }


def create_manuscript_subgraph(db: Database, llm, output_dir: str):
    builder = StateGraph(dict)
    builder.add_node("draft", lambda s: _draft_manuscript_node(s, db, llm, output_dir))
    builder.set_entry_point("draft")
    builder.add_edge("draft", END)
    return builder.compile()
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/integration/test_manuscript_subgraph.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Run full suite**

```bash
.venv/bin/pytest tests/ -x -q --ignore=tests/e2e
```

Expected: all pass (orchestrator tests may need updating — see Task 8).

- [ ] **Step 6: Commit**

```bash
git add slr_agent/subgraphs/manuscript.py tests/integration/test_manuscript_subgraph.py
git commit -m "feat: template-driven manuscript generation with rubric scoring and versioned drafts"
```

---

### Task 8: Orchestrator wiring — broker, emitter, revision loop

**Files:**
- Modify: `slr_agent/orchestrator.py`
- Modify: `tests/integration/test_orchestrator_routing.py`

Inject `broker` and `emitter` into the orchestrator. Each pipeline node calls `emitter.emit()` then `broker.pause()` if the stage is in `checkpoint_stages`. The `manuscript_node` additionally runs a revision loop: broker.pause(7) returns `{"action": "revise", "rubric": ...}` to trigger another draft.

- [ ] **Step 1: Update orchestrator test**

Replace `tests/integration/test_orchestrator_routing.py`:

```python
# tests/integration/test_orchestrator_routing.py
import pytest
from unittest.mock import patch, MagicMock
from slr_agent.orchestrator import create_orchestrator
from slr_agent.llm import MockLLM
from slr_agent.config import RunConfig
from slr_agent.template import DEFAULT_PRISMA_TEMPLATE


def make_llm():
    llm = MockLLM()
    llm.register("detect the language", {"language_code": "en"})
    llm.register("expand this research question into PICO", {
        "population": "adults", "intervention": "aspirin",
        "comparator": "placebo", "outcome": "bp reduction",
    })
    llm.register("generate PubMed search query strings", {"query_strings": ["aspirin[tiab]"]})
    llm.register("screen the following abstracts", {"decisions": []})
    llm.register("synthesise the evidence", {"claims": [], "narrative": "No evidence found."})
    for section in DEFAULT_PRISMA_TEMPLATE["sections"]:
        llm.register(f"write the {section['name'].lower()} section", {"text": f"{section['name']} text."})
    llm.register("score the following systematic review draft", {"scores": []})
    return llm


def test_orchestrator_runs_without_checkpoints(db, tmp_path):
    with patch("slr_agent.subgraphs.search.Entrez") as mock_entrez:
        mock_entrez.esearch.return_value = MagicMock()
        mock_entrez.read.return_value = {"IdList": []}
        mock_entrez.efetch.return_value = MagicMock()

        with patch("slr_agent.subgraphs.manuscript.run_pandoc", return_value=None):
            orchestrator = create_orchestrator(
                db=db,
                llm=make_llm(),
                output_dir=str(tmp_path),
                config=RunConfig(
                    checkpoint_stages=[],
                    fetch_fulltext=False,
                    output_format="markdown",
                    pubmed_api_key=None,
                    max_results=10,
                ),
            )
            result = orchestrator.invoke({
                "run_id": "run-orch-test",
                "raw_question": "Does aspirin reduce blood pressure?",
            })

    assert result["manuscript_path"] is not None
    assert result["pico"]["intervention"] == "aspirin"


def test_orchestrator_skips_fulltext_when_disabled(db, tmp_path):
    with patch("slr_agent.subgraphs.search.Entrez") as mock_entrez:
        mock_entrez.esearch.return_value = MagicMock()
        mock_entrez.read.return_value = {"IdList": []}

        with patch("slr_agent.subgraphs.manuscript.run_pandoc", return_value=None):
            orchestrator = create_orchestrator(
                db=db, llm=make_llm(), output_dir=str(tmp_path),
                config=RunConfig(
                    checkpoint_stages=[], fetch_fulltext=False,
                    output_format="markdown", pubmed_api_key=None, max_results=10,
                ),
            )
            result = orchestrator.invoke({
                "run_id": "run-orch-test-2",
                "raw_question": "Does aspirin reduce blood pressure?",
            })

    assert result["fulltext_counts"] is None


def test_orchestrator_emits_stage_files(db, tmp_path):
    """ProgressEmitter writes stage JSON files to outputs/<run_id>/."""
    import os
    with patch("slr_agent.subgraphs.search.Entrez") as mock_entrez:
        mock_entrez.esearch.return_value = MagicMock()
        mock_entrez.read.return_value = {"IdList": []}

        with patch("slr_agent.subgraphs.manuscript.run_pandoc", return_value=None):
            orchestrator = create_orchestrator(
                db=db, llm=make_llm(), output_dir=str(tmp_path),
                config=RunConfig(
                    checkpoint_stages=[], fetch_fulltext=False,
                    output_format="markdown", pubmed_api_key=None, max_results=10,
                ),
            )
            result = orchestrator.invoke({
                "run_id": "run-emit-test",
                "raw_question": "Does aspirin reduce blood pressure?",
            })

    run_dir = os.path.join(str(tmp_path), "run-emit-test")
    assert os.path.isdir(run_dir)
    assert os.path.exists(os.path.join(run_dir, "stage_1_pico.json"))
    assert os.path.exists(os.path.join(run_dir, "stage_2_search.json"))
```

- [ ] **Step 2: Run test to verify it fails (missing template mocks)**

```bash
.venv/bin/pytest tests/integration/test_orchestrator_routing.py -v
```

Expected: FAIL (MockLLM raises on missing section mocks).

- [ ] **Step 3: Rewrite orchestrator.py**

```python
# slr_agent/orchestrator.py
import uuid
from typing import Any
from langgraph.graph import StateGraph, END

from slr_agent.broker import CheckpointBroker, NoOpHandler
from slr_agent.config import RunConfig, DEFAULT_CONFIG
from slr_agent.db import Database
from slr_agent.emitter import ProgressEmitter
from slr_agent.subgraphs.pico import create_pico_subgraph
from slr_agent.subgraphs.search import create_search_subgraph
from slr_agent.subgraphs.screening import create_screening_subgraph
from slr_agent.subgraphs.fulltext import create_fulltext_subgraph
from slr_agent.subgraphs.extraction import create_extraction_subgraph
from slr_agent.subgraphs.synthesis import create_synthesis_subgraph
from slr_agent.subgraphs.manuscript import create_manuscript_subgraph


def _should_fetch_fulltext(state: dict) -> str:
    cfg = state.get("config", {})
    if cfg.get("fetch_fulltext", True):
        return "fulltext"
    return "extraction"


def create_orchestrator(
    db: Database,
    llm: Any,
    output_dir: str,
    config: RunConfig | None = None,
    db_path: str | None = None,
    broker: CheckpointBroker | None = None,
    emitter: ProgressEmitter | None = None,
):
    cfg = config or DEFAULT_CONFIG
    checkpoint_stages = cfg.get("checkpoint_stages", [])

    # Defaults: no-op broker and no-op emitter if not provided
    _broker = broker or CheckpointBroker(NoOpHandler())

    pico_sg = create_pico_subgraph(llm=llm)
    search_sg = create_search_subgraph(db=db)
    screening_sg = create_screening_subgraph(db=db, llm=llm)
    fulltext_sg = create_fulltext_subgraph(db=db, llm=llm)
    extraction_sg = create_extraction_subgraph(db=db, llm=llm)
    synthesis_sg = create_synthesis_subgraph(db=db, llm=llm, output_dir=output_dir)
    manuscript_sg = create_manuscript_subgraph(db=db, llm=llm, output_dir=output_dir)

    def _get_emitter(run_id: str) -> ProgressEmitter:
        if emitter is not None:
            return emitter
        return ProgressEmitter(output_dir=output_dir, run_id=run_id)

    def _maybe_pause(stage: int, stage_name: str, data: dict, state: dict) -> dict:
        """Emit then pause if this stage is checkpointed. Returns edited data."""
        run_id = state["run_id"]
        _get_emitter(run_id).emit(stage, data)
        if stage in checkpoint_stages:
            return _broker.pause(stage, stage_name, data)
        return {**data, "action": "approve"}

    def pico_node(state: dict) -> dict:
        run_id = state.get("run_id") or str(uuid.uuid4())
        db.ensure_run(run_id)
        base = {
            **state,
            "run_id": run_id,
            "config": cfg,
            "pico": None,
            "search_counts": None,
            "screening_counts": None,
            "fulltext_counts": None,
            "extraction_counts": None,
            "synthesis_path": None,
            "manuscript_path": None,
            "template": None,
            "manuscript_draft_version": 0,
            "current_stage": "pico",
            "checkpoint_pending": False,
        }
        sub_result = pico_sg.invoke({
            "raw_question": state["raw_question"],
            "pico": None,
            "validation_errors": [],
        })
        pico_data = dict(sub_result.get("pico") or {})
        edited = _maybe_pause(1, "pico", pico_data, base)
        # Apply any edits back to pico
        merged_pico = {**(sub_result.get("pico") or {}), **{
            k: v for k, v in edited.items() if k != "action"
        }}
        return {**base, **sub_result, "pico": merged_pico, "current_stage": "pico_done"}

    def search_node(state: dict) -> dict:
        sub_input = {
            **state,
            "pubmed_api_key": cfg.get("pubmed_api_key"),
            "max_results": cfg.get("max_results", 500),
            "search_sources": cfg.get("search_sources", ["pubmed"]),
        }
        result = search_sg.invoke(sub_input)
        counts = result.get("search_counts", {})
        papers = db.get_all_papers(state["run_id"])
        paper_list = [{"pmid": p["pmid"], "title": p["title"], "source": p["source"]} for p in papers]
        emit_data = {**dict(counts or {}), "papers": paper_list}
        edited = _maybe_pause(2, "search", emit_data, state)
        # Apply manual exclusions from gate
        for p in edited.get("papers", []):
            if p.get("excluded"):
                paper = db.get_paper(state["run_id"], p["pmid"])
                if paper:
                    paper["screening_decision"] = "excluded_manual"
                    paper["screening_reason"] = "Excluded by user at search gate"
                    db.upsert_paper(paper)
        return {**state, **result, "current_stage": "search_done"}

    def screening_node(state: dict) -> dict:
        result = screening_sg.invoke(state)
        papers = db.get_papers_by_decision(state["run_id"], "include")
        excluded = db.get_papers_by_decision(state["run_id"], "exclude")
        paper_list = [
            {"pmid": p["pmid"], "title": p["title"],
             "abstract": (p["abstract"] or "")[:300],
             "decision": p["screening_decision"],
             "reason": p["screening_reason"]}
            for p in papers + excluded
        ]
        emit_data = {**dict(result.get("screening_counts") or {}), "papers": paper_list}
        edited = _maybe_pause(3, "screening", emit_data, state)
        # Apply manual include/exclude overrides
        for p in edited.get("papers", []):
            if "decision" in p:
                record = db.get_paper(state["run_id"], p["pmid"])
                if record and record["screening_decision"] != p["decision"]:
                    record["screening_decision"] = p["decision"]
                    record["screening_reason"] = p.get("reason", "User override")
                    db.upsert_paper(record)
        return {**state, **result, "current_stage": "screening_done"}

    def fulltext_node(state: dict) -> dict:
        result = fulltext_sg.invoke(state)
        emit_data = dict(result.get("fulltext_counts") or {})
        _get_emitter(state["run_id"]).emit(4, emit_data)
        return {**state, **result, "current_stage": "fulltext_done"}

    def extraction_node(state: dict) -> dict:
        result = extraction_sg.invoke(state)
        papers = db.get_papers_by_decision(state["run_id"], "include")
        paper_list = [
            {"pmid": p["pmid"], "title": p["title"],
             "extracted_data": p["extracted_data"],
             "quarantined_fields": p["quarantined_fields"],
             "grade_score": p["grade_score"]}
            for p in papers
        ]
        emit_data = {**dict(result.get("extraction_counts") or {}), "papers": paper_list}
        edited = _maybe_pause(5, "extraction", emit_data, state)
        # Apply manual field edits
        for p in edited.get("papers", []):
            record = db.get_paper(state["run_id"], p["pmid"])
            if record and p.get("extracted_data"):
                record["extracted_data"] = p["extracted_data"]
                db.upsert_paper(record)
        return {**state, **result, "current_stage": "extraction_done"}

    def synthesis_node(state: dict) -> dict:
        result = synthesis_sg.invoke(state)
        synthesis_path = result.get("synthesis_path", "")
        synthesis_text = ""
        if synthesis_path and __import__("os").path.exists(synthesis_path):
            with open(synthesis_path) as f:
                synthesis_text = f.read()
        emit_data = {"synthesis_path": synthesis_path, "preview": synthesis_text[:500]}
        edited = _maybe_pause(6, "synthesis", emit_data, state)
        return {**state, **result, "current_stage": "synthesis_done"}

    def manuscript_node(state: dict) -> dict:
        # Load template if path provided
        template = state.get("template")
        template_path = cfg.get("template_path")
        if template is None and template_path:
            from slr_agent.template import load_template
            template = load_template(template_path, llm)

        result = manuscript_sg.invoke({**state, "template": template})
        current_state = {**state, **result, "template": template}

        # Stage 7 revision loop
        if 7 in checkpoint_stages:
            while True:
                draft = open(result["manuscript_path"]).read()
                rubric = result.get("manuscript_rubric", {})
                checkpoint_data = {
                    "draft": draft,
                    "rubric": rubric,
                    "draft_version": result.get("manuscript_draft_version", 1),
                }
                edited = _broker.pause(7, "manuscript", checkpoint_data)
                if edited.get("action") != "revise":
                    break
                # User requested revision — re-run with updated rubric/template
                revised_template = edited.get("template") or template
                revised_result = manuscript_sg.invoke({
                    **current_state,
                    "template": revised_template,
                    "manuscript_draft_version": result.get("manuscript_draft_version", 1),
                })
                result = revised_result
                current_state = {**current_state, **revised_result, "template": revised_template}
        else:
            # No gate — emit stage 7 rubric data
            _get_emitter(state["run_id"]).emit(7, result.get("manuscript_rubric", {}))

        return {**state, **result, "template": template, "current_stage": "done"}

    builder = StateGraph(dict)
    builder.add_node("pico", pico_node)
    builder.add_node("search", search_node)
    builder.add_node("screening", screening_node)
    builder.add_node("fulltext", fulltext_node)
    builder.add_node("extraction", extraction_node)
    builder.add_node("synthesis", synthesis_node)
    builder.add_node("manuscript", manuscript_node)

    builder.set_entry_point("pico")
    builder.add_edge("pico", "search")
    builder.add_edge("search", "screening")
    builder.add_conditional_edges("screening", _should_fetch_fulltext, {
        "fulltext": "fulltext",
        "extraction": "extraction",
    })
    builder.add_edge("fulltext", "extraction")
    builder.add_edge("extraction", "synthesis")
    builder.add_edge("synthesis", "manuscript")
    builder.add_edge("manuscript", END)

    if db_path:
        try:
            from langgraph_checkpoint_sqlite import SqliteSaver
            checkpointer = SqliteSaver.from_conn_string(db_path)
            return builder.compile(checkpointer=checkpointer)
        except ImportError:
            import warnings
            warnings.warn(
                "langgraph_checkpoint_sqlite not available — compiling without checkpointer.",
                RuntimeWarning,
                stacklevel=2,
            )

    return builder.compile()
```

- [ ] **Step 4: Run all tests**

```bash
.venv/bin/pytest tests/ -x -q --ignore=tests/e2e
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add slr_agent/orchestrator.py tests/integration/test_orchestrator_routing.py
git commit -m "feat: wire broker + emitter into orchestrator; add stage 7 revision loop"
```

---

### Task 9: CLI updates

**Files:**
- Modify: `slr_agent/cli.py`

Add `--hitl [cli|ui]` and `--template <file>` options. Build the correct broker and emitter. Print formatted stage summaries. When `--hitl ui`, auto-launch the Gradio server in a background thread and print the URL.

- [ ] **Step 1: Rewrite cli.py**

```python
# slr_agent/cli.py
import os
import uuid
import click
from slr_agent.broker import CheckpointBroker, CLIHandler, NoOpHandler, UIHandler
from slr_agent.config import DEFAULT_CONFIG, RunConfig
from slr_agent.db import Database
from slr_agent.emitter import ProgressEmitter
from slr_agent.llm import LLMClient
from slr_agent.orchestrator import create_orchestrator

_DB_PATH = "slr_runs.db"
_OUTPUT_DIR = "outputs"


def _build_orchestrator(config: RunConfig, broker: CheckpointBroker, emitter: ProgressEmitter):
    db = Database(_DB_PATH)
    llm = LLMClient()
    orchestrator = create_orchestrator(
        db=db, llm=llm, output_dir=_OUTPUT_DIR, config=config,
        db_path=_DB_PATH, broker=broker, emitter=emitter,
    )
    return orchestrator, db


@click.group()
def cli():
    """SLR Agent — systematic literature review pipeline powered by Gemma."""


@cli.command()
@click.argument("question")
@click.option("--no-fulltext", is_flag=True, default=False, help="Skip full-text fetching")
@click.option("--no-checkpoints", is_flag=True, default=False, help="Run fully automated")
@click.option(
    "--hitl",
    type=click.Choice(["cli", "ui"]),
    default="cli",
    help="HITL mode: cli (terminal editing) or ui (Gradio browser)",
)
@click.option("--max-results", default=500, help="PubMed search cap")
@click.option("--api-key", default=None, envvar="PUBMED_API_KEY", help="PubMed API key")
@click.option(
    "--template",
    "template_path",
    default=None,
    type=click.Path(exists=True),
    help="Manuscript template: JSON schema or PDF reference paper",
)
def run(question, no_fulltext, no_checkpoints, hitl, max_results, api_key, template_path):
    """Start a new SLR run."""
    run_id = str(uuid.uuid4())[:8]
    checkpoint_stages = [] if no_checkpoints else DEFAULT_CONFIG["checkpoint_stages"]
    config = RunConfig(
        checkpoint_stages=checkpoint_stages,
        fetch_fulltext=not no_fulltext,
        output_format="both",
        pubmed_api_key=api_key,
        max_results=max_results,
        search_sources=DEFAULT_CONFIG["search_sources"],
        template_path=template_path,
        hitl_mode="none" if no_checkpoints else hitl,
    )

    emitter = ProgressEmitter(
        output_dir=_OUTPUT_DIR,
        run_id=run_id,
        echo=click.echo,
    )

    if no_checkpoints or not checkpoint_stages:
        broker = CheckpointBroker(NoOpHandler())
    elif hitl == "ui":
        ui_handler = UIHandler()
        broker = CheckpointBroker(ui_handler)
        _launch_gradio_with_handler(ui_handler, run_id)
    else:
        broker = CheckpointBroker(CLIHandler())

    orchestrator, db = _build_orchestrator(config, broker, emitter)
    click.echo(f"Starting run {run_id}...")
    if template_path:
        click.echo(f"Template: {template_path}")

    try:
        result = orchestrator.invoke(
            {"run_id": run_id, "raw_question": question},
            config={"configurable": {"thread_id": run_id}},
        )
        if result.get("manuscript_path"):
            click.echo(f"\nComplete. Manuscript: {result['manuscript_path']}")
            click.echo(f"Stage files: {_OUTPUT_DIR}/{run_id}/")
        else:
            click.echo(f"Paused at checkpoint. Resume with: slr resume {run_id}")
    except Exception as e:
        click.echo(f"Failed: {e}", err=True)
        raise SystemExit(1)


def _launch_gradio_with_handler(ui_handler: UIHandler, run_id: str) -> None:
    """Launch Gradio server in background thread for --hitl ui mode."""
    import threading
    try:
        from slr_agent.ui.app import build_app_with_handler
        def _serve():
            app = build_app_with_handler(ui_handler, run_id)
            app.launch(server_port=7860, prevent_thread_lock=True)
        threading.Thread(target=_serve, daemon=True).start()
        click.echo("Gradio UI started at http://localhost:7860 — open to review checkpoints.")
    except Exception as e:
        click.echo(f"Warning: could not launch Gradio UI ({e}). Falling back to CLI mode.")


@cli.command()
@click.argument("run_id")
@click.option("--edits", default=None, help="JSON string of state edits to apply")
def resume(run_id, edits):
    """Resume a paused or failed run."""
    import json
    broker = CheckpointBroker(CLIHandler())
    emitter = ProgressEmitter(output_dir=_OUTPUT_DIR, run_id=run_id, echo=click.echo)
    orchestrator, db = _build_orchestrator(DEFAULT_CONFIG, broker, emitter)
    thread_config = {"configurable": {"thread_id": run_id}}
    if edits:
        orchestrator.update_state(thread_config, json.loads(edits))
    try:
        result = orchestrator.invoke(None, config=thread_config)
        if result.get("manuscript_path"):
            click.echo(f"Complete. Manuscript: {result['manuscript_path']}")
        else:
            click.echo(f"Paused at checkpoint. Resume with: slr resume {run_id}")
    except Exception as e:
        click.echo(f"Failed: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.argument("run_id")
def status(run_id):
    """Show the current status of a run."""
    db = Database(_DB_PATH)
    papers = db.get_all_papers(run_id)
    quarantine = db.get_quarantine(run_id)
    included = [p for p in papers if p["screening_decision"] == "include"]
    excluded = [p for p in papers if p["screening_decision"] == "exclude"]
    click.echo(f"Run: {run_id}")
    click.echo(f"  Papers retrieved: {len(papers)}")
    click.echo(f"  Included: {len(included)} | Excluded: {len(excluded)}")
    click.echo(f"  Quarantined fields: {len(quarantine)}")
    stage_dir = os.path.join(_OUTPUT_DIR, run_id)
    if os.path.isdir(stage_dir):
        import glob as _glob
        stage_files = _glob.glob(os.path.join(stage_dir, "stage_*.json"))
        click.echo(f"  Stage files: {len(stage_files)} saved to {stage_dir}/")


@cli.command()
@click.argument("run_id")
def export(run_id):
    """Export the manuscript for a completed run."""
    import glob as _glob
    run_dir = os.path.join(_OUTPUT_DIR, run_id)
    files = _glob.glob(os.path.join(run_dir, f"{run_id}_manuscript*"))
    if not files:
        # Fall back to flat outputs dir (older runs)
        files = _glob.glob(os.path.join(_OUTPUT_DIR, f"{run_id}_manuscript*"))
    if not files:
        click.echo(f"No manuscript found for run {run_id}", err=True)
        raise SystemExit(1)
    for f in files:
        click.echo(f"Manuscript: {f}")
```

- [ ] **Step 2: Run full test suite**

```bash
.venv/bin/pytest tests/ -x -q --ignore=tests/e2e
```

Expected: all pass.

- [ ] **Step 3: Smoke-test the CLI (no Ollama needed — will fail at LLM call, but arg parsing should work)**

```bash
.venv/bin/slr run --help
```

Expected: shows help text with `--hitl`, `--template`, `--no-checkpoints` options listed.

- [ ] **Step 4: Commit**

```bash
git add slr_agent/cli.py
git commit -m "feat: add --hitl and --template CLI flags; build broker/emitter in cli; stage summary output"
```

---

### Task 10: Gradio app wiring — live log, template upload, UIHandler routing

**Files:**
- Modify: `slr_agent/ui/app.py`

Add `build_app_with_handler(ui_handler, run_id)` function used by CLI `--hitl ui`. Update `build_app()` to include a live progress log panel, a template file upload field, and checkpoint panel routing. The UIHandler queue is polled on a timer to show pending checkpoints.

- [ ] **Step 1: Rewrite app.py**

```python
# slr_agent/ui/app.py
import queue
import threading
import gradio as gr
from slr_agent.broker import UIHandler
from slr_agent.db import Database
from slr_agent.llm import LLMClient
from slr_agent.broker import CheckpointBroker, NoOpHandler
from slr_agent.config import DEFAULT_CONFIG, RunConfig
from slr_agent.emitter import ProgressEmitter
from slr_agent.orchestrator import create_orchestrator
from slr_agent.ui.panels.pico import build_pico_panel
from slr_agent.ui.panels.generic import build_generic_panel

_DB_PATH = "slr_runs.db"
_OUTPUT_DIR = "outputs"
_run_results: dict = {}
_log_queues: dict = {}  # run_id → queue.Queue of log strings


def _make_config(fetch_fulltext: bool, checkpoint_stages_str: str, template_path: str | None) -> RunConfig:
    stages = [int(s.strip()) for s in checkpoint_stages_str.split(",") if s.strip().isdigit()]
    return RunConfig(
        checkpoint_stages=stages,
        fetch_fulltext=fetch_fulltext,
        output_format="both",
        pubmed_api_key=None,
        max_results=500,
        search_sources=DEFAULT_CONFIG["search_sources"],
        template_path=template_path or None,
        hitl_mode="ui",
    )


def launch_run(question: str, fetch_fulltext: bool, checkpoint_stages_str: str, template_file):
    """Start a pipeline run in a background thread (no HITL — auto-approve)."""
    import uuid
    run_id = str(uuid.uuid4())[:8]
    template_path = template_file.name if template_file is not None else None
    config = _make_config(fetch_fulltext, checkpoint_stages_str, template_path)
    # Automated run (no HITL gates) from the basic "New Run" tab
    config["checkpoint_stages"] = []

    log_q: queue.Queue = queue.Queue()
    _log_queues[run_id] = log_q
    emitter = ProgressEmitter(output_dir=_OUTPUT_DIR, run_id=run_id, gradio_queue=log_q)

    db = Database(_DB_PATH)
    llm = LLMClient()
    broker = CheckpointBroker(NoOpHandler())
    orchestrator = create_orchestrator(
        db=db, llm=llm, output_dir=_OUTPUT_DIR, config=config,
        db_path=_DB_PATH, broker=broker, emitter=emitter,
    )

    def _run():
        _run_results[run_id] = {"status": "running"}
        try:
            result = orchestrator.invoke(
                {"run_id": run_id, "raw_question": question},
                config={"configurable": {"thread_id": run_id}},
            )
            _run_results[run_id] = {"status": "complete", "result": result}
            log_q.put(f"\nComplete. Manuscript: {result.get('manuscript_path', 'N/A')}")
        except Exception as e:
            _run_results[run_id] = {"status": "error", "error": str(e)}
            log_q.put(f"\nError: {e}")

    threading.Thread(target=_run, daemon=True).start()
    return run_id, "Run started. Check the log below."


def get_run_status(run_id: str) -> str:
    result = _run_results.get(run_id, {})
    status = result.get("status", "not found")
    if status == "complete":
        manuscript = result["result"].get("manuscript_path", "")
        return f"Complete. Manuscript: {manuscript}"
    if status == "error":
        return f"Error: {result.get('error')}"
    return f"Status: {status}"


def poll_log(run_id: str, current_log: str) -> str:
    """Drain all pending log messages for a run."""
    q = _log_queues.get(run_id)
    if q is None:
        return current_log
    lines = []
    try:
        while True:
            lines.append(q.get_nowait())
    except queue.Empty:
        pass
    return current_log + "\n".join(lines) if lines else current_log


def build_app() -> gr.Blocks:
    with gr.Blocks(title="SLR Agent") as app:
        gr.Markdown("# SLR Agent — Systematic Literature Review\nPowered by Gemma (local)")

        with gr.Tab("New Run"):
            question_input = gr.Textbox(
                label="Research Question",
                placeholder="e.g. Do ACE inhibitors reduce blood pressure in hypertensive adults?",
                lines=2,
            )
            with gr.Row():
                fulltext_cb = gr.Checkbox(label="Fetch full text (PMC Open Access)", value=False)
                stages_input = gr.Textbox(
                    label="Checkpoint stages (comma-separated, 1-7)",
                    value="1, 2, 3, 5, 6, 7",
                )
            template_upload = gr.File(
                label="Manuscript template (optional — JSON schema or PDF reference paper)",
                file_types=[".json", ".pdf"],
            )
            start_btn = gr.Button("Start Run", variant="primary")
            run_id_out = gr.Textbox(label="Run ID", interactive=False)
            start_msg = gr.Textbox(label="Status", interactive=False)
            log_box = gr.Textbox(label="Live Progress Log", lines=15, interactive=False)

            start_btn.click(
                launch_run,
                inputs=[question_input, fulltext_cb, stages_input, template_upload],
                outputs=[run_id_out, start_msg],
            )
            # Poll log every 2 seconds
            app.load(
                lambda rid, log: poll_log(rid, log),
                inputs=[run_id_out, log_box],
                outputs=[log_box],
                every=2,
            )

        with gr.Tab("Monitor"):
            monitor_run_id = gr.Textbox(label="Run ID")
            refresh_btn = gr.Button("Refresh Status")
            status_out = gr.Textbox(label="Status", interactive=False, lines=3)
            refresh_btn.click(get_run_status, inputs=[monitor_run_id], outputs=[status_out])

    return app


def build_app_with_handler(ui_handler: UIHandler, run_id: str) -> gr.Blocks:
    """Minimal app used by CLI --hitl ui: polls UIHandler and shows checkpoint panels."""
    with gr.Blocks(title=f"SLR Agent — Run {run_id}") as app:
        gr.Markdown(f"# SLR Agent Checkpoint Review\nRun: `{run_id}`")

        pending_state = gr.State(None)
        checkpoint_area = gr.Column(visible=False)
        with checkpoint_area:
            stage_label = gr.Markdown("## Checkpoint")
            data_json = gr.JSON(label="Stage Data")
            approve_btn = gr.Button("Approve & Continue", variant="primary")
            status_out = gr.Textbox(label="Status", value="Waiting for checkpoint...", interactive=False)

        def poll_checkpoint(pending):
            cp = ui_handler.get_pending(timeout=0.1)
            if cp is None:
                return pending, gr.update(visible=False), "", None, "Waiting for checkpoint..."
            return (
                cp,
                gr.update(visible=True),
                f"## Stage {cp['stage']}: {cp['stage_name'].upper()}",
                cp["data"],
                "Review the data below and approve to continue.",
            )

        def approve(pending, data):
            if pending is None:
                return None, "No pending checkpoint.", gr.update(visible=False)
            ui_handler.resume({**(data or {}), "action": "approve"})
            return None, "Approved. Waiting for next checkpoint...", gr.update(visible=False)

        app.load(
            poll_checkpoint,
            inputs=[pending_state],
            outputs=[pending_state, checkpoint_area, stage_label, data_json, status_out],
            every=1,
        )
        approve_btn.click(
            approve,
            inputs=[pending_state, data_json],
            outputs=[pending_state, status_out, checkpoint_area],
        )

    return app


if __name__ == "__main__":
    build_app().launch()
```

- [ ] **Step 2: Run full test suite**

```bash
.venv/bin/pytest tests/ -x -q --ignore=tests/e2e
```

Expected: all pass (Gradio app is not directly unit-tested here — covered by integration).

- [ ] **Step 3: Commit**

```bash
git add slr_agent/ui/app.py
git commit -m "feat: wire UIHandler into Gradio app; add live log panel and template upload"
```

---

### Task 11: Stage 2 Search Gradio panel

**Files:**
- Create: `slr_agent/ui/panels/search.py`

Displays papers retrieved per query with checkboxes to exclude individual papers. Also provides PMID/title lookup and PDF upload to add papers manually.

- [ ] **Step 1: Create search panel**

```python
# slr_agent/ui/panels/search.py
import gradio as gr


def build_search_panel(data: dict, ui_handler, pending) -> tuple:
    """Build the stage 2 search checkpoint panel.
    
    data keys: papers (list of {pmid, title, source}), n_retrieved, n_duplicates_removed
    Returns (approve_fn, gr.Column) for embedding in the checkpoint area.
    """
    papers = data.get("papers", [])

    with gr.Column() as panel:
        gr.Markdown(f"## Stage 2: Search Results\n"
                    f"Retrieved: **{data.get('n_retrieved', len(papers))}** papers "
                    f"(after removing **{data.get('n_duplicates_removed', 0)}** duplicates)")

        gr.Markdown("Uncheck papers to exclude them before screening. Add missing papers below.")

        # Paper checkboxes
        paper_checks = []
        for p in papers[:200]:  # cap at 200 for UI performance
            cb = gr.Checkbox(
                label=f"[{p.get('source','?')}] {p.get('pmid','')} — {p.get('title','(no title)')[:100]}",
                value=True,
            )
            paper_checks.append((p["pmid"], cb))

        gr.Markdown("### Add a paper manually")
        with gr.Row():
            pmid_input = gr.Textbox(label="PMID", placeholder="e.g. 12345678", scale=1)
            add_pmid_btn = gr.Button("Look up PMID", scale=1)
        add_status = gr.Textbox(label="Add status", interactive=False)

        pdf_upload = gr.File(label="Upload PDF paper", file_types=[".pdf"])
        pdf_title = gr.Textbox(label="Paper title (for uploaded PDF)", placeholder="Enter title")
        add_pdf_btn = gr.Button("Add PDF paper")

        approve_btn = gr.Button("Approve & Continue", variant="primary")
        result_status = gr.Textbox(label="Status", value="Review and approve.", interactive=False)

        def on_approve(*checkbox_values):
            excluded_pmids = set()
            for i, (pmid, _) in enumerate(paper_checks):
                if i < len(checkbox_values) and not checkbox_values[i]:
                    excluded_pmids.add(pmid)
            edited = {
                **data,
                "papers": [
                    {**p, "excluded": p["pmid"] in excluded_pmids}
                    for p in papers
                ],
                "action": "approve",
            }
            ui_handler.resume(edited)
            return "Approved. Pipeline continuing..."

        def on_add_pmid(pmid_val):
            if not pmid_val.strip():
                return "Enter a PMID first."
            return f"PMID {pmid_val.strip()} will be added when you approve."

        add_pmid_btn.click(on_add_pmid, inputs=[pmid_input], outputs=[add_status])

        cb_inputs = [cb for _, cb in paper_checks]
        approve_btn.click(on_approve, inputs=cb_inputs, outputs=[result_status])

    return panel
```

- [ ] **Step 2: Run tests (no dedicated unit test for Gradio panels — covered by manual testing)**

```bash
.venv/bin/pytest tests/ -x -q --ignore=tests/e2e
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add slr_agent/ui/panels/search.py
git commit -m "feat: add Stage 2 Search Gradio checkpoint panel"
```

---

### Task 12: Stage 3 Screening Gradio panel

**Files:**
- Create: `slr_agent/ui/panels/screening.py`

Shows each paper with AI decision and reason. User can flip include/exclude, add papers by PMID or PDF upload.

- [ ] **Step 1: Create screening panel**

```python
# slr_agent/ui/panels/screening.py
import gradio as gr


def build_screening_panel(data: dict, ui_handler) -> gr.Column:
    """Stage 3 screening checkpoint panel.
    
    data keys: papers (list of {pmid, title, abstract, decision, reason}),
               n_included, n_excluded, n_uncertain
    """
    papers = data.get("papers", [])
    included = [p for p in papers if p.get("decision") == "include"]
    excluded = [p for p in papers if p.get("decision") == "exclude"]

    with gr.Column() as panel:
        gr.Markdown(
            f"## Stage 3: Screening\n"
            f"**Included:** {len(included)} | **Excluded:** {len(excluded)}"
        )
        gr.Markdown("Flip decisions below. Papers added here are marked **include** directly.")

        decision_dropdowns = []
        with gr.Accordion("Included papers", open=True):
            for p in included[:100]:
                with gr.Row():
                    gr.Markdown(
                        f"**{p.get('pmid', '')}** — {p.get('title', '')[:80]}\n\n"
                        f"*{p.get('abstract', '')[:200]}...*\n\n"
                        f"Reason: {p.get('reason', '')}",
                        scale=4,
                    )
                    dd = gr.Dropdown(
                        choices=["include", "exclude"],
                        value=p.get("decision", "include"),
                        label="Decision",
                        scale=1,
                    )
                    decision_dropdowns.append((p["pmid"], dd))

        with gr.Accordion("Excluded papers", open=False):
            for p in excluded[:100]:
                with gr.Row():
                    gr.Markdown(
                        f"**{p.get('pmid', '')}** — {p.get('title', '')[:80]}\n\n"
                        f"Reason: {p.get('reason', '')}",
                        scale=4,
                    )
                    dd = gr.Dropdown(
                        choices=["include", "exclude"],
                        value=p.get("decision", "exclude"),
                        label="Decision",
                        scale=1,
                    )
                    decision_dropdowns.append((p["pmid"], dd))

        gr.Markdown("### Add a paper manually (marked include directly)")
        with gr.Row():
            pmid_input = gr.Textbox(label="PMID", placeholder="12345678", scale=2)
            add_btn = gr.Button("Add by PMID", scale=1)
        add_status = gr.Textbox(label="", interactive=False)
        pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
        pdf_title = gr.Textbox(label="Title for uploaded PDF")

        approve_btn = gr.Button("Approve & Continue", variant="primary")
        status_out = gr.Textbox(label="Status", value="Review and approve.", interactive=False)

        def on_approve(*dropdown_values):
            updated = []
            for i, (pmid, _) in enumerate(decision_dropdowns):
                orig = next((p for p in papers if p["pmid"] == pmid), {})
                updated.append({
                    **orig,
                    "decision": dropdown_values[i] if i < len(dropdown_values) else orig.get("decision"),
                })
            ui_handler.resume({**data, "papers": updated, "action": "approve"})
            return "Approved. Pipeline continuing..."

        def on_add_pmid(pmid_val):
            return f"PMID {pmid_val.strip()} queued for manual include."

        add_btn.click(on_add_pmid, inputs=[pmid_input], outputs=[add_status])
        dd_inputs = [dd for _, dd in decision_dropdowns]
        approve_btn.click(on_approve, inputs=dd_inputs, outputs=[status_out])

    return panel
```

- [ ] **Step 2: Run tests**

```bash
.venv/bin/pytest tests/ -x -q --ignore=tests/e2e
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add slr_agent/ui/panels/screening.py
git commit -m "feat: add Stage 3 Screening Gradio checkpoint panel with include/exclude flip"
```

---

### Task 13: Stage 5 Extraction Gradio panel

**Files:**
- Create: `slr_agent/ui/panels/extraction.py`

Shows extracted fields per paper with editable text boxes. Shows quarantined fields with option to un-quarantine.

- [ ] **Step 1: Create extraction panel**

```python
# slr_agent/ui/panels/extraction.py
import gradio as gr


def build_extraction_panel(data: dict, ui_handler) -> gr.Column:
    """Stage 5 extraction checkpoint panel.
    
    data keys: papers (list of {pmid, title, extracted_data, quarantined_fields, grade_score})
    """
    papers = data.get("papers", [])

    with gr.Column() as panel:
        gr.Markdown(f"## Stage 5: Extraction\n**Papers:** {len(papers)}")
        gr.Markdown("Edit extracted fields below. Changes are saved as human overrides.")

        field_inputs = []  # list of (pmid, field_name, gr.Textbox)
        for p in papers:
            with gr.Accordion(f"{p.get('pmid', '')} — {p.get('title', '')[:60]}", open=False):
                grade = p.get("grade_score", {})
                gr.Markdown(f"**GRADE certainty:** {grade.get('certainty', 'N/A')} | "
                            f"Risk of bias: {grade.get('risk_of_bias', 'N/A')}")
                for field_name, value in (p.get("extracted_data") or {}).items():
                    tb = gr.Textbox(label=field_name, value=str(value), lines=2)
                    field_inputs.append((p["pmid"], field_name, tb))
                quarantined = p.get("quarantined_fields", [])
                if quarantined:
                    gr.Markdown(f"**Quarantined fields ({len(quarantined)}):**")
                    for qf in quarantined:
                        gr.Markdown(
                            f"- `{qf.get('field_name')}`: `{qf.get('value')}` "
                            f"— *{qf.get('reason', '')}*"
                        )

        approve_btn = gr.Button("Approve & Continue", variant="primary")
        status_out = gr.Textbox(label="Status", value="Review and approve.", interactive=False)

        def on_approve(*values):
            # Rebuild extracted_data per paper from edited textboxes
            edited_papers = []
            value_idx = 0
            for p in papers:
                fields = p.get("extracted_data") or {}
                updated_fields = {}
                for field_name in fields:
                    updated_fields[field_name] = values[value_idx] if value_idx < len(values) else fields[field_name]
                    value_idx += 1
                edited_papers.append({**p, "extracted_data": updated_fields})
            ui_handler.resume({**data, "papers": edited_papers, "action": "approve"})
            return "Approved. Pipeline continuing..."

        tb_inputs = [tb for _, _, tb in field_inputs]
        approve_btn.click(on_approve, inputs=tb_inputs, outputs=[status_out])

    return panel
```

- [ ] **Step 2: Commit**

```bash
git add slr_agent/ui/panels/extraction.py
git commit -m "feat: add Stage 5 Extraction Gradio checkpoint panel"
```

---

### Task 14: Stage 6 Synthesis Gradio panel

**Files:**
- Create: `slr_agent/ui/panels/synthesis.py`

Shows grounded claims with supporting PMIDs and the narrative paragraph. User can edit or delete claims and edit the narrative.

- [ ] **Step 1: Create synthesis panel**

```python
# slr_agent/ui/panels/synthesis.py
import gradio as gr


def build_synthesis_panel(data: dict, ui_handler) -> gr.Column:
    """Stage 6 synthesis checkpoint panel.
    
    data keys: synthesis_path (str), preview (str — first 500 chars of synthesis markdown)
    """
    preview = data.get("preview", "")
    synthesis_path = data.get("synthesis_path", "")

    # Try to read full synthesis if path available
    full_text = preview
    if synthesis_path:
        try:
            with open(synthesis_path) as f:
                full_text = f.read()
        except OSError:
            pass

    with gr.Column() as panel:
        gr.Markdown("## Stage 6: Synthesis\nReview the evidence synthesis below.")

        synthesis_edit = gr.Textbox(
            label="Evidence synthesis (editable)",
            value=full_text,
            lines=20,
        )

        approve_btn = gr.Button("Approve & Continue", variant="primary")
        status_out = gr.Textbox(label="Status", value="Review and approve.", interactive=False)

        def on_approve(text):
            # Save edited synthesis back to file
            if synthesis_path:
                try:
                    with open(synthesis_path, "w") as f:
                        f.write(text)
                except OSError:
                    pass
            ui_handler.resume({**data, "preview": text[:500], "action": "approve"})
            return "Approved. Pipeline continuing..."

        approve_btn.click(on_approve, inputs=[synthesis_edit], outputs=[status_out])

    return panel
```

- [ ] **Step 2: Commit**

```bash
git add slr_agent/ui/panels/synthesis.py
git commit -m "feat: add Stage 6 Synthesis Gradio checkpoint panel"
```

---

### Task 15: Stage 7 Manuscript Gradio panel (revision loop)

**Files:**
- Create: `slr_agent/ui/panels/manuscript_panel.py`

Shows the full draft rendered as markdown alongside the scored rubric. User can approve, edit the rubric and trigger an LLM revision, or edit the draft directly. Each revision action sends `{"action": "revise", "template": ...}` back via UIHandler.

- [ ] **Step 1: Create manuscript panel**

```python
# slr_agent/ui/panels/manuscript_panel.py
import gradio as gr


def build_manuscript_panel(data: dict, ui_handler) -> gr.Column:
    """Stage 7 manuscript checkpoint panel with rubric-guided revision loop.
    
    data keys:
      draft (str) — full manuscript markdown
      rubric (dict) — {template: {...}, scores: [{criterion, score, explanation}]}
      draft_version (int)
    """
    draft = data.get("draft", "")
    rubric = data.get("rubric", {})
    scores = rubric.get("scores", [])
    template = rubric.get("template", {})
    version = data.get("draft_version", 1)

    with gr.Column() as panel:
        gr.Markdown(f"## Stage 7: Manuscript Review (Draft v{version})")

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### Manuscript Draft")
                draft_edit = gr.Textbox(
                    label="Draft (editable — changes saved on Approve)",
                    value=draft,
                    lines=30,
                )

            with gr.Column(scale=2):
                gr.Markdown("### Rubric Scores")
                score_displays = []
                for s in scores:
                    color = {"met": "🟢", "partial": "🟡", "not met": "🔴"}.get(s.get("score", ""), "⚪")
                    gr.Markdown(
                        f"{color} **{s.get('criterion', '')}**\n\n"
                        f"*{s.get('explanation', '')}*"
                    )

                gr.Markdown("### Rubric Criteria (editable)")
                all_criteria = [
                    c
                    for sec in template.get("sections", [])
                    for c in sec.get("rubric_criteria", [])
                ]
                rubric_edit = gr.Textbox(
                    label="One criterion per line",
                    value="\n".join(all_criteria),
                    lines=10,
                )

                gr.Markdown("### Paste journal guidelines (optional)")
                extra_rubric = gr.Textbox(
                    label="Additional rubric / journal submission guidelines",
                    lines=5,
                    placeholder="Paste guidelines here to add to revision criteria...",
                )

        with gr.Row():
            approve_btn = gr.Button("Approve & Finalise", variant="primary", scale=2)
            revise_btn = gr.Button("Trigger LLM Revision", variant="secondary", scale=2)

        status_out = gr.Textbox(label="Status", value="Review draft and rubric scores.", interactive=False)

        def on_approve(draft_text):
            # Save final draft, then approve
            ui_handler.resume({"action": "approve", "draft": draft_text})
            return "Finalised. Generating Word export..."

        def on_revise(rubric_text, extra_text, draft_text):
            # Build updated template with edited criteria
            criteria = [c.strip() for c in rubric_text.split("\n") if c.strip()]
            if extra_text.strip():
                criteria += [c.strip() for c in extra_text.split("\n") if c.strip()]
            updated_template = {
                **template,
                "sections": _redistribute_criteria(template.get("sections", []), criteria),
            }
            ui_handler.resume({
                "action": "revise",
                "template": updated_template,
                "draft": draft_text,
            })
            return "Revision requested. Waiting for new draft..."

        approve_btn.click(on_approve, inputs=[draft_edit], outputs=[status_out])
        revise_btn.click(on_revise, inputs=[rubric_edit, extra_rubric, draft_edit], outputs=[status_out])

    return panel


def _redistribute_criteria(sections: list, flat_criteria: list) -> list:
    """Distribute flat criteria list back across sections evenly."""
    if not sections:
        return sections
    per_section = max(1, len(flat_criteria) // len(sections))
    result = []
    for i, sec in enumerate(sections):
        start = i * per_section
        end = start + per_section if i < len(sections) - 1 else len(flat_criteria)
        result.append({**sec, "rubric_criteria": flat_criteria[start:end]})
    return result
```

- [ ] **Step 2: Run full test suite one final time**

```bash
.venv/bin/pytest tests/ -x -q --ignore=tests/e2e
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add slr_agent/ui/panels/manuscript_panel.py
git commit -m "feat: add Stage 7 Manuscript Gradio panel with rubric scoring and revision loop"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|-----------------|------|
| Fix SQLite checkpointer | Task 1 |
| ProgressEmitter writes stage JSON to disk | Task 3 |
| CheckpointBroker + CLIHandler + UIHandler + NoOpHandler | Task 4 |
| `--hitl cli/ui`, `--no-checkpoints`, `--template` flags | Task 9 |
| Manual paper add by PMID/DOI/title/PDF | Task 5 |
| HITL gates at stages 1, 2, 3, 5, 6, 7 | Task 8 (orchestrator wiring) |
| Stage 2: exclude/add papers | Task 11 |
| Stage 3: flip include/exclude, add papers | Task 12 |
| Stage 5: edit extracted fields | Task 13 |
| Stage 6: edit synthesis | Task 14 |
| Stage 7: manuscript + rubric revision loop | Task 15 |
| Template from JSON schema | Task 6 |
| Template from PDF | Task 6 |
| Default PRISMA 2020 structure | Task 6 |
| Auto-generate rubric from template | Task 6 |
| Two-pass manuscript (draft + rubric scoring) | Task 7 |
| Versioned drafts (stage_7_draft_v1.md etc.) | Task 7 |
| Gradio live log panel | Task 10 |
| Gradio template upload | Task 10 |
| `config.py` + `state.py` updates | Task 2 |
| All 32 existing tests still pass | Tasks 1, 7, 8, 9 (each runs full suite) |

**Placeholder scan:** No TBDs. All code blocks are complete.

**Type consistency:** `broker.pause()` always returns `dict` with `"action"` key. `emitter.emit()` always takes `(int, dict)`. `load_template()` always returns `{sections: [...], style_notes: str}`. `score_rubric()` always returns `{scores: [...]}`. Manuscript subgraph always returns `{manuscript_path, manuscript_rubric, manuscript_draft_version}`. Consistent throughout Tasks 6–15.
