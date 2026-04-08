# slr_agent/ui/app.py
import json
import queue
import re
import threading
import uuid
import gradio as gr
from slr_agent.broker import CheckpointBroker, NoOpHandler, UIHandler
from slr_agent.db import Database
from slr_agent.llm import LLMClient
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
        template_path=template_path,
        hitl_mode="ui",
    )


def launch_run(question: str, fetch_fulltext: bool, checkpoint_stages_str: str, template_file):
    """Start a pipeline run in a background thread (no HITL — auto-approve)."""
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
        gr.Markdown("# SLR Agent — Systematic Literature Review\nPowered by Gemma 4 via Ollama")

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
            gr.Timer(2).tick(
                lambda rid, log: poll_log(rid, log),
                inputs=[run_id_out, log_box],
                outputs=[log_box],
            )

        with gr.Tab("Monitor"):
            monitor_run_id = gr.Textbox(label="Run ID")
            refresh_btn = gr.Button("Refresh Status")
            status_out = gr.Textbox(label="Status", interactive=False, lines=3)
            refresh_btn.click(get_run_status, inputs=[monitor_run_id], outputs=[status_out])

    return app


def _search_papers_to_df_data(papers: list[dict]) -> list[list]:
    """Convert search result papers to rows for the search review dataframe."""
    return [
        [p.get("pmid", ""), (p.get("title") or "")[:100], p.get("source", ""), False]
        for p in papers
    ]


def _format_criterion_scores(criterion_scores: list[dict]) -> str:
    """Format criterion scores as a readable text block for the UI detail panel."""
    if not criterion_scores:
        return ""
    icons = {"yes": "✓", "no": "✗", "unclear": "?"}
    type_labels = {"inclusion": "IN", "exclusion": "EX", "study_design": "SD"}
    lines = []
    for s in criterion_scores:
        icon = icons.get(s.get("met", ""), "?")
        label = type_labels.get(s.get("type", ""), "  ")
        criterion = s.get("criterion", "")
        note = s.get("note", "")
        lines.append(f"[{label}] {icon}  {criterion}")
        if note:
            lines.append(f"       → {note}")
    return "\n".join(lines)


def _screening_filter(papers: list[dict], filter_val: str = "All") -> tuple[list[list], list[int]]:
    """Return (df_rows, original_indices) for the given filter."""
    indices = [
        i for i, p in enumerate(papers)
        if filter_val == "All" or p.get("decision", "uncertain") == filter_val.lower()
    ]
    rows = [
        [
            papers[i].get("pmid", ""),
            (papers[i].get("title") or "")[:80],
            papers[i].get("decision", "uncertain"),
            (papers[i].get("reason") or "")[:120],
        ]
        for i in indices
    ]
    return rows, indices


def _papers_to_df_data(papers: list[dict]) -> list[list]:
    """Convert papers list to rows for the extraction dataframe."""
    rows = []
    for p in papers:
        grade = (p.get("grade_score") or {}).get("certainty", "")
        n_quarantined = len(p.get("quarantined_fields") or [])
        rows.append([
            p.get("pmid", ""),
            (p.get("title") or "")[:80],
            grade,
            n_quarantined,
            False,   # exclude
            False,   # llm_ground
        ])
    return rows


_SECTION_REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}


def _replace_section(draft: str, section_name: str, new_text: str) -> str:
    """Replace the body of a ## section in a markdown draft."""
    pattern = re.compile(
        rf"(## {re.escape(section_name.strip())}\n)(.*?)(?=\n## |\Z)",
        re.DOTALL,
    )
    if not pattern.search(draft):
        return draft  # section not found — return unchanged
    return pattern.sub(rf"\g<1>{new_text.strip()}\n", draft)


def build_app_with_handler(ui_handler: UIHandler, run_id: str, llm=None) -> gr.Blocks:
    """Minimal app used by CLI --hitl ui: polls UIHandler and shows checkpoint panels."""
    _llm = llm or LLMClient()
    _css = """
    /* Prevent checkpoint panel groups from stretching to fill viewport height */
    .checkpoint-panel { height: auto !important; min-height: unset !important; }
    """
    with gr.Blocks(title=f"SLR Agent — Run {run_id}", css=_css) as app:
        gr.Markdown(f"# SLR Agent Checkpoint Review\nRun: `{run_id}`")

        pending_state = gr.State(None)
        papers_state = gr.State([])               # full paper dicts for the extraction detail view
        screening_papers_state = gr.State([])     # full paper dicts for the screening review panel
        screening_filter_indices = gr.State([])   # indices into screening_papers_state (current filter)
        selected_paper_idx = gr.State(None)       # index in screening_papers_state of selected paper
        status_out = gr.Textbox(label="Status", value="Waiting for pipeline checkpoint...", interactive=False)

        # Generic panel — used for all stages except 5
        with gr.Group(visible=False) as checkpoint_area:
            stage_label = gr.Markdown("## Checkpoint")
            data_code = gr.Code(label="Stage Data (editable JSON)", language="json", interactive=True)
            approve_btn = gr.Button("Approve & Continue", variant="primary")

        # Stage 2 search panel — table of retrieved papers with Exclude checkbox + manual add
        with gr.Group(visible=False, elem_classes="checkpoint-panel") as search_panel:
            gr.Markdown("## Stage 2: Search Results")
            gr.Markdown(
                "Check **Exclude** to remove a paper before screening. "
                "To add papers by PMID, enter them in the field below."
            )
            search_df = gr.Dataframe(
                headers=["PMID", "Title", "Source", "Exclude"],
                datatype=["str", "str", "str", "bool"],
                column_count=(4, "fixed"),
                interactive=True,
                wrap=True,
                max_height=250,
            )
            add_pmids_input = gr.Textbox(
                label="Add papers by PMID (comma-separated)",
                placeholder="e.g. 12345678, 87654321",
            )
            approve_btn_search = gr.Button("Approve & Continue", variant="primary")

        # Stage 3 screening panel — filter tabs, row select, per-paper decision buttons
        with gr.Group(visible=False, elem_classes="checkpoint-panel") as screening_panel:
            gr.Markdown("## Stage 3: Screening Review")
            gr.Markdown(
                "Click a row to expand the abstract below. "
                "Use the buttons to override the AI decision for that paper. "
                "Click **Approve & Continue** when done."
            )
            screening_filter_radio = gr.Radio(
                choices=["All", "Include", "Uncertain", "Exclude"],
                value="All",
                label="Filter by decision",
            )
            screening_df = gr.Dataframe(
                headers=["PMID", "Title", "Decision", "Reason"],
                datatype=["str", "str", "str", "str"],
                column_count=(4, "fixed"),
                interactive=False,
                wrap=True,
                max_height=300,
            )
            with gr.Group():
                screening_title = gr.Textbox(label="Title", interactive=False)
                screening_abstract = gr.Textbox(label="Abstract", lines=5, interactive=False)
                screening_reason_box = gr.Textbox(label="AI Reason", interactive=False)
                screening_criterion_scores = gr.Textbox(
                    label="Criterion Scores  [IN] = inclusion  [EX] = exclusion  [SD] = study design  ✓ met  ✗ not met  ? unclear",
                    lines=8,
                    interactive=False,
                )
                with gr.Row():
                    include_btn = gr.Button("✓ Include", variant="primary")
                    uncertain_btn = gr.Button("? Uncertain", variant="secondary")
                    exclude_btn = gr.Button("✗ Exclude", variant="stop")
            approve_btn_screening = gr.Button("Approve & Continue", variant="primary")

        # Stage 5 extraction panel — checkboxes for exclude / LLM ground per paper
        with gr.Group(visible=False, elem_classes="checkpoint-panel") as extraction_panel:
            gr.Markdown("## Stage 5: Extraction Review")
            gr.Markdown(
                "Click any row to see its extracted and quarantined fields below. "
                "Check **Exclude** to remove a paper from synthesis. "
                "Check **LLM ground** to re-verify quarantined fields using the LLM "
                "(use when fuzzy matching incorrectly rejected a valid extraction)."
            )
            papers_df = gr.Dataframe(
                headers=["PMID", "Title", "GRADE", "Quarantined fields", "Exclude", "LLM ground"],
                datatype=["str", "str", "str", "number", "bool", "bool"],
                column_count=(6, "fixed"),
                interactive=True,
                wrap=True,
                max_height=250,
            )
            with gr.Row():
                with gr.Column():
                    extracted_detail = gr.Code(
                        label="Extracted fields (selected paper)",
                        language="json",
                        interactive=False,
                        value="",
                        max_lines=12,
                    )
                with gr.Column():
                    quarantined_detail = gr.Code(
                        label="Quarantined fields (selected paper)",
                        language="json",
                        interactive=False,
                        value="",
                        max_lines=12,
                    )
            approve_btn_extract = gr.Button("Approve & Continue", variant="primary")

        # Stage 7 manuscript panel — editable draft + section rewrite + rubric + approve/revise
        with gr.Group(visible=False, elem_classes="checkpoint-panel") as manuscript_panel:
            gr.Markdown("## Stage 7: Manuscript Review")
            gr.Markdown(
                "Edit the draft directly, or use **Rewrite Section** to have the LLM rewrite "
                "a specific section with your instructions. Click **Approve** to finalise "
                "(your edits are saved), or **Full Revise** to regenerate the whole draft."
            )
            draft_version_label = gr.Markdown("Draft v1")
            draft_display = gr.Code(
                label="Manuscript Draft (editable)",
                language="markdown",
                interactive=True,
                value="",
                max_lines=30,
            )
            with gr.Accordion("Rewrite a Section (LLM-assisted)", open=False):
                gr.Markdown(
                    "Enter the section name exactly as it appears in the draft "
                    "(e.g. `Methods`, `Results`, `Discussion`) and an instruction for the LLM."
                )
                with gr.Row():
                    section_name_input = gr.Textbox(
                        label="Section name",
                        placeholder="e.g. Methods",
                        scale=1,
                    )
                    section_instruction_input = gr.Textbox(
                        label="Instruction",
                        placeholder="e.g. Add more detail about the search strategy and date range",
                        scale=3,
                    )
                rewrite_section_btn = gr.Button("Rewrite Section", variant="secondary")
                rewrite_status = gr.Markdown("")
            rubric_display = gr.Code(
                label="Rubric Scores",
                language="json",
                interactive=False,
                value="",
                max_lines=15,
            )
            with gr.Row():
                approve_btn_manuscript = gr.Button("Approve & Finalise", variant="primary")
                revise_btn_manuscript = gr.Button("Full Revise (re-run LLM)", variant="secondary")

        # ── poll ────────────────────────────────────────────────────────────────

        def poll_checkpoint(pending):
            if pending is not None:
                return (gr.update(),) * 19
            cp = ui_handler.get_pending(timeout=0.1)
            if cp is None:
                return (
                    pending, [], "Waiting for pipeline checkpoint...",
                    gr.update(visible=False), "", "",
                    gr.update(visible=False), gr.update(), gr.update(),
                    gr.update(visible=False), gr.update(), gr.update(),
                    gr.update(visible=False), gr.update(), gr.update(value=""),
                    gr.update(visible=False), [], gr.update(value=[]), [],
                )
            is_search = cp["stage"] == 2
            is_screening = cp["stage"] == 3 and cp["stage_name"] == "screening"
            is_extraction = cp["stage"] == 5
            is_manuscript = cp["stage"] == 7
            is_generic = not is_search and not is_screening and not is_extraction and not is_manuscript
            search_papers = cp["data"].get("papers", []) if is_search else []
            screening_papers = cp["data"].get("papers", []) if is_screening else []
            extract_papers = cp["data"].get("papers", []) if is_extraction else []
            draft = cp["data"].get("draft", "") if is_manuscript else ""
            rubric = cp["data"].get("rubric", {}) if is_manuscript else {}
            version = cp["data"].get("draft_version", 1) if is_manuscript else 1
            s_rows, s_indices = _screening_filter(screening_papers, "All")
            return (
                cp,
                extract_papers,
                "Review below, then click Approve.",
                gr.update(visible=is_generic),
                f"## Stage {cp['stage']}: {cp['stage_name'].upper()}",
                json.dumps(cp["data"], indent=2) if is_generic else "",
                gr.update(visible=is_extraction),
                gr.update(value=_papers_to_df_data(extract_papers)),
                gr.update(),
                gr.update(visible=is_manuscript),
                gr.update(value=draft, label=f"Manuscript Draft v{version} (read-only preview)"),
                gr.update(value=json.dumps(rubric, indent=2)),
                gr.update(visible=is_search),
                gr.update(value=_search_papers_to_df_data(search_papers)),
                gr.update(value=""),
                gr.update(visible=is_screening),
                screening_papers,
                gr.update(value=s_rows),
                s_indices,
            )

        gr.Timer(1).tick(
            poll_checkpoint,
            inputs=[pending_state],
            outputs=[
                pending_state, papers_state, status_out,
                checkpoint_area, stage_label, data_code,
                extraction_panel, papers_df, approve_btn_extract,
                manuscript_panel, draft_display, rubric_display,
                search_panel, search_df, add_pmids_input,
                screening_panel, screening_papers_state, screening_df, screening_filter_indices,
            ],
        )

        # ── row select → detail panel ────────────────────────────────────────────

        def show_paper_detail(papers, evt: gr.SelectData):
            idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            if not papers or idx >= len(papers):
                return "", ""
            p = papers[idx]
            return (
                json.dumps(p.get("extracted_data") or {}, indent=2),
                json.dumps(p.get("quarantined_fields") or [], indent=2),
            )

        papers_df.select(
            show_paper_detail,
            inputs=[papers_state],
            outputs=[extracted_detail, quarantined_detail],
        )

        # ── generic approve ──────────────────────────────────────────────────────

        def approve(pending, data_str):
            if pending is None:
                return None, "No pending checkpoint.", gr.update(visible=False)
            try:
                edited = json.loads(data_str) if data_str else {}
            except json.JSONDecodeError:
                return pending, "Invalid JSON — fix the data and try again.", gr.update(visible=True)
            ui_handler.resume({**edited, "action": "approve"})
            return None, "Approved. Waiting for next checkpoint...", gr.update(visible=False)

        approve_btn.click(
            approve,
            inputs=[pending_state, data_code],
            outputs=[pending_state, status_out, checkpoint_area],
        )

        # ── search approve ───────────────────────────────────────────────────────

        def approve_search(pending, df_value, add_pmids_str):
            if pending is None:
                return None, "No pending checkpoint.", gr.update(visible=False), gr.update(value="")
            rows = df_value.get("data", []) if isinstance(df_value, dict) else []
            papers_orig = pending["data"].get("papers", [])
            papers_out = []
            for i, row in enumerate(rows):
                pmid = row[0] if len(row) > 0 else (papers_orig[i]["pmid"] if i < len(papers_orig) else "")
                exclude = bool(row[3]) if len(row) > 3 else False
                entry = {"pmid": pmid, "excluded": exclude}
                if i < len(papers_orig):
                    entry["title"] = papers_orig[i].get("title", "")
                papers_out.append(entry)
            for pmid in [p.strip() for p in (add_pmids_str or "").split(",") if p.strip()]:
                papers_out.append({"pmid": pmid, "title": "", "excluded": False, "manual_add": True})
            ui_handler.resume({"papers": papers_out, "action": "approve"})
            return None, "Approved. Waiting for next checkpoint...", gr.update(visible=False), gr.update(value="")

        approve_btn_search.click(
            approve_search,
            inputs=[pending_state, search_df, add_pmids_input],
            outputs=[pending_state, status_out, search_panel, add_pmids_input],
        )

        # ── extraction approve ───────────────────────────────────────────────────

        def approve_extraction(pending, df_value):
            if pending is None:
                return None, "No pending checkpoint.", gr.update(visible=False)
            # df_value is a dict {"headers": [...], "data": [[...]]} in Gradio 6
            rows = df_value.get("data", []) if isinstance(df_value, dict) else []
            papers_orig = pending["data"].get("papers", [])
            papers_out = []
            for i, row in enumerate(rows):
                pmid = row[0] if len(row) > 0 else (papers_orig[i]["pmid"] if i < len(papers_orig) else "")
                exclude = bool(row[4]) if len(row) > 4 else False
                llm_ground = bool(row[5]) if len(row) > 5 else False
                entry = {"pmid": pmid, "exclude": exclude, "llm_ground": llm_ground}
                if i < len(papers_orig):
                    entry["extracted_data"] = papers_orig[i].get("extracted_data", {})
                papers_out.append(entry)
            ui_handler.resume({"papers": papers_out, "action": "approve"})
            return None, "Approved. Waiting for next checkpoint...", gr.update(visible=False)

        approve_btn_extract.click(
            approve_extraction,
            inputs=[pending_state, papers_df],
            outputs=[pending_state, status_out, extraction_panel],
        )

        # ── screening filter, row select, decision buttons, approve ─────────────

        def on_screening_filter(papers, filter_val):
            rows, indices = _screening_filter(papers, filter_val)
            return gr.update(value=rows), indices

        screening_filter_radio.change(
            on_screening_filter,
            inputs=[screening_papers_state, screening_filter_radio],
            outputs=[screening_df, screening_filter_indices],
        )

        def on_screening_row_select(papers, filter_indices, evt: gr.SelectData):
            row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            if not filter_indices or row_idx >= len(filter_indices):
                return None, "", "", "", ""
            paper_idx = filter_indices[row_idx]
            p = papers[paper_idx]
            scores_text = _format_criterion_scores(p.get("criterion_scores") or [])
            return paper_idx, p.get("title", ""), p.get("abstract", ""), p.get("reason", ""), scores_text

        screening_df.select(
            on_screening_row_select,
            inputs=[screening_papers_state, screening_filter_indices],
            outputs=[selected_paper_idx, screening_title, screening_abstract, screening_reason_box, screening_criterion_scores],
        )

        def _set_decision(papers, filter_indices, selected_idx, filter_val, new_decision):
            if selected_idx is None or selected_idx >= len(papers):
                return papers, gr.update(), filter_indices
            papers = list(papers)
            papers[selected_idx] = {**papers[selected_idx], "decision": new_decision}
            rows, indices = _screening_filter(papers, filter_val)
            return papers, gr.update(value=rows), indices

        include_btn.click(
            lambda p, fi, si, fv: _set_decision(p, fi, si, fv, "include"),
            inputs=[screening_papers_state, screening_filter_indices, selected_paper_idx, screening_filter_radio],
            outputs=[screening_papers_state, screening_df, screening_filter_indices],
        )
        uncertain_btn.click(
            lambda p, fi, si, fv: _set_decision(p, fi, si, fv, "uncertain"),
            inputs=[screening_papers_state, screening_filter_indices, selected_paper_idx, screening_filter_radio],
            outputs=[screening_papers_state, screening_df, screening_filter_indices],
        )
        exclude_btn.click(
            lambda p, fi, si, fv: _set_decision(p, fi, si, fv, "exclude"),
            inputs=[screening_papers_state, screening_filter_indices, selected_paper_idx, screening_filter_radio],
            outputs=[screening_papers_state, screening_df, screening_filter_indices],
        )

        def approve_screening(pending, screening_papers):
            if pending is None:
                return None, "No pending checkpoint.", gr.update(visible=False)
            papers_out = [
                {"pmid": p["pmid"], "decision": p.get("decision", "uncertain"),
                 "reason": p.get("reason", "")}
                for p in screening_papers
            ]
            ui_handler.resume({"papers": papers_out, "action": "approve"})
            return None, "Approved. Waiting for next checkpoint...", gr.update(visible=False)

        approve_btn_screening.click(
            approve_screening,
            inputs=[pending_state, screening_papers_state],
            outputs=[pending_state, status_out, screening_panel],
        )

        # ── manuscript section rewrite ───────────────────────────────────────────

        def rewrite_section(pending, current_draft, section_name, instruction):
            if pending is None:
                return current_draft, "No pending checkpoint."
            if not section_name.strip():
                return current_draft, "Enter a section name."
            if not instruction.strip():
                return current_draft, "Enter an instruction for the LLM."
            # Extract current section content to include in the prompt
            pattern = re.compile(
                rf"## {re.escape(section_name.strip())}\n(.*?)(?=\n## |\Z)",
                re.DOTALL,
            )
            m = pattern.search(current_draft)
            if not m:
                return current_draft, f"Section '{section_name}' not found in draft."
            current_section_text = m.group(1).strip()
            try:
                result = _llm.chat([{
                    "role": "user",
                    "content": (
                        f"Rewrite the '{section_name.strip()}' section of this systematic review "
                        f"manuscript according to the instruction below.\n\n"
                        f"Current section text:\n{current_section_text}\n\n"
                        f"Instruction: {instruction.strip()}\n\n"
                        "Return JSON with field 'text' containing the rewritten section body "
                        "(markdown only, no section header — it will be added automatically)."
                    ),
                }], schema=_SECTION_REWRITE_SCHEMA)
                new_text = result.get("text", "").strip()
                if not new_text:
                    return current_draft, "LLM returned empty text — try again."
                updated_draft = _replace_section(current_draft, section_name.strip(), new_text)
                return updated_draft, f"Section '{section_name.strip()}' rewritten."
            except Exception as exc:
                return current_draft, f"Error: {exc}"

        rewrite_section_btn.click(
            rewrite_section,
            inputs=[pending_state, draft_display, section_name_input, section_instruction_input],
            outputs=[draft_display, rewrite_status],
        )

        # ── manuscript approve / revise ──────────────────────────────────────────

        def approve_manuscript(pending, current_draft):
            if pending is None:
                return None, "No pending checkpoint.", gr.update(visible=False)
            # Pass the current (possibly user-edited) draft back so the orchestrator saves it
            ui_handler.resume({"action": "approve", "edited_draft": current_draft})
            return None, "Approved. Manuscript finalised.", gr.update(visible=False)

        def revise_manuscript(pending):
            if pending is None:
                return None, "No pending checkpoint.", gr.update(visible=False)
            ui_handler.resume({"action": "revise"})
            return None, "Full revision requested. Waiting for new draft...", gr.update(visible=False)

        approve_btn_manuscript.click(
            approve_manuscript,
            inputs=[pending_state, draft_display],
            outputs=[pending_state, status_out, manuscript_panel],
        )
        revise_btn_manuscript.click(
            revise_manuscript,
            inputs=[pending_state],
            outputs=[pending_state, status_out, manuscript_panel],
        )

    return app


if __name__ == "__main__":
    build_app().launch()
