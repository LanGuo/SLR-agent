# slr_agent/ui/app.py
import queue
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


def build_app_with_handler(ui_handler: UIHandler, run_id: str) -> gr.Blocks:
    """Minimal app used by CLI --hitl ui: polls UIHandler and shows checkpoint panels."""
    with gr.Blocks(title=f"SLR Agent — Run {run_id}") as app:
        gr.Markdown(f"# SLR Agent Checkpoint Review\nRun: `{run_id}`")

        pending_state = gr.State(None)
        # Status shown at all times (outside the hidden panel)
        status_out = gr.Textbox(label="Status", value="Waiting for pipeline checkpoint...", interactive=False)

        # gr.Group supports visibility toggling reliably in Gradio 6
        with gr.Group(visible=False) as checkpoint_area:
            stage_label = gr.Markdown("## Checkpoint")
            data_json = gr.JSON(label="Stage Data")
            approve_btn = gr.Button("Approve & Continue", variant="primary")

        def poll_checkpoint(pending):
            if pending is not None:
                # Already holding a checkpoint — don't dequeue again
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            cp = ui_handler.get_pending(timeout=0.1)
            if cp is None:
                return pending, "Waiting for pipeline checkpoint...", gr.update(visible=False), "", None
            return (
                cp,
                "Review the data below and approve to continue.",
                gr.update(visible=True),
                f"## Stage {cp['stage']}: {cp['stage_name'].upper()}",
                cp["data"],
            )

        def approve(pending, data):
            if pending is None:
                return None, "No pending checkpoint.", gr.update(visible=False)
            ui_handler.resume({**(data or {}), "action": "approve"})
            return None, "Approved. Waiting for next checkpoint...", gr.update(visible=False)

        gr.Timer(1).tick(
            poll_checkpoint,
            inputs=[pending_state],
            outputs=[pending_state, status_out, checkpoint_area, stage_label, data_json],
        )
        approve_btn.click(
            approve,
            inputs=[pending_state, data_json],
            outputs=[pending_state, status_out, checkpoint_area],
        )

    return app


if __name__ == "__main__":
    build_app().launch()
