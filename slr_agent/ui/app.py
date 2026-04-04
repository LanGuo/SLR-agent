# slr_agent/ui/app.py
import threading
import gradio as gr
from slr_agent.db import Database
from slr_agent.llm import LLMClient
from slr_agent.config import DEFAULT_CONFIG, RunConfig
from slr_agent.orchestrator import create_orchestrator
from slr_agent.ui.panels.pico import build_pico_panel
from slr_agent.ui.panels.generic import build_generic_panel

_DB_PATH = "slr_runs.db"
_OUTPUT_DIR = "outputs"
_run_results: dict = {}


def launch_run(question: str, fetch_fulltext: bool, checkpoint_stages_str: str):
    """Start a pipeline run in a background thread, returning run_id."""
    import uuid
    run_id = str(uuid.uuid4())[:8]
    stages = [int(s.strip()) for s in checkpoint_stages_str.split(",") if s.strip().isdigit()]
    config = RunConfig(
        checkpoint_stages=stages,
        fetch_fulltext=fetch_fulltext,
        output_format="both",
        pubmed_api_key=None,
        max_results=500,
        search_sources=DEFAULT_CONFIG["search_sources"],
    )
    db = Database(_DB_PATH)
    llm = LLMClient()
    orchestrator = create_orchestrator(
        db=db, llm=llm, output_dir=_OUTPUT_DIR, config=config, db_path=_DB_PATH
    )

    def _run():
        _run_results[run_id] = {"status": "running"}
        try:
            result = orchestrator.invoke(
                {"run_id": run_id, "raw_question": question},
                config={"configurable": {"thread_id": run_id}},
            )
            _run_results[run_id] = {"status": "complete", "result": result}
        except Exception as e:
            _run_results[run_id] = {"status": "error", "error": str(e)}

    threading.Thread(target=_run, daemon=True).start()
    return run_id, "Run started. Check status below."


def get_run_status(run_id: str):
    result = _run_results.get(run_id, {})
    status = result.get("status", "not found")
    if status == "complete":
        manuscript = result["result"].get("manuscript_path", "")
        return f"Complete. Manuscript: {manuscript}"
    elif status == "error":
        return f"Error: {result.get('error')}"
    return f"Status: {status}"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="SLR Agent") as app:
        gr.Markdown("# SLR Agent — Systematic Literature Review\nPowered by Gemma 4 (local)")

        with gr.Tab("New Run"):
            question_input = gr.Textbox(
                label="Research Question",
                placeholder="e.g. Do ACE inhibitors reduce blood pressure in hypertensive adults?",
                lines=2,
            )
            with gr.Row():
                fulltext_cb = gr.Checkbox(label="Fetch full text (PMC Open Access)", value=True)
                stages_input = gr.Textbox(
                    label="Checkpoint stages (comma-separated stage numbers 1-7)",
                    value="1, 3, 5",
                )
            start_btn = gr.Button("Start Run", variant="primary")
            run_id_out = gr.Textbox(label="Run ID", interactive=False)
            start_msg = gr.Textbox(label="Status", interactive=False)
            start_btn.click(
                launch_run,
                inputs=[question_input, fulltext_cb, stages_input],
                outputs=[run_id_out, start_msg],
            )

        with gr.Tab("Monitor"):
            monitor_run_id = gr.Textbox(label="Run ID")
            refresh_btn = gr.Button("Refresh Status")
            status_out = gr.Textbox(label="Status", interactive=False, lines=3)
            refresh_btn.click(get_run_status, inputs=[monitor_run_id], outputs=[status_out])

    return app


if __name__ == "__main__":
    build_app().launch()
