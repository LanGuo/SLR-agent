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
