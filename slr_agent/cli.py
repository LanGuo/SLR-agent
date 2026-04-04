# slr_agent/cli.py
import os
import uuid
import click
from slr_agent.db import Database
from slr_agent.llm import LLMClient
from slr_agent.config import DEFAULT_CONFIG, RunConfig
from slr_agent.orchestrator import create_orchestrator

_DB_PATH = "slr_runs.db"
_OUTPUT_DIR = "outputs"


def _get_orchestrator(config: RunConfig):
    db = Database(_DB_PATH)
    llm = LLMClient()
    return create_orchestrator(db=db, llm=llm, output_dir=_OUTPUT_DIR, config=config, db_path=_DB_PATH), db


@click.group()
def cli():
    """SLR Agent — systematic literature review pipeline powered by Gemma 4."""


@cli.command()
@click.argument("question")
@click.option("--no-fulltext", is_flag=True, default=False, help="Skip full-text fetching")
@click.option("--no-checkpoints", is_flag=True, default=False, help="Run fully automated")
@click.option("--max-results", default=500, help="PubMed search cap")
@click.option("--api-key", default=None, envvar="PUBMED_API_KEY", help="PubMed API key")
def run(question, no_fulltext, no_checkpoints, max_results, api_key):
    """Start a new SLR run. Returns a run_id for resuming."""
    run_id = str(uuid.uuid4())[:8]
    config = RunConfig(
        checkpoint_stages=[] if no_checkpoints else DEFAULT_CONFIG["checkpoint_stages"],
        fetch_fulltext=not no_fulltext,
        output_format="both",
        pubmed_api_key=api_key,
        max_results=max_results,
        search_sources=DEFAULT_CONFIG["search_sources"],
    )
    orchestrator, db = _get_orchestrator(config)
    click.echo(f"Starting run {run_id}...")
    try:
        result = orchestrator.invoke(
            {"run_id": run_id, "raw_question": question},
            config={"configurable": {"thread_id": run_id}},
        )
        if result.get("manuscript_path"):
            click.echo(f"Complete. Manuscript: {result['manuscript_path']}")
        else:
            click.echo(f"Paused at checkpoint. Resume with: slr resume {run_id}")
    except Exception as e:
        click.echo(f"Failed: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.argument("run_id")
@click.option("--edits", default=None, help="JSON string of state edits to apply")
def resume(run_id, edits):
    """Resume a paused or failed run."""
    import json
    orchestrator, db = _get_orchestrator(DEFAULT_CONFIG)
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


@cli.command()
@click.argument("run_id")
def export(run_id):
    """Export the manuscript for a completed run."""
    import glob as _glob
    files = _glob.glob(f"{_OUTPUT_DIR}/{run_id}_manuscript*")
    if not files:
        click.echo(f"No manuscript found for run {run_id}", err=True)
        raise SystemExit(1)
    for f in files:
        click.echo(f"Manuscript: {f}")
