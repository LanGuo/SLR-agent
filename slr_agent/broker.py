import json
import queue
import threading
from typing import TYPE_CHECKING, Any

import click

if TYPE_CHECKING:
    from slr_agent.trace import TraceWriter


class CheckpointBroker:
    """Pauses the pipeline at a stage and delegates to a handler for human input."""

    def __init__(self, handler: Any, trace_writer: "TraceWriter | None" = None):
        self._handler = handler
        self._trace = trace_writer

    def pause(self, stage: int, stage_name: str, data: dict) -> dict:
        """Block until human approves. Returns edited data with 'action' key."""
        result = self._handler.handle(stage, stage_name, data)
        if self._trace is not None:
            self._trace.write_hitl(
                stage=stage,
                stage_name=stage_name,
                before=data,
                after=result,
            )
        return result


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
                "[A]pprove  [E]dit fields inline  [S]kip",
                default="A",
            ).strip().upper()

            if choice in ("A", "S"):
                return {**data, "action": "approve"}
            if choice == "E":
                edited = self._edit_inline(data)
                return {**edited, "action": "approve"}

    def _edit_inline(self, data: dict) -> dict:
        """Prompt for each field; Enter keeps current value."""
        click.echo("\nEdit fields (press Enter to keep current value):\n")
        result = {}
        for key, value in data.items():
            if isinstance(value, list):
                click.echo(f"  {key} (list, one item per line; blank line to finish):")
                for i, item in enumerate(value):
                    click.echo(f"    [{i}] {item}")
                click.echo("  Enter new items (blank line = keep as-is):")
                new_items = []
                while True:
                    line = click.prompt("   +", default="", show_default=False)
                    if not line:
                        break
                    new_items.append(line)
                result[key] = new_items if new_items else value
            elif isinstance(value, dict):
                click.echo(f"  {key}: (dict — skipped, approve to keep)")
                result[key] = value
            else:
                new_val = click.prompt(f"  {key}", default=str(value), show_default=True)
                result[key] = new_val
        return result


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
