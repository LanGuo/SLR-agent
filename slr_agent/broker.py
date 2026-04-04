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
