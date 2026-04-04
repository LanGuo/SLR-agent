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

    def log(self, message: str) -> None:
        """Emit a freeform status message without writing to disk."""
        self._echo(message)
        if self._gradio_queue is not None:
            self._gradio_queue.put(message)

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
