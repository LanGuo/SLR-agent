# slr_agent/trace.py
"""Append-only trajectory logging for LLM calls and HITL gate interactions.

Two JSONL files are written to the run directory:
  llm_trace.jsonl  — one entry per Ollama call (prompt, response, thinking, latency, tokens)
  hitl_trace.jsonl — one entry per HITL gate (stage, before/after diff, user action)
"""
import json
import os
import time


class TraceWriter:
    """Thread-safe append-only writer for LLM and HITL trace files."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.llm_path = os.path.join(run_dir, "llm_trace.jsonl")
        self.hitl_path = os.path.join(run_dir, "hitl_trace.jsonl")

    def write_llm(
        self,
        *,
        messages: list[dict],
        schema: dict | None,
        think: bool,
        thinking: str | None,
        response_text: str,
        latency_s: float,
        attempt: int,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        model: str,
        error: str | None = None,
    ) -> None:
        """Append one LLM call record to llm_trace.jsonl."""
        entry = {
            "ts": time.time(),
            "model": model,
            "think": think,
            "attempt": attempt,
            "n_messages": len(messages),
            "messages": messages,
            "schema_keys": sorted(schema["properties"].keys()) if schema else None,
            "thinking": thinking,
            "response": response_text,
            "latency_s": round(latency_s, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "error": error,
        }
        self._append(self.llm_path, entry)

    def write_hitl(
        self,
        *,
        stage: int,
        stage_name: str,
        before: dict,
        after: dict,
    ) -> None:
        """Append one HITL interaction record to hitl_trace.jsonl."""
        diff = _compute_diff(before, after)
        entry = {
            "ts": time.time(),
            "stage": stage,
            "stage_name": stage_name,
            "action": after.get("action", "approve"),
            "n_changes": len(diff),
            "diff": diff,
            "before": before,
            "after": {k: v for k, v in after.items() if k != "action"},
        }
        self._append(self.hitl_path, entry)

    @staticmethod
    def _append(path: str, entry: dict) -> None:
        with open(path, "a") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")


def _compute_diff(before: dict, after: dict) -> dict:
    """Return a dict of keys whose values changed between before and after."""
    changes: dict = {}
    all_keys = set(before.keys()) | set(after.keys())
    for k in all_keys:
        if k == "action":
            continue
        b, a = before.get(k), after.get(k)
        if b != a:
            changes[k] = {"before": b, "after": a}
    return changes
