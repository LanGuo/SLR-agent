# slr_agent/llm.py
import json
import time
import ollama
from typing import Any

class LLMClient:
    """Thin wrapper around Ollama with retry and structured JSON output."""

    def __init__(self, model: str = "gemma4:26b", max_retries: int = 3):
        self.model = model
        self.max_retries = max_retries

    def chat(self, messages: list[dict], schema: dict | None = None) -> dict:
        """Call Ollama and return parsed JSON. Raises after max_retries."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
                if schema:
                    kwargs["format"] = schema
                response = ollama.chat(**kwargs)
                content = response["message"]["content"]
                return json.loads(content) if schema else {"text": content}
            except json.JSONDecodeError as e:
                last_error = e
                # Ask model to correct its malformed JSON output
                raw_content = response["message"]["content"]
                messages = messages + [
                    {"role": "assistant", "content": raw_content},
                    {"role": "user", "content": "Your response was not valid JSON. Reply with valid JSON only."},
                ]
                time.sleep(2 ** attempt)
            except Exception as e:
                last_error = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM failed after {self.max_retries} attempts: {last_error}")


class MockLLM:
    """Deterministic LLM for tests. Register canned responses by prompt substring."""

    def __init__(self):
        self._responses: list[tuple[str, dict]] = []

    def register(self, prompt_contains: str, response: dict) -> None:
        for i, (substr, _) in enumerate(self._responses):
            if substr == prompt_contains:
                self._responses[i] = (prompt_contains, response)
                return
        self._responses.append((prompt_contains, response))

    def chat(self, messages: list[dict], schema: dict | None = None) -> dict:
        full_text = " ".join(m.get("content", "") for m in messages)
        for substr, resp in self._responses:
            if substr in full_text:
                return resp
        raise ValueError(f"MockLLM: no registered response for prompt: {full_text[:120]}")
