# slr_agent/llm.py
import json
import time
import ollama
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from slr_agent.trace import TraceWriter


class LLMClient:
    """Thin wrapper around Ollama with retry and structured JSON output."""

    def __init__(
        self,
        model: str = "gemma4:e4b",
        max_retries: int = 3,
        trace_writer: "TraceWriter | None" = None,
    ):
        self.model = model
        self.max_retries = max_retries
        self._trace = trace_writer

    def chat(self, messages: list[dict], schema: dict | None = None, think: bool = False) -> dict:
        """Call Ollama and return parsed JSON. Raises after max_retries.

        Args:
            messages: Chat messages. Each dict may include an ``images`` key with
                a list of base64-encoded PNG strings for multimodal calls.
            schema: JSON schema for structured output (uses Ollama format= param).
            think: Enable Gemma 4 thinking mode. The reasoning chain is saved to
                llm_trace.jsonl (if tracing is enabled) and discarded from the return value.
        """
        last_error = None
        t_start = time.time()
        for attempt in range(self.max_retries):
            t_attempt = time.time()
            try:
                kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
                if schema:
                    kwargs["format"] = schema
                if think:
                    kwargs["think"] = True
                response = ollama.chat(**kwargs)
                content = response["message"]["content"]
                result = json.loads(content) if schema else {"text": content}

                if self._trace is not None:
                    thinking = None
                    try:
                        thinking = response["message"].get("thinking")
                    except Exception:
                        pass
                    prompt_tokens = completion_tokens = None
                    try:
                        prompt_tokens = response.get("prompt_eval_count")
                        completion_tokens = response.get("eval_count")
                    except Exception:
                        pass
                    self._trace.write_llm(
                        messages=messages,
                        schema=schema,
                        think=think,
                        thinking=thinking,
                        response_text=content,
                        latency_s=time.time() - t_attempt,
                        attempt=attempt + 1,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        model=self.model,
                    )

                return result
            except json.JSONDecodeError as e:
                last_error = e
                raw_content = response["message"]["content"]
                if self._trace is not None:
                    self._trace.write_llm(
                        messages=messages, schema=schema, think=think, thinking=None,
                        response_text=raw_content, latency_s=time.time() - t_attempt,
                        attempt=attempt + 1, prompt_tokens=None, completion_tokens=None,
                        model=self.model, error=f"JSONDecodeError: {e}",
                    )
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
        self._responses.append((prompt_contains, response))

    def chat(self, messages: list[dict], schema: dict | None = None, think: bool = False) -> dict:
        full_text = " ".join(m.get("content", "") for m in messages)
        for substr, resp in self._responses:
            if substr in full_text:
                return resp
        raise ValueError(f"MockLLM: no registered response for prompt: {full_text[:120]}")
