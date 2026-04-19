# slr_agent/cache.py
import hashlib
import json
import os
from datetime import datetime, timezone


class LLMCache:
    """Disk-backed cache for LLM call results, keyed by SHA-256 of inputs.

    Cache files live in cache_dir as <hash>.json. Each file stores the
    result dict, the model name, and the ISO timestamp of the cache write.
    The cache is intentionally run-scoped (callers pass outputs/<run_id>/.llm_cache/)
    so stale results from a different model or prompt version can't leak in.
    """

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _key(
        self,
        model: str,
        messages: list[dict],
        schema: dict | None,
        think: bool,
    ) -> str:
        payload = json.dumps(
            {"model": model, "messages": messages, "schema": schema, "think": think},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(
        self,
        model: str,
        messages: list[dict],
        schema: dict | None,
        think: bool,
    ) -> dict | None:
        path = self._path(self._key(model, messages, schema, think))
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)["result"]

    def put(
        self,
        model: str,
        messages: list[dict],
        schema: dict | None,
        think: bool,
        result: dict,
    ) -> None:
        path = self._path(self._key(model, messages, schema, think))
        with open(path, "w") as f:
            json.dump(
                {
                    "result": result,
                    "model": model,
                    "cached_at": datetime.now(timezone.utc).isoformat(),
                },
                f,
            )
