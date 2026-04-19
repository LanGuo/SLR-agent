# tests/unit/test_cache.py
import pytest
from slr_agent.cache import LLMCache


def test_cache_miss_returns_none(tmp_path):
    cache = LLMCache(str(tmp_path))
    result = cache.get("gemma4:e4b", [{"role": "user", "content": "hi"}], None, False)
    assert result is None


def test_cache_hit_returns_stored_result(tmp_path):
    cache = LLMCache(str(tmp_path))
    model = "gemma4:e4b"
    messages = [{"role": "user", "content": "extract data"}]
    schema = {"type": "object", "properties": {"val": {"type": "string"}}, "required": ["val"]}
    expected = {"val": "42"}

    cache.put(model, messages, schema, False, expected)
    assert cache.get(model, messages, schema, False) == expected


def test_different_messages_produce_different_keys(tmp_path):
    cache = LLMCache(str(tmp_path))
    model = "gemma4:e4b"
    result = {"val": "x"}
    cache.put(model, [{"role": "user", "content": "hello"}], None, False, result)
    assert cache.get(model, [{"role": "user", "content": "goodbye"}], None, False) is None


def test_different_think_flag_produces_different_keys(tmp_path):
    cache = LLMCache(str(tmp_path))
    model = "gemma4:e4b"
    messages = [{"role": "user", "content": "screen this"}]
    result = {"decision": "include"}
    cache.put(model, messages, None, think=False, result=result)
    assert cache.get(model, messages, None, think=True) is None


def test_cache_creates_directory_on_init(tmp_path):
    cache_dir = str(tmp_path / "nested" / "cache")
    LLMCache(cache_dir)
    import os
    assert os.path.isdir(cache_dir)


def test_cache_file_contains_metadata(tmp_path):
    import json, os
    cache = LLMCache(str(tmp_path))
    model = "gemma4:e4b"
    messages = [{"role": "user", "content": "hi"}]
    cache.put(model, messages, None, False, {"text": "ok"})
    files = os.listdir(str(tmp_path))
    assert len(files) == 1
    with open(os.path.join(str(tmp_path), files[0])) as f:
        data = json.load(f)
    assert data["result"] == {"text": "ok"}
    assert data["model"] == model
    assert "cached_at" in data


def test_llm_cache_path_convention(tmp_path):
    """Verify the cache directory naming convention used by cli.py."""
    import os
    run_id = "abc12345"
    output_dir = str(tmp_path)
    expected_cache_dir = os.path.join(output_dir, run_id, ".llm_cache")
    cache = LLMCache(expected_cache_dir)
    assert os.path.isdir(expected_cache_dir)
    assert expected_cache_dir.endswith(f"{run_id}/.llm_cache")
