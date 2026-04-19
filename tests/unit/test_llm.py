# tests/unit/test_llm.py
import pytest
from unittest.mock import patch, MagicMock
from slr_agent.llm import MockLLM
from slr_agent.cache import LLMCache
from slr_agent.llm import LLMClient

def test_mock_llm_returns_registered_response():
    llm = MockLLM()
    llm.register("detect language", {"language_code": "en"})
    result = llm.chat([{"role": "user", "content": "Please detect language of this text"}])
    assert result == {"language_code": "en"}

def test_mock_llm_raises_on_no_match():
    llm = MockLLM()
    with pytest.raises(ValueError, match="no registered response"):
        llm.chat([{"role": "user", "content": "something unregistered"}])

def test_mock_llm_first_match_wins():
    llm = MockLLM()
    llm.register("foo", {"answer": "first"})
    llm.register("foo", {"answer": "second"})
    result = llm.chat([{"role": "user", "content": "foo bar"}])
    assert result["answer"] == "first"


def test_llm_client_returns_cached_result_without_calling_ollama(tmp_path):
    cache = LLMCache(str(tmp_path))
    llm = LLMClient(model="gemma4:e4b", cache=cache)

    messages = [{"role": "user", "content": "extract"}]
    schema = {"type": "object", "properties": {"val": {"type": "string"}}, "required": ["val"]}
    expected = {"val": "cached"}

    cache.put("gemma4:e4b", messages, schema, False, expected)

    with patch("slr_agent.llm.ollama") as mock_ollama:
        result = llm.chat(messages, schema=schema)
        mock_ollama.chat.assert_not_called()

    assert result == expected


def test_llm_client_populates_cache_after_ollama_call(tmp_path):
    cache = LLMCache(str(tmp_path))
    llm = LLMClient(model="gemma4:e4b", cache=cache)

    messages = [{"role": "user", "content": "screen"}]
    schema = {"type": "object", "properties": {"decision": {"type": "string"}}, "required": ["decision"]}
    expected = {"decision": "include"}

    mock_response = {
        "message": {"content": '{"decision": "include"}'},
        "prompt_eval_count": 10,
        "eval_count": 5,
    }

    with patch("slr_agent.llm.ollama") as mock_ollama:
        mock_ollama.chat.return_value = mock_response
        result = llm.chat(messages, schema=schema)

    assert result == expected
    assert cache.get("gemma4:e4b", messages, schema, False) == expected


def test_llm_client_without_cache_still_works(tmp_path):
    llm = LLMClient(model="gemma4:e4b")  # no cache

    messages = [{"role": "user", "content": "hello"}]
    mock_response = {
        "message": {"content": '{"text": "hi"}'},
        "prompt_eval_count": 5,
        "eval_count": 3,
    }

    with patch("slr_agent.llm.ollama") as mock_ollama:
        mock_ollama.chat.return_value = mock_response
        result = llm.chat(messages, schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]})

    assert result == {"text": "hi"}
