# tests/unit/test_llm.py
import pytest
from slr_agent.llm import MockLLM

def test_mock_llm_returns_registered_response():
    llm = MockLLM()
    llm.register("detect language", {"language_code": "en"})
    result = llm.chat([{"role": "user", "content": "Please detect language of this text"}])
    assert result == {"language_code": "en"}

def test_mock_llm_raises_on_no_match():
    llm = MockLLM()
    with pytest.raises(ValueError, match="no registered response"):
        llm.chat([{"role": "user", "content": "something unregistered"}])

def test_mock_llm_last_register_wins_for_same_key():
    # re-registering the same substring replaces the earlier response (upsert)
    llm = MockLLM()
    llm.register("foo", {"answer": "first"})
    llm.register("foo", {"answer": "second"})
    result = llm.chat([{"role": "user", "content": "foo bar"}])
    assert result["answer"] == "second"
