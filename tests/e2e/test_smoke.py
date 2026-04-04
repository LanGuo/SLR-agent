"""
End-to-end smoke test. Requires:
- Ollama running with gemma4:26b
- Internet access (PubMed)

Run manually: pytest tests/e2e/test_smoke.py -v -s
Not run in CI (excluded by default via 'not e2e' marker).
"""
import pytest
import os

pytestmark = pytest.mark.e2e


@pytest.fixture
def orchestrator(tmp_path):
    from slr_agent.db import Database
    from slr_agent.llm import LLMClient
    from slr_agent.config import RunConfig
    from slr_agent.orchestrator import create_orchestrator
    db = Database(str(tmp_path / "smoke.db"))
    llm = LLMClient(model="gemma4:26b")
    config = RunConfig(
        checkpoint_stages=[],        # fully automated for smoke test
        fetch_fulltext=False,
        output_format="markdown",
        pubmed_api_key=os.getenv("PUBMED_API_KEY"),
        max_results=50,
        search_sources=["pubmed"],
    )
    return create_orchestrator(
        db=db, llm=llm,
        output_dir=str(tmp_path / "outputs"),
        config=config,
        db_path=str(tmp_path / "smoke.db"),
    ), db


def test_full_pipeline_aspirin(orchestrator, tmp_path):
    graph, db = orchestrator
    result = graph.invoke({
        "run_id": "smoke-aspirin",
        "raw_question": "Does aspirin reduce cardiovascular events in adults?",
    }, config={"configurable": {"thread_id": "smoke-aspirin"}})

    assert result["pico"] is not None
    assert result["pico"]["intervention"] != ""
    assert result["search_counts"]["n_retrieved"] > 0
    assert result["screening_counts"] is not None
    assert result["manuscript_path"] is not None
    assert os.path.exists(result["manuscript_path"])

    content = open(result["manuscript_path"]).read()
    assert "Methods" in content
    assert "Results" in content
    assert len(content) > 500
