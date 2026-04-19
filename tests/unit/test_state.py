from slr_agent.state import OrchestratorState, PICOResult
from slr_agent.config import RunConfig

def test_orchestrator_state_keys():
    state: OrchestratorState = {
        "run_id": "test-123",
        "config": RunConfig(
            checkpoint_stages=[1, 3, 5],
            fetch_fulltext=True,
            output_format="both",
            pubmed_api_key=None,
            max_results=500,
            search_sources=["pubmed", "biorxiv"],
        ),
        "pico": None,
        "search_counts": None,
        "screening_counts": None,
        "fulltext_counts": None,
        "extraction_counts": None,
        "synthesis_path": None,
        "manuscript_path": None,
        "current_stage": "pico",
        "checkpoint_pending": False,
    }
    assert state["run_id"] == "test-123"
    assert state["config"]["checkpoint_stages"] == [1, 3, 5]

def test_pico_result_keys():
    pico = PICOResult(
        population="adults with hypertension",
        intervention="ACE inhibitors",
        comparator="placebo",
        outcome="blood pressure reduction",
        query_strings=["hypertension[MeSH] AND ACE inhibitors[tiab]"],
        source_language="en",
        search_language="en",
        output_language="en",
    )
    assert len(pico["query_strings"]) == 1


def test_run_config_new_fields():
    from slr_agent.config import RunConfig
    cfg: RunConfig = {
        "checkpoint_stages": [1, 3],
        "fetch_fulltext": False,
        "output_format": "markdown",
        "pubmed_api_key": None,
        "max_results": 50,
        "search_sources": ["pubmed"],
        "template_path": "/tmp/template.json",
        "hitl_mode": "cli",
    }
    assert cfg["hitl_mode"] == "cli"
    assert cfg["template_path"] == "/tmp/template.json"


def test_orchestrator_state_new_fields():
    from slr_agent.state import OrchestratorState
    keys = OrchestratorState.__annotations__.keys()
    assert "template" in keys
    assert "manuscript_draft_version" in keys
    assert "date_from" in keys
    assert "date_to" in keys
    assert "screening_criteria" in keys


def test_orchestrator_state_has_citation_network_field():
    from slr_agent.state import OrchestratorState
    keys = OrchestratorState.__annotations__
    assert "citation_network" in keys
