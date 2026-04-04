# tests/integration/test_orchestrator_routing.py
import pytest
from unittest.mock import patch, MagicMock
from slr_agent.orchestrator import create_orchestrator
from slr_agent.llm import MockLLM
from slr_agent.config import RunConfig


def make_llm():
    llm = MockLLM()
    llm.register("detect the language", {"language_code": "en"})
    llm.register("expand this research question into PICO", {
        "population": "adults", "intervention": "aspirin",
        "comparator": "placebo", "outcome": "bp reduction",
    })
    llm.register("generate PubMed search query strings", {"query_strings": ["aspirin[tiab]"]})
    llm.register("screen the following abstracts", {"decisions": []})
    llm.register("synthesise the evidence", {"claims": [], "narrative": "No evidence found."})
    llm.register("write the Methods section", {"text": "Methods."})
    llm.register("write the Results section", {"text": "Results."})
    llm.register("write the Discussion section", {"text": "Discussion."})
    return llm


def test_orchestrator_runs_without_checkpoints(db, tmp_path):
    with patch("slr_agent.subgraphs.search.Entrez") as mock_entrez:
        mock_entrez.esearch.return_value = MagicMock()
        mock_entrez.read.return_value = {"IdList": []}
        mock_entrez.efetch.return_value = MagicMock()

        with patch("slr_agent.subgraphs.manuscript.run_pandoc", return_value=None):
            orchestrator = create_orchestrator(
                db=db,
                llm=make_llm(),
                output_dir=str(tmp_path),
                config=RunConfig(
                    checkpoint_stages=[],   # no checkpoints
                    fetch_fulltext=False,
                    output_format="markdown",
                    pubmed_api_key=None,
                    max_results=10,
                ),
            )
            result = orchestrator.invoke({
                "run_id": "run-orch-test",
                "raw_question": "Does aspirin reduce blood pressure?",
            })

    assert result["manuscript_path"] is not None
    assert result["pico"]["intervention"] == "aspirin"


def test_orchestrator_skips_fulltext_when_disabled(db, tmp_path):
    with patch("slr_agent.subgraphs.search.Entrez") as mock_entrez:
        mock_entrez.esearch.return_value = MagicMock()
        mock_entrez.read.return_value = {"IdList": []}

        with patch("slr_agent.subgraphs.manuscript.run_pandoc", return_value=None):
            orchestrator = create_orchestrator(
                db=db, llm=make_llm(), output_dir=str(tmp_path),
                config=RunConfig(
                    checkpoint_stages=[], fetch_fulltext=False,
                    output_format="markdown", pubmed_api_key=None, max_results=10,
                ),
            )
            result = orchestrator.invoke({
                "run_id": "run-orch-test-2",
                "raw_question": "Does aspirin reduce blood pressure?",
            })

    assert result["fulltext_counts"] is None
