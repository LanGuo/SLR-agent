# tests/integration/test_orchestrator_routing.py
import os
import pytest
from unittest.mock import patch, MagicMock
from slr_agent.orchestrator import create_orchestrator
from slr_agent.llm import MockLLM
from slr_agent.config import RunConfig
from slr_agent.template import DEFAULT_PRISMA_TEMPLATE


def make_llm():
    llm = MockLLM()
    llm.register("detect the language", {"language_code": "en"})
    llm.register("expand this research question into PICO", {
        "population": "adults", "intervention": "aspirin",
        "comparator": "placebo", "outcome": "bp reduction",
    })
    llm.register("Generate 3-4 PubMed search query strings", {"query_strings": ["aspirin AND blood pressure"]})
    llm.register("Generate explicit, specific inclusion and exclusion criteria", {
        "inclusion_criteria": ["RCTs and systematic reviews", "Adults ≥18"],
        "exclusion_criteria": ["Animal studies", "Non-English"],
        "study_designs": ["RCT", "meta-analysis"],
    })
    llm.register("You are screening abstracts for a systematic review", {"decisions": []})
    llm.register("synthesise the evidence", {"claims": [], "narrative": "No evidence found."})
    for section in DEFAULT_PRISMA_TEMPLATE["sections"]:
        llm.register(f"write the {section['name'].lower()} section", {"text": f"{section['name']} text."})
    llm.register("score the following systematic review draft", {"scores": []})
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
                    checkpoint_stages=[],
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


def test_orchestrator_computes_citation_network(db, tmp_path):
    """Citation network summary is present in final state after a run."""
    from unittest.mock import patch, MagicMock
    from slr_agent.llm import MockLLM
    from slr_agent.orchestrator import create_orchestrator
    from slr_agent.broker import CheckpointBroker, NoOpHandler
    from slr_agent.emitter import ProgressEmitter
    from slr_agent.config import DEFAULT_CONFIG

    llm = MockLLM()
    llm.register("detect the language", {"language_code": "en"})
    llm.register("expand this research question into PICO", {
        "population": "adults", "intervention": "aspirin",
        "comparator": "placebo", "outcome": "BP reduction",
    })
    llm.register("Generate 3-4 PubMed search query strings", {"query_strings": ["hypertension"]})
    llm.register("Generate explicit, specific inclusion and exclusion criteria", {
        "inclusion_criteria": ["RCT"], "exclusion_criteria": ["review"], "study_designs": ["RCT"]
    })
    llm.register("You are screening abstracts for a systematic review", {"decisions": [
        {"pmid": "11111", "decision": "include", "reason": "RCT",
         "criterion_scores": [{"criterion": "RCT", "type": "study_design", "met": "yes", "note": ""}]}
    ]})
    llm.register("Screen this full text", {"decision": "include", "reason": "RCT confirmed"})
    llm.register("Extract structured data", {
        "sample_size": "100", "intervention": "aspirin", "comparator": "placebo",
        "primary_outcome": "BP", "result": "reduced", "study_design": "RCT",
        "follow_up_duration": "12 weeks", "population_details": "adults with hypertension",
    })
    llm.register("supported", {"supported": True, "span": "aspirin"})
    llm.register("synthesise the evidence", {
        "narrative": "Aspirin reduces BP.", "claims": [], "unresolved_questions": []
    })
    # Section writing must come before GRADE to avoid matching "GRADE" in the search_context
    for section_name in ["Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusions"]:
        llm.register(f"write the {section_name.lower()} section", {"text": f"{section_name} text."})
    llm.register("Anchor citations for a systematic review", {"anchored": []})
    llm.register("You are an adversarial reviewer", {"issues": []})
    llm.register("score the following systematic review draft", {"scores": []})
    llm.register("GRADE", {"certainty": "moderate", "risk_of_bias": "low",
                            "inconsistency": "no", "indirectness": "no",
                            "imprecision": "no", "rationale": "RCT"})
    llm.register("Does the source text", {"supported": False, "span": ""})

    output_dir = str(tmp_path)
    orchestrator = create_orchestrator(
        db=db, llm=llm, output_dir=output_dir,
        config={**DEFAULT_CONFIG, "checkpoint_stages": [], "fetch_fulltext": False,
                "search_sources": ["pubmed"]},
        broker=CheckpointBroker(NoOpHandler()),
        emitter=ProgressEmitter(output_dir=output_dir, run_id="test-cn"),
    )

    with patch("slr_agent.subgraphs.search.Entrez") as mock_e:
        mock_e.esearch.return_value = MagicMock()
        mock_e.read.side_effect = [
            {"IdList": ["11111"]}, {"IdList": []},
            {"PubmedArticle": [{"MedlineCitation": {
                "PMID": "11111",
                "Article": {"ArticleTitle": "Test", "Abstract": {"AbstractText": "aspirin reduces BP"}},
            }}]},
        ]
        mock_e.efetch.return_value = MagicMock()
        result = orchestrator.invoke(
            {"run_id": "test-cn", "raw_question": "Does aspirin reduce BP?"}
        )

    assert "citation_network" in result
    cn = result["citation_network"]
    assert isinstance(cn, dict)
    assert cn["n_papers"] >= 0


def test_orchestrator_emits_stage_files(db, tmp_path):
    """ProgressEmitter writes stage JSON files to outputs/<run_id>/."""
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
                "run_id": "run-emit-test",
                "raw_question": "Does aspirin reduce blood pressure?",
            })

    run_dir = os.path.join(str(tmp_path), "run-emit-test")
    assert os.path.isdir(run_dir)
    assert os.path.exists(os.path.join(run_dir, "stage_1_pico.json"))
    assert os.path.exists(os.path.join(run_dir, "stage_2_search.json"))
