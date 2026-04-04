# tests/integration/test_screening_subgraph.py
import pytest
from slr_agent.subgraphs.screening import create_screening_subgraph
from slr_agent.llm import MockLLM
from slr_agent.db import Database, PaperRecord, GRADEScore

@pytest.fixture
def db_with_papers(db, sample_paper):
    db.ensure_run("run-test")
    db.upsert_paper(sample_paper)
    # Add a paper that should be excluded
    excluded = dict(sample_paper)
    excluded["pmid"] = "88888"
    excluded["title"] = "Aspirin for headache: case report"
    excluded["abstract"] = "We report a case of headache treated with aspirin."
    db.upsert_paper(excluded)
    return db

def test_screening_includes_and_excludes(db_with_papers, mock_llm):
    mock_llm.register(
        "You are screening abstracts for a systematic review",
        {
            "decisions": [
                {"pmid": "99999", "decision": "include", "reason": "Matches PICO criteria: RCT, hypertension, aspirin"},
                {"pmid": "88888", "decision": "exclude", "reason": "Case report, not an RCT, wrong population"},
            ]
        },
    )
    graph = create_screening_subgraph(db=db_with_papers, llm=mock_llm)
    from slr_agent.state import PICOResult
    result = graph.invoke({
        "run_id": "run-test",
        "pico": PICOResult(
            population="adults with hypertension", intervention="aspirin",
            comparator="placebo", outcome="blood pressure reduction",
            query_strings=[], source_language="en",
            search_language="en", output_language="en",
        ),
        "screening_counts": None,
    })
    assert result["screening_counts"]["n_included"] == 1
    assert result["screening_counts"]["n_excluded"] == 1
    included = db_with_papers.get_papers_by_decision("run-test", "include")
    assert len(included) == 1
    assert included[0]["pmid"] == "99999"
