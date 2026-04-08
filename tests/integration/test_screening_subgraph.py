# tests/integration/test_screening_subgraph.py
import pytest
from slr_agent.subgraphs.screening import create_screening_subgraph, _derive_decision
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


def test_screening_stores_criterion_scores(db_with_papers, mock_llm):
    """Criterion scores are stored on the paper record and drive the decision."""
    scores_99999 = [
        {"criterion": "Adults with hypertension", "type": "inclusion", "met": "yes", "note": "500 adults randomised"},
        {"criterion": "Aspirin as intervention", "type": "inclusion", "met": "yes", "note": "aspirin 100mg daily"},
        {"criterion": "Case reports or editorials", "type": "exclusion", "met": "no", "note": "This is an RCT"},
        {"criterion": "RCT", "type": "study_design", "met": "yes", "note": "randomised controlled trial"},
    ]
    scores_88888 = [
        {"criterion": "Adults with hypertension", "type": "inclusion", "met": "no", "note": "Headache patient, not hypertension"},
        {"criterion": "Aspirin as intervention", "type": "inclusion", "met": "yes", "note": "aspirin mentioned"},
        {"criterion": "Case reports or editorials", "type": "exclusion", "met": "yes", "note": "explicitly a case report"},
        {"criterion": "RCT", "type": "study_design", "met": "no", "note": "case report"},
    ]
    mock_llm.register(
        "You are screening abstracts for a systematic review",
        {
            "decisions": [
                {"pmid": "99999", "decision": "include", "reason": "All inclusion criteria met, RCT design",
                 "criterion_scores": scores_99999},
                {"pmid": "88888", "decision": "exclude", "reason": "Exclusion criterion met: case report",
                 "criterion_scores": scores_88888},
            ]
        },
    )
    graph = create_screening_subgraph(db=db_with_papers, llm=mock_llm)
    from slr_agent.state import PICOResult
    graph.invoke({
        "run_id": "run-test",
        "pico": PICOResult(
            population="adults with hypertension", intervention="aspirin",
            comparator="placebo", outcome="blood pressure reduction",
            query_strings=[], source_language="en",
            search_language="en", output_language="en",
        ),
        "screening_counts": None,
    })
    included = db_with_papers.get_papers_by_decision("run-test", "include")
    excluded = db_with_papers.get_papers_by_decision("run-test", "exclude")
    assert len(included) == 1 and included[0]["pmid"] == "99999"
    assert len(excluded) == 1 and excluded[0]["pmid"] == "88888"
    assert len(included[0]["criterion_scores"]) == 4
    assert len(excluded[0]["criterion_scores"]) == 4


def test_derive_decision_logic():
    """_derive_decision correctly applies inclusion/exclusion/study_design rules."""
    # All met → include
    assert _derive_decision([
        {"criterion": "adults", "type": "inclusion", "met": "yes", "note": ""},
        {"criterion": "case report", "type": "exclusion", "met": "no", "note": ""},
        {"criterion": "RCT", "type": "study_design", "met": "yes", "note": ""},
    ]) == "include"

    # Exclusion criterion met → exclude
    assert _derive_decision([
        {"criterion": "adults", "type": "inclusion", "met": "yes", "note": ""},
        {"criterion": "case report", "type": "exclusion", "met": "yes", "note": ""},
    ]) == "exclude"

    # Inclusion criterion not met → exclude
    assert _derive_decision([
        {"criterion": "adults", "type": "inclusion", "met": "no", "note": ""},
        {"criterion": "case report", "type": "exclusion", "met": "no", "note": ""},
    ]) == "exclude"

    # Unclear → uncertain (when not already excluded)
    assert _derive_decision([
        {"criterion": "adults", "type": "inclusion", "met": "unclear", "note": ""},
        {"criterion": "case report", "type": "exclusion", "met": "no", "note": ""},
    ]) == "uncertain"

    # No study design matched → exclude
    assert _derive_decision([
        {"criterion": "adults", "type": "inclusion", "met": "yes", "note": ""},
        {"criterion": "RCT", "type": "study_design", "met": "no", "note": ""},
        {"criterion": "cohort", "type": "study_design", "met": "no", "note": ""},
    ]) == "exclude"


def test_screening_retries_missing_pmid(db_with_papers, mock_llm):
    """LLM returns decision for only one PMID in the batch; retry covers the other."""
    from slr_agent.state import PICOResult

    call_count = {"n": 0}
    original_chat = mock_llm.chat

    def patched_chat(messages, schema=None, think=False):
        call_count["n"] += 1
        text = " ".join(m.get("content", "") for m in messages)
        if call_count["n"] == 1:
            # First batch call — deliberately omits 88888
            return {"decisions": [
                {"pmid": "99999", "decision": "include", "reason": "RCT on hypertension and aspirin"},
            ]}
        # Retry call for 88888
        if "88888" in text:
            return {"decisions": [
                {"pmid": "88888", "decision": "exclude", "reason": "Case report only"},
            ]}
        return {"decisions": []}

    mock_llm.chat = patched_chat
    graph = create_screening_subgraph(db=db_with_papers, llm=mock_llm)
    result = graph.invoke({
        "run_id": "run-test",
        "pico": PICOResult(
            population="adults with hypertension", intervention="aspirin",
            comparator="placebo", outcome="blood pressure reduction",
            query_strings=[], source_language="en",
            search_language="en", output_language="en",
        ),
        "screening_counts": None,
        "config": {"screening_batch_size": 5},  # one batch for both papers
    })
    # Both papers should have received decisions
    assert result["screening_counts"]["n_included"] == 1
    assert result["screening_counts"]["n_excluded"] == 1
    assert call_count["n"] == 2  # initial batch + 1 retry
