# tests/integration/test_synthesis_subgraph.py
import pytest
from slr_agent.subgraphs.synthesis import create_synthesis_subgraph
from slr_agent.llm import MockLLM


def test_synthesis_writes_narrative(db, sample_paper, mock_llm, tmp_path):
    db.ensure_run("run-test")
    db.upsert_paper(sample_paper)

    mock_llm.register("synthesise the evidence", {
        "claims": [
            {"text": "Aspirin significantly reduces systolic blood pressure.", "supporting_pmids": ["99999"]},
            {"text": "Effect size was approximately 8 mmHg.", "supporting_pmids": ["99999"]},
        ],
        "narrative": "Based on one RCT, aspirin reduces SBP by approximately 8 mmHg compared to placebo.",
    })

    graph = create_synthesis_subgraph(db=db, llm=mock_llm, output_dir=str(tmp_path))
    from slr_agent.state import PICOResult, ScreeningCounts, ExtractionCounts
    result = graph.invoke({
        "run_id": "run-test",
        "pico": PICOResult(
            population="adults with hypertension", intervention="aspirin",
            comparator="placebo", outcome="blood pressure reduction",
            query_strings=[], source_language="en",
            search_language="en", output_language="en",
        ),
        "search_counts": {"n_retrieved": 10, "n_duplicates_removed": 0},
        "screening_counts": ScreeningCounts(n_included=1, n_excluded=9, n_uncertain=0),
        "fulltext_counts": None,
        "extraction_counts": ExtractionCounts(
            n_extracted=1, n_grade_high=0, n_grade_moderate=1,
            n_grade_low=0, n_grade_very_low=0, n_quarantined_fields=0,
        ),
        "synthesis_path": None,
    })

    assert result["synthesis_path"] is not None
    import os
    assert os.path.exists(result["synthesis_path"])


def test_synthesis_returns_unresolved_questions(db, mock_llm, sample_paper):
    """synthesis_node returns unresolved_questions in state."""
    db.ensure_run("run-qs")
    p = dict(sample_paper)
    p["run_id"] = "run-qs"
    db.upsert_paper(p)

    mock_llm.register(
        "synthesise the evidence",
        {
            "claims": [{"text": "Aspirin reduces SBP.", "supporting_pmids": ["99999"]}],
            "narrative": "Evidence shows aspirin reduces SBP.",
            "unresolved_questions": [
                {
                    "question": "Does the effect persist beyond 12 months?",
                    "relevant_pmids": ["99999"],
                    "importance": "high",
                }
            ],
        },
    )

    from slr_agent.subgraphs.synthesis import create_synthesis_subgraph
    from slr_agent.state import PICOResult
    graph = create_synthesis_subgraph(db=db, llm=mock_llm, output_dir="/tmp")
    result = graph.invoke({
        "run_id": "run-qs",
        "pico": PICOResult(
            population="adults with hypertension", intervention="aspirin",
            comparator="placebo", outcome="blood pressure reduction",
            query_strings=[], source_language="en",
            search_language="en", output_language="en",
        ),
        "search_counts": {},
        "screening_counts": {"n_included": 1, "n_excluded": 0},
        "fulltext_counts": None,
        "extraction_counts": {},
    })

    assert "unresolved_questions" in result
    assert len(result["unresolved_questions"]) == 1
    assert result["unresolved_questions"][0]["importance"] == "high"
