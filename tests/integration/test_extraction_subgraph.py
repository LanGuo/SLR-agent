# tests/integration/test_extraction_subgraph.py
import pytest
from slr_agent.subgraphs.extraction import create_extraction_subgraph
from slr_agent.llm import MockLLM


def test_extraction_grounds_fields(db, sample_paper, mock_llm):
    db.ensure_run("run-test")
    db.upsert_paper(sample_paper)

    mock_llm.register("extract structured data", {
        "sample_size": "500 adults",
        "intervention": "aspirin 100mg daily",
        "comparator": "placebo",
        "primary_outcome": "blood pressure reduction",
        "result": "SBP reduction 8.2 mmHg vs 1.1 mmHg (p<0.001)",
        "study_design": "randomised controlled trial",
    })
    mock_llm.register("assess the quality of evidence", {
        "certainty": "moderate",
        "risk_of_bias": "low",
        "inconsistency": "no",
        "indirectness": "no",
        "imprecision": "some",
        "rationale": "Moderate certainty: single RCT",
    })

    graph = create_extraction_subgraph(db=db, llm=mock_llm)
    from slr_agent.state import PICOResult
    result = graph.invoke({
        "run_id": "run-test",
        "pico": PICOResult(
            population="adults with hypertension", intervention="aspirin",
            comparator="placebo", outcome="blood pressure reduction",
            query_strings=[], source_language="en",
            search_language="en", output_language="en",
        ),
        "extraction_counts": None,
    })

    assert result["extraction_counts"]["n_extracted"] == 1
    paper = db.get_paper("run-test", "99999")
    assert paper["extracted_data"]["sample_size"] == "500 adults"
    assert paper["grade_score"]["certainty"] == "moderate"


def test_extraction_quarantines_hallucinated_field(db, mock_llm):
    from slr_agent.db import PaperRecord, GRADEScore
    paper = PaperRecord(
        pmid="77777", run_id="run-test",
        title="Paper", abstract="10 patients were treated.",
        fulltext=None, source="abstract",
        screening_decision="include", screening_reason="ok",
        extracted_data={}, grade_score=GRADEScore(
            certainty="low", risk_of_bias="high", inconsistency="no",
            indirectness="no", imprecision="no", rationale="low"
        ),
        provenance=[], quarantined_fields=[],
    )
    db.ensure_run("run-test")
    db.upsert_paper(paper)

    mock_llm.register("extract structured data", {
        "sample_size": "10 patients",
        "hallucinated_field": "5000 participants from 12 countries with diabetes",
    })
    mock_llm.register("assess the quality of evidence", {
        "certainty": "very_low", "risk_of_bias": "high",
        "inconsistency": "no", "indirectness": "no", "imprecision": "serious",
        "rationale": "Very low certainty",
    })

    graph = create_extraction_subgraph(db=db, llm=mock_llm)
    from slr_agent.state import PICOResult
    graph.invoke({
        "run_id": "run-test",
        "pico": PICOResult(
            population="patients", intervention="treatment",
            comparator="none", outcome="outcome",
            query_strings=[], source_language="en",
            search_language="en", output_language="en",
        ),
        "extraction_counts": None,
    })

    quarantined = db.get_quarantine("run-test")
    quarantined_fields = [q["field_name"] for q in quarantined]
    assert "hallucinated_field" in quarantined_fields
