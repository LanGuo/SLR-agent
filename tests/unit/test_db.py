import pytest
import sqlite3
from slr_agent.db import Database, PaperRecord, Span, QuarantinedField, GRADEScore

@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "test.db"))

def test_create_schema(db):
    with db.connect() as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    names = {r[0] for r in tables}
    assert {"papers", "quarantine", "runs"}.issubset(names)

def test_upsert_and_get_paper(db):
    paper = PaperRecord(
        pmid="12345",
        run_id="run-1",
        title="Test paper",
        abstract="This is an abstract.",
        fulltext=None,
        source="abstract",
        screening_decision="include",
        screening_reason="Matches PICO criteria",
        extracted_data={},
        grade_score=GRADEScore(
            certainty="moderate",
            risk_of_bias="low",
            inconsistency="no",
            indirectness="no",
            imprecision="some",
            rationale="Moderate certainty based on RCT evidence",
        ),
        provenance=[],
        quarantined_fields=[],
    )
    db.upsert_paper(paper)
    retrieved = db.get_paper("run-1", "12345")
    assert retrieved["title"] == "Test paper"
    assert retrieved["screening_decision"] == "include"

def test_quarantine_insert(db):
    field = QuarantinedField(
        field_name="sample_size",
        value="150",
        stage="extraction",
        reason="no matching span",
    )
    db.insert_quarantine("run-1", "12345", field)
    rows = db.get_quarantine("run-1")
    assert len(rows) == 1
    assert rows[0]["field_name"] == "sample_size"

def test_get_papers_by_decision(db):
    for pmid, decision in [("1", "include"), ("2", "exclude"), ("3", "include")]:
        db.upsert_paper(PaperRecord(
            pmid=pmid, run_id="run-1", title=f"Paper {pmid}",
            abstract="Abstract.", fulltext=None, source="abstract",
            screening_decision=decision, screening_reason="reason",
            extracted_data={}, grade_score=GRADEScore(
                certainty="low", risk_of_bias="high", inconsistency="serious",
                indirectness="no", imprecision="no", rationale="Low quality"
            ),
            provenance=[], quarantined_fields=[],
        ))
    included = db.get_papers_by_decision("run-1", "include")
    assert len(included) == 2
