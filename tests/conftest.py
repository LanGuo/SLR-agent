# tests/conftest.py
import pytest
from slr_agent.db import Database, PaperRecord, GRADEScore
from slr_agent.llm import MockLLM

@pytest.fixture
def mock_llm():
    return MockLLM()

@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "test.db"))

@pytest.fixture
def sample_paper() -> PaperRecord:
    return PaperRecord(
        pmid="99999",
        run_id="run-test",
        title="Aspirin reduces cardiovascular events: an RCT",
        abstract=(
            "Background: Aspirin is widely used for cardiovascular prevention. "
            "Methods: 500 adults with hypertension were randomised to aspirin 100mg "
            "daily or placebo for 12 months. Primary outcome: blood pressure reduction. "
            "Results: Mean SBP reduction was 8.2 mmHg in the aspirin group vs 1.1 mmHg "
            "in placebo (p<0.001). Conclusion: Aspirin significantly reduces SBP."
        ),
        fulltext=None,
        page_image_paths=[],
        source="abstract",
        screening_decision="include",
        screening_reason="Matches PICO: hypertension, aspirin, placebo, BP reduction",
        criterion_scores=[],
        extracted_data={
            "sample_size": "500",
            "intervention": "aspirin 100mg daily",
            "comparator": "placebo",
            "primary_outcome": "blood pressure reduction",
            "result": "SBP reduction 8.2 mmHg vs 1.1 mmHg (p<0.001)",
        },
        grade_score=GRADEScore(
            certainty="moderate",
            risk_of_bias="low",
            inconsistency="no",
            indirectness="no",
            imprecision="some",
            rationale="Moderate certainty: single RCT, adequate allocation concealment",
        ),
        provenance=[],
        quarantined_fields=[],
    )
