import pytest
from slr_agent.grounding import ExtractionGrounder, GroundedField
from slr_agent.db import Span

ABSTRACT = (
    "500 adults with hypertension were randomised to aspirin 100mg daily or placebo "
    "for 12 months. Mean SBP reduction was 8.2 mmHg in the aspirin group vs 1.1 mmHg "
    "in placebo (p<0.001)."
)

def test_grounds_exact_match():
    grounder = ExtractionGrounder(threshold=85)
    result = grounder.ground_field(
        field_name="sample_size",
        value="500 adults",
        source_text=ABSTRACT,
        pmid="1",
        source="abstract",
    )
    assert result["status"] == "grounded"
    assert result["span"] is not None
    assert result["confidence"] >= 85

def test_quarantines_hallucinated_value():
    grounder = ExtractionGrounder(threshold=85)
    result = grounder.ground_field(
        field_name="sample_size",
        value="1200 patients with diabetes",   # not in abstract
        source_text=ABSTRACT,
        pmid="1",
        source="abstract",
    )
    assert result["status"] == "quarantined"
    assert result["span"] is None

def test_ground_extracted_data_returns_grounded_and_quarantined():
    grounder = ExtractionGrounder(threshold=85)
    extracted = {
        "sample_size": "500 adults",
        "intervention": "aspirin 100mg daily",
        "hallucination": "1200 diabetic patients treated with metformin",
    }
    grounded, quarantined = grounder.ground_extracted_data(
        extracted_data=extracted,
        source_text=ABSTRACT,
        pmid="1",
        source="abstract",
        stage="extraction",
    )
    assert "sample_size" in grounded
    assert "intervention" in grounded
    assert len(quarantined) == 1
    assert quarantined[0]["field_name"] == "hallucination"
