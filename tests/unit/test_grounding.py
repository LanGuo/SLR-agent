import pytest
from slr_agent.grounding import ExtractionGrounder, GroundedField, SynthesisGrounder
from slr_agent.db import Span
from slr_agent.llm import MockLLM

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

def test_synthesis_grounding_passes_with_citations(mock_llm):
    mock_llm.register(
        "Is this claim supported",
        {"supporting_pmids": ["1", "2"], "confidence": "high"},
    )
    grounder = SynthesisGrounder(llm=mock_llm)
    result = grounder.ground_claim(
        claim="Aspirin significantly reduces systolic blood pressure.",
        paper_extractions=[
            {"pmid": "1", "result": "SBP reduction 8.2 mmHg (p<0.001)"},
            {"pmid": "2", "result": "SBP reduction 7.1 mmHg (p<0.01)"},
        ],
    )
    assert result["status"] == "grounded"
    assert "1" in result["supporting_pmids"]

def test_synthesis_grounding_quarantines_unsupported_claim(mock_llm):
    mock_llm.register(
        "Is this claim supported",
        {"supporting_pmids": [], "confidence": "none"},
    )
    grounder = SynthesisGrounder(llm=mock_llm)
    result = grounder.ground_claim(
        claim="This treatment eliminates all cardiovascular risk.",
        paper_extractions=[{"pmid": "1", "result": "modest effect"}],
    )
    assert result["status"] == "quarantined"


def test_exact_match_sets_direct_provenance():
    grounder = ExtractionGrounder()
    result = grounder.ground_field(
        field_name="sample_size",
        value="500 adults",   # verbatim in ABSTRACT
        source_text=ABSTRACT,
        pmid="1",
        source="abstract",
    )
    assert result["status"] == "grounded"
    assert result["span"]["provenance_type"] == "direct"


def test_fuzzy_match_sets_paraphrased_provenance():
    # No explicit threshold: uses source-adaptive default (75 for "abstract")
    grounder = ExtractionGrounder()
    result = grounder.ground_field(
        field_name="result",
        value="aspirin 100mg daily or placebo for 12 months mean SBP reduction 8.2 mmHg",
        source_text=ABSTRACT,
        pmid="1",
        source="abstract",
    )
    assert result["status"] == "grounded"
    assert result["span"]["provenance_type"] == "paraphrased"
