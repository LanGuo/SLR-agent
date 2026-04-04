"""
Critical safety regression test: hallucinated extracted fields must be quarantined.
If this test fails, the grounding layer is broken and the system will propagate
unverified claims into the manuscript.
"""
from slr_agent.grounding import ExtractionGrounder

REAL_ABSTRACT = (
    "A randomised controlled trial of 245 patients with type 2 diabetes assigned to "
    "metformin 500mg twice daily or placebo. Primary endpoint: HbA1c reduction at 24 weeks. "
    "Results: HbA1c reduced by 1.2% in metformin group vs 0.1% in placebo (p=0.003). "
    "Adverse events: nausea in 18% of metformin patients."
)

HALLUCINATED_FIELDS = [
    ("sample_size", "1200 patients from 15 countries"),
    ("intervention", "insulin glargine 30 units daily"),
    ("follow_up", "5 years"),
    ("result", "HbA1c reduced by 3.8% (p<0.0001)"),
    ("adverse_events", "fatal hepatotoxicity in 4 patients"),
]

REAL_FIELDS = [
    ("sample_size", "245 patients"),
    ("intervention", "metformin 500mg twice daily"),
    ("primary_endpoint", "HbA1c reduction at 24 weeks"),
    ("result", "HbA1c reduced by 1.2%"),
]


def test_all_hallucinated_fields_are_quarantined():
    grounder = ExtractionGrounder(threshold=85)
    for field_name, value in HALLUCINATED_FIELDS:
        result = grounder.ground_field(
            field_name=field_name,
            value=value,
            source_text=REAL_ABSTRACT,
            pmid="test",
            source="abstract",
        )
        assert result["status"] == "quarantined", (
            f"SAFETY FAILURE: hallucinated field '{field_name}' = '{value}' "
            f"was NOT quarantined (score={result['confidence']:.1f}). "
            f"The grounding layer is passing fabricated data."
        )


def test_real_fields_are_grounded():
    grounder = ExtractionGrounder(threshold=85)
    for field_name, value in REAL_FIELDS:
        result = grounder.ground_field(
            field_name=field_name,
            value=value,
            source_text=REAL_ABSTRACT,
            pmid="test",
            source="abstract",
        )
        assert result["status"] == "grounded", (
            f"Real field '{field_name}' = '{value}' was incorrectly quarantined "
            f"(score={result['confidence']:.1f}). Grounding threshold may be too strict."
        )
