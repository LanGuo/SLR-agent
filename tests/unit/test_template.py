import json
import pytest
from unittest.mock import patch, MagicMock
from slr_agent.template import load_template, score_rubric, DEFAULT_PRISMA_TEMPLATE


def test_default_prisma_template_has_required_sections():
    sections = {s["name"] for s in DEFAULT_PRISMA_TEMPLATE["sections"]}
    for required in ("Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusions"):
        assert required in sections, f"Missing section: {required}"


def test_load_json_template(tmp_path):
    template_data = {
        "sections": [
            {
                "name": "Methods",
                "instructions": "Describe the search strategy.",
                "rubric_criteria": ["Names all databases", "States eligibility criteria"],
            }
        ],
        "style_notes": "Use passive voice.",
    }
    path = tmp_path / "template.json"
    path.write_text(json.dumps(template_data))
    result = load_template(str(path))
    assert result["sections"][0]["name"] == "Methods"
    assert result["style_notes"] == "Use passive voice."
    assert len(result["sections"][0]["rubric_criteria"]) == 2


def test_load_json_template_fills_missing_rubric_criteria(tmp_path):
    template_data = {
        "sections": [{"name": "Results", "instructions": "Report findings."}],
        "style_notes": "",
    }
    path = tmp_path / "template.json"
    path.write_text(json.dumps(template_data))
    result = load_template(str(path))
    # rubric_criteria defaults to empty list if not provided
    assert result["sections"][0]["rubric_criteria"] == []


def test_score_rubric_returns_scored_template():
    from slr_agent.llm import MockLLM
    llm = MockLLM()
    llm.register("score the following systematic review draft", {
        "scores": [{"criterion": "Names all databases", "score": "met", "explanation": "PubMed named."}]
    })
    template = {
        "sections": [{"name": "Methods", "instructions": "...", "rubric_criteria": ["Names all databases"]}],
        "style_notes": "",
    }
    result = score_rubric("## Methods\n\nWe searched PubMed.", template, llm)
    assert "scores" in result
    assert result["scores"][0]["score"] == "met"


def test_load_unsupported_format_raises(tmp_path):
    path = tmp_path / "template.txt"
    path.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported"):
        load_template(str(path))
