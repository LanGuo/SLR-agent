# tests/integration/test_manuscript_subgraph.py
import os
import pytest
from unittest.mock import patch
from slr_agent.subgraphs.manuscript import create_manuscript_subgraph
from slr_agent.llm import MockLLM
from slr_agent.state import PICOResult
from slr_agent.template import DEFAULT_PRISMA_TEMPLATE


def _make_llm():
    llm = MockLLM()
    # Section generation — one response matched per section name
    for section in DEFAULT_PRISMA_TEMPLATE["sections"]:
        name = section["name"]
        llm.register(f"write the {name.lower()} section", {"text": f"Content of {name}."})
    # Rubric scoring
    llm.register("score the following systematic review draft", {
        "scores": [{"criterion": "Names all databases", "score": "met", "explanation": "PubMed named."}]
    })
    return llm


def test_manuscript_writes_markdown(db, tmp_path):
    db.ensure_run("run-test")
    synthesis_path = str(tmp_path / "synthesis.md")
    with open(synthesis_path, "w") as f:
        f.write("# Synthesis\n\nAspirin reduces SBP by 8 mmHg.")

    with patch("slr_agent.subgraphs.manuscript.run_pandoc", return_value=None):
        graph = create_manuscript_subgraph(db=db, llm=_make_llm(), output_dir=str(tmp_path))
        result = graph.invoke({
            "run_id": "run-test",
            "pico": PICOResult(
                population="adults with hypertension", intervention="aspirin",
                comparator="placebo", outcome="blood pressure reduction",
                query_strings=[], source_language="en",
                search_language="en", output_language="en",
            ),
            "synthesis_path": synthesis_path,
            "screening_counts": {"n_included": 1, "n_excluded": 9, "n_uncertain": 0},
            "manuscript_path": None,
            "template": None,
            "manuscript_draft_version": 0,
        })

    assert result["manuscript_path"] is not None
    assert os.path.exists(result["manuscript_path"])
    content = open(result["manuscript_path"]).read()
    assert "Methods" in content
    assert "Results" in content


def test_manuscript_uses_custom_template(db, tmp_path):
    db.ensure_run("run-test2")
    synthesis_path = str(tmp_path / "synthesis.md")
    with open(synthesis_path, "w") as f:
        f.write("Narrative.")

    custom_template = {
        "sections": [
            {"name": "Background", "instructions": "Describe rationale.", "rubric_criteria": []},
            {"name": "Methods", "instructions": "Detail methods.", "rubric_criteria": []},
        ],
        "style_notes": "",
    }

    llm = MockLLM()
    llm.register("write the background section", {"text": "Background content."})
    llm.register("write the methods section", {"text": "Methods content."})
    llm.register("score the following systematic review draft", {"scores": []})

    with patch("slr_agent.subgraphs.manuscript.run_pandoc", return_value=None):
        graph = create_manuscript_subgraph(db=db, llm=llm, output_dir=str(tmp_path))
        result = graph.invoke({
            "run_id": "run-test2",
            "pico": PICOResult(
                population="adults", intervention="aspirin",
                comparator="placebo", outcome="bp",
                query_strings=[], source_language="en",
                search_language="en", output_language="en",
            ),
            "synthesis_path": synthesis_path,
            "screening_counts": None,
            "manuscript_path": None,
            "template": custom_template,
            "manuscript_draft_version": 0,
        })

    content = open(result["manuscript_path"]).read()
    assert "Background" in content
    assert "Methods" in content


def test_manuscript_rubric_saved(db, tmp_path):
    db.ensure_run("run-test3")
    synthesis_path = str(tmp_path / "synthesis.md")
    with open(synthesis_path, "w") as f:
        f.write("Narrative.")

    llm = _make_llm()
    with patch("slr_agent.subgraphs.manuscript.run_pandoc", return_value=None):
        graph = create_manuscript_subgraph(db=db, llm=llm, output_dir=str(tmp_path))
        result = graph.invoke({
            "run_id": "run-test3",
            "pico": PICOResult(
                population="adults", intervention="aspirin",
                comparator="placebo", outcome="bp",
                query_strings=[], source_language="en",
                search_language="en", output_language="en",
            ),
            "synthesis_path": synthesis_path,
            "screening_counts": None,
            "manuscript_path": None,
            "template": None,
            "manuscript_draft_version": 0,
        })

    assert "manuscript_rubric" in result
    assert "scores" in result["manuscript_rubric"]


def test_manuscript_citations_come_from_synthesis_claims(db, tmp_path):
    """Citations in the draft are anchored by the verifier from synthesis claims, not hallucinated."""
    db.ensure_run("run-cite")
    synthesis_path = str(tmp_path / "synthesis.md")
    with open(synthesis_path, "w") as f:
        f.write("# Evidence Synthesis\n\n## Grounded Claims\n\n"
                "- Aspirin reduces SBP by 8 mmHg. [99999]\n")

    llm = MockLLM()
    # Writer drafts without any citation syntax
    for section in DEFAULT_PRISMA_TEMPLATE["sections"]:
        llm.register(
            f"write the {section['name'].lower()} section",
            {"text": f"Content of {section['name']} without any citations."},
        )
    # Verifier: matches "Anchor citations" substring
    llm.register("Anchor citations", {
        "anchored": [
            {"claim": "Aspirin reduces SBP by 8 mmHg.",
             "pmids": ["99999"],
             "section": "Results"},
        ]
    })
    # Rubric
    llm.register("score the following systematic review draft", {"scores": []})

    with patch("slr_agent.subgraphs.manuscript.run_pandoc", return_value=None):
        graph = create_manuscript_subgraph(db=db, llm=llm, output_dir=str(tmp_path))
        result = graph.invoke({
            "run_id": "run-cite",
            "pico": PICOResult(
                population="adults with hypertension", intervention="aspirin",
                comparator="placebo", outcome="blood pressure reduction",
                query_strings=[], source_language="en",
                search_language="en", output_language="en",
            ),
            "synthesis_path": synthesis_path,
            "screening_counts": None,
            "manuscript_path": None,
            "template": None,
            "manuscript_draft_version": 0,
        })

    content = open(result["manuscript_path"]).read()
    # PMID should be injected by the verifier pass in the citation tag format
    assert "(PMID: 99999)" in content
