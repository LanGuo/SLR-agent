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


def test_adversarial_review_returns_critique(db, tmp_path):
    """Adversarial reviewer returns structured issues in result."""
    db.ensure_run("run-review")
    synthesis_path = str(tmp_path / "synth.md")
    with open(synthesis_path, "w") as f:
        f.write("# Synthesis\n\nAspirin reduces SBP. [99999]\n")

    llm = MockLLM()
    for section in DEFAULT_PRISMA_TEMPLATE["sections"]:
        llm.register(f"write the {section['name'].lower()} section",
                     {"text": f"Content of {section['name']}."})
    llm.register("Anchor citations", {"anchored": []})
    llm.register("adversarial reviewer", {
        "issues": [
            {"severity": "MAJOR", "section": "Results",
             "issue": "Confidence interval not reported.",
             "suggestion": "Add 95% CI to all effect estimates.",
             "rerun_stage": None},
        ]
    })
    llm.register("score the following systematic review draft", {"scores": []})

    with patch("slr_agent.subgraphs.manuscript.run_pandoc", return_value=None):
        graph = create_manuscript_subgraph(db=db, llm=llm, output_dir=str(tmp_path))
        result = graph.invoke({
            "run_id": "run-review",
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

    assert "adversarial_review" in result
    issues = result["adversarial_review"]["issues"]
    assert len(issues) == 1
    assert issues[0]["severity"] == "MAJOR"


def test_adversarial_review_fatal_sets_rerun_stage(db, tmp_path):
    """FATAL issue with rerun_stage is passed through in result."""
    db.ensure_run("run-fatal")
    synthesis_path = str(tmp_path / "synth.md")
    with open(synthesis_path, "w") as f:
        f.write("# Synthesis\n\nAspirin reduces SBP. [99999]\n")

    llm = MockLLM()
    for section in DEFAULT_PRISMA_TEMPLATE["sections"]:
        llm.register(f"write the {section['name'].lower()} section",
                     {"text": "Content."})
    llm.register("Anchor citations", {"anchored": []})
    llm.register("adversarial reviewer", {
        "issues": [
            {"severity": "FATAL", "section": "Methods",
             "issue": "Inclusion criteria not applied consistently.",
             "suggestion": "Re-screen abstracts with corrected criteria.",
             "rerun_stage": "screening"},
        ]
    })
    llm.register("score the following systematic review draft", {"scores": []})

    with patch("slr_agent.subgraphs.manuscript.run_pandoc", return_value=None):
        graph = create_manuscript_subgraph(db=db, llm=llm, output_dir=str(tmp_path))
        result = graph.invoke({
            "run_id": "run-fatal",
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

    fatal_issues = [i for i in result["adversarial_review"]["issues"]
                    if i["severity"] == "FATAL"]
    assert len(fatal_issues) == 1
    assert fatal_issues[0]["rerun_stage"] == "screening"


def test_adversarial_reviewer_receives_citation_network_warning(db, tmp_path, mock_llm):
    """Citation network warning appears in the adversarial reviewer prompt."""
    from slr_agent.subgraphs.manuscript import create_manuscript_subgraph
    import os

    # Write a synthesis file
    syn_dir = str(tmp_path)
    syn_path = os.path.join(syn_dir, "synthesis.md")
    with open(syn_path, "w") as f:
        f.write("- Aspirin reduces BP. [11111]\n")

    # Register responses for each section in the default template
    from slr_agent.template import DEFAULT_PRISMA_TEMPLATE
    for section in DEFAULT_PRISMA_TEMPLATE["sections"]:
        mock_llm.register(f"write the {section['name'].lower()} section",
                          {"text": f"Content of {section['name']}."})
    mock_llm.register("Anchor citations", {"anchored": []})
    mock_llm.register("adversarial reviewer", {"issues": []})
    mock_llm.register("score the following systematic review draft", {"scores": []})

    sg = create_manuscript_subgraph(db=db, llm=mock_llm, output_dir=str(tmp_path))
    state = {
        "run_id": "test-adv-cn",
        "pico": {"population": "adults", "intervention": "aspirin",
                 "comparator": "placebo", "outcome": "BP",
                 "query_strings": [], "source_language": "en",
                 "search_language": "en", "output_language": "en"},
        "synthesis_path": syn_path,
        "manuscript_draft_version": 1,
        "search_counts": {"n_retrieved": 1, "n_duplicates_removed": 0,
                          "n_pubmed": 1, "n_biorxiv": 0, "n_arxiv": 0},
        "screening_counts": {"n_included": 1, "n_excluded": 0, "n_uncertain": 0},
        "fulltext_counts": None,
        "template": None,
        "config": {},
        "search_sources": ["pubmed"],
        "date_from": "2000-01-01",
        "date_to": "2026-12-31",
        "citation_network": {
            "n_papers": 3,
            "n_cross_citations": 3,
            "echo_chamber_ratio": 1.0,
            "dominant_pmid": "11111",
            "dominant_count": 2,
            "warning": "PMID 11111 is cited by 2/3 included papers. Evidence may not be independent.",
        },
    }

    # The adversarial reviewer prompt should include the warning text.
    # We verify by checking what prompt was passed to the LLM.
    calls = []
    original_chat = mock_llm.chat
    def capturing_chat(messages, schema=None, think=False):
        calls.append(messages)
        return original_chat(messages, schema=schema, think=think)
    mock_llm.chat = capturing_chat

    sg.invoke(state)

    adversarial_calls = [
        msgs for msgs in calls
        if any("adversarial" in (m.get("content") or "").lower() for m in msgs)
    ]
    assert adversarial_calls, "No adversarial reviewer call found"
    adversarial_prompt = " ".join(m.get("content", "") for m in adversarial_calls[0])
    # The citation network warning text must appear in the adversarial reviewer prompt
    assert "CITATION NETWORK ALERT" in adversarial_prompt, (
        "Citation network warning was not passed to the adversarial reviewer prompt"
    )
    assert "PMID 11111 is cited by 2/3 included papers" in adversarial_prompt
