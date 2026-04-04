# tests/integration/test_manuscript_subgraph.py
import os
import pytest
from unittest.mock import patch
from slr_agent.subgraphs.manuscript import create_manuscript_subgraph
from slr_agent.llm import MockLLM


def test_manuscript_writes_markdown(db, sample_paper, mock_llm, tmp_path):
    db.ensure_run("run-test")
    db.upsert_paper(sample_paper)
    synthesis_path = str(tmp_path / "synthesis.md")
    with open(synthesis_path, "w") as f:
        f.write("# Synthesis\n\nAspirin reduces SBP by 8 mmHg.")

    mock_llm.register("write the Methods section", {"text": "We searched PubMed using PICO-derived queries."})
    mock_llm.register("write the Results section", {"text": "One RCT (n=500) was included."})
    mock_llm.register("write the Discussion section", {"text": "Aspirin appears effective for hypertension."})

    with patch("slr_agent.subgraphs.manuscript.run_pandoc", return_value=None):
        graph = create_manuscript_subgraph(db=db, llm=mock_llm, output_dir=str(tmp_path))
        from slr_agent.state import PICOResult
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
        })

    assert result["manuscript_path"] is not None
    assert os.path.exists(result["manuscript_path"])
    content = open(result["manuscript_path"]).read()
    assert "Methods" in content
    assert "Results" in content
