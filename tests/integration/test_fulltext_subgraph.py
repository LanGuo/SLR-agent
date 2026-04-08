# tests/integration/test_fulltext_subgraph.py
import pytest
from unittest.mock import patch, MagicMock
from slr_agent.subgraphs.fulltext import create_fulltext_subgraph, fetch_pmc_fulltext
from slr_agent.llm import MockLLM


def test_fetch_pmc_fulltext_returns_none_when_unavailable():
    with patch("slr_agent.subgraphs.fulltext.Entrez") as mock_entrez:
        mock_entrez.elink.side_effect = Exception("Not in PMC")
        fulltext, pmc_id = fetch_pmc_fulltext("99999")
    assert fulltext is None
    assert pmc_id is None


def test_fulltext_subgraph_falls_back_to_abstract(db, sample_paper, mock_llm):
    db.ensure_run("run-test")
    db.upsert_paper(sample_paper)

    mock_llm.register("screen this full text", {
        "decision": "include",
        "reason": "Full-text confirms RCT methodology and relevant population",
    })

    from slr_agent.state import PICOResult
    pico = PICOResult(
        population="adults with hypertension", intervention="aspirin",
        comparator="placebo", outcome="blood pressure reduction",
        query_strings=[], source_language="en",
        search_language="en", output_language="en",
    )

    # fetch_pmc_fulltext now returns (fulltext, pmc_id) — mock returns (None, None)
    with patch("slr_agent.subgraphs.fulltext.fetch_pmc_fulltext", return_value=(None, None)):
        graph = create_fulltext_subgraph(db=db, llm=mock_llm)
        result = graph.invoke({
            "run_id": "run-test",
            "pico": pico,
            "fulltext_counts": None,
        })

    # Paper should remain with source="abstract" since PMC fetch returned None
    paper = db.get_paper("run-test", "99999")
    assert paper["source"] == "abstract"
    assert result["fulltext_counts"]["n_unavailable"] == 1
