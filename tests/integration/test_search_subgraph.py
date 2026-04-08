# tests/integration/test_search_subgraph.py
import pytest
from unittest.mock import patch, MagicMock
from slr_agent.subgraphs.search import create_search_subgraph
from slr_agent.state import PICOResult

MOCK_PMIDS = ["11111", "22222", "33333"]

MOCK_FETCH_RESULT = {
    "PubmedArticle": [
        {
            "MedlineCitation": {
                "PMID": {"#text": pmid},
                "Article": {
                    "ArticleTitle": f"Paper {pmid}",
                    "Abstract": {"AbstractText": f"Abstract for {pmid}."},
                },
            }
        }
        for pmid in MOCK_PMIDS
    ]
}

MOCK_BIORXIV_RESPONSE = {
    "collection": [
        {
            "doi": "10.1101/2024.01.01.000001",
            "title": "bioRxiv Preprint on Hypertension",
            "abstract": "A preprint about hypertension treatment.",
        }
    ]
}

@pytest.fixture
def pico():
    return PICOResult(
        population="adults with hypertension",
        intervention="aspirin",
        comparator="placebo",
        outcome="blood pressure reduction",
        query_strings=["hypertension[MeSH]"],
        source_language="en",
        search_language="en",
        output_language="en",
    )

def test_search_pubmed_only(db, pico):
    with patch("slr_agent.subgraphs.search.Entrez") as mock_entrez:
        mock_entrez.esearch.return_value = MagicMock()
        mock_entrez.read.side_effect = [
            {"IdList": MOCK_PMIDS},   # esearch: main query
            {"IdList": []},           # esearch: fallback query (fires when results < threshold)
            MOCK_FETCH_RESULT,        # efetch: batch abstract fetch
        ]
        mock_entrez.efetch.return_value = MagicMock()
        graph = create_search_subgraph(db=db)
        result = graph.invoke({
            "run_id": "run-test",
            "pico": pico,
            "search_counts": None,
            "pubmed_api_key": None,
            "max_results": 500,
            "search_sources": ["pubmed"],
        })

    assert result["search_counts"]["n_retrieved"] == 3
    papers = db.get_all_papers("run-test")
    assert len(papers) == 3
    assert papers[0]["title"] in ["Paper 11111", "Paper 22222", "Paper 33333"]

def test_search_biorxiv_adds_preprints(db, pico):
    with patch("slr_agent.subgraphs.search.Entrez") as mock_entrez, \
         patch("slr_agent.subgraphs.search.httpx") as mock_httpx:
        mock_entrez.esearch.return_value = MagicMock()
        mock_entrez.read.side_effect = [{"IdList": MOCK_PMIDS}, {"IdList": []}, MOCK_FETCH_RESULT]
        mock_entrez.efetch.return_value = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = MOCK_BIORXIV_RESPONSE
        mock_httpx.get.return_value = mock_resp

        graph = create_search_subgraph(db=db)
        result = graph.invoke({
            "run_id": "run-biorxiv",
            "pico": pico,
            "search_counts": None,
            "pubmed_api_key": None,
            "max_results": 500,
            "search_sources": ["pubmed", "biorxiv"],
        })

    papers = db.get_all_papers("run-biorxiv")
    pmids = [p["pmid"] for p in papers]
    assert any(p.startswith("biorxiv:") for p in pmids)
    assert result["search_counts"]["n_retrieved"] == 4  # 3 PubMed + 1 bioRxiv

def test_search_biorxiv_failure_does_not_crash(db, pico):
    """bioRxiv unavailability must not fail the pipeline."""
    with patch("slr_agent.subgraphs.search.Entrez") as mock_entrez, \
         patch("slr_agent.subgraphs.search.httpx") as mock_httpx:
        mock_entrez.esearch.return_value = MagicMock()
        mock_entrez.read.side_effect = [
            {"IdList": ["99999"]},   # esearch: main query
            {"IdList": []},          # esearch: fallback query
            {                        # efetch: batch abstract fetch
                "PubmedArticle": [{
                    "MedlineCitation": {
                        "PMID": {"#text": "99999"},
                        "Article": {"ArticleTitle": "PubMed paper", "Abstract": {"AbstractText": "abstract"}},
                    }
                }]
            },
        ]
        mock_entrez.efetch.return_value = MagicMock()
        mock_httpx.get.side_effect = Exception("Connection refused")

        graph = create_search_subgraph(db=db)
        result = graph.invoke({
            "run_id": "run-biorxiv-fail",
            "pico": pico,
            "search_counts": None,
            "pubmed_api_key": None,
            "max_results": 500,
            "search_sources": ["pubmed", "biorxiv"],
        })

    # Should complete with PubMed results only
    assert result["search_counts"]["n_retrieved"] >= 1
    papers = db.get_all_papers("run-biorxiv-fail")
    assert len(papers) >= 1
