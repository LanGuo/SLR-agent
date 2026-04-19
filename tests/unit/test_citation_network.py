# tests/unit/test_citation_network.py
import pytest
from slr_agent.citation_network import (
    build_citation_network,
    CitationNetworkSummary,
    _extract_cited_pmids_from_xml,
)


def _make_xml(*cited_pmids: str) -> str:
    """Minimal PubMed XML with a reference list citing the given PMIDs."""
    refs = "".join(
        f"""<Reference>
          <ArticleIdList>
            <ArticleId IdType="pubmed">{pmid}</ArticleId>
          </ArticleIdList>
        </Reference>"""
        for pmid in cited_pmids
    )
    return f"""<?xml version="1.0"?>
    <PubmedArticleSet><PubmedArticle>
      <PubmedData><ReferenceList>{refs}</ReferenceList></PubmedData>
    </PubmedArticle></PubmedArticleSet>"""


def test_extract_pmids_from_valid_xml():
    xml = _make_xml("11111111", "22222222")
    pmids = _extract_cited_pmids_from_xml(xml)
    assert set(pmids) == {"11111111", "22222222"}


def test_extract_pmids_ignores_non_pubmed_ids():
    xml = """<root>
      <ArticleId IdType="doi">10.1001/foo</ArticleId>
      <ArticleId IdType="pubmed">99999999</ArticleId>
      <ArticleId IdType="pmc">PMC123456</ArticleId>
    </root>"""
    pmids = _extract_cited_pmids_from_xml(xml)
    assert pmids == ["99999999"]


def test_extract_pmids_returns_empty_on_bad_xml():
    pmids = _extract_cited_pmids_from_xml("this is not xml <<<")
    assert pmids == []


def test_extract_pmids_returns_empty_on_none():
    pmids = _extract_cited_pmids_from_xml(None)
    assert pmids == []


def test_build_citation_network_no_fulltext():
    """Papers without fulltext contribute zero edges."""
    papers = [
        {"pmid": "111", "fulltext": None},
        {"pmid": "222", "fulltext": None},
        {"pmid": "333", "fulltext": None},
    ]
    summary = build_citation_network(papers)
    assert summary.n_papers == 3
    assert summary.n_cross_citations == 0
    assert summary.echo_chamber_ratio == 0.0
    assert summary.dominant_pmid is None
    assert summary.warning is None


def test_build_citation_network_counts_cross_citations():
    """Papers citing each other are counted as cross-citations."""
    papers = [
        {"pmid": "111", "fulltext": _make_xml("222")},   # 111 cites 222
        {"pmid": "222", "fulltext": _make_xml("111")},   # 222 cites 111
        {"pmid": "333", "fulltext": None},
    ]
    summary = build_citation_network(papers)
    assert summary.n_cross_citations == 2
    assert summary.echo_chamber_ratio == pytest.approx(2 / 3)


def test_build_citation_network_excludes_self_citations():
    """A paper citing itself must not be counted."""
    papers = [
        {"pmid": "111", "fulltext": _make_xml("111", "222")},
        {"pmid": "222", "fulltext": None},
    ]
    summary = build_citation_network(papers)
    assert summary.n_cross_citations == 1  # only 111→222, not 111→111


def test_build_citation_network_detects_dominant_paper():
    """When one paper is cited by >50% of the corpus, a warning is raised."""
    papers = [
        {"pmid": "111", "fulltext": _make_xml("333")},
        {"pmid": "222", "fulltext": _make_xml("333")},
        {"pmid": "333", "fulltext": None},
    ]
    summary = build_citation_network(papers)
    assert summary.dominant_pmid == "333"
    assert summary.dominant_count == 2
    assert summary.warning is not None
    assert "333" in summary.warning


def test_build_citation_network_no_warning_when_citations_spread():
    """No warning when neither dominant-paper nor echo-chamber threshold is met.

    Only 1/3 papers cites another corpus paper (echo_ratio=0.33 ≤ 0.5) and
    no single paper is cited by >50% of the corpus (dominant_count/n=0.33 ≤ 0.5).
    """
    papers = [
        {"pmid": "111", "fulltext": _make_xml("222")},  # 111 cites 222
        {"pmid": "222", "fulltext": None},
        {"pmid": "333", "fulltext": None},
    ]
    summary = build_citation_network(papers)
    assert summary.warning is None


def test_build_citation_network_empty_corpus():
    summary = build_citation_network([])
    assert summary.n_papers == 0
    assert summary.n_cross_citations == 0
    assert summary.warning is None


def test_citation_network_summary_to_dict():
    """CitationNetworkSummary.to_dict() produces JSON-serializable output."""
    import dataclasses
    papers = [{"pmid": "111", "fulltext": _make_xml("222")}, {"pmid": "222", "fulltext": None}]
    summary = build_citation_network(papers)
    d = summary.to_dict()
    assert isinstance(d, dict)
    assert d["n_papers"] == 2
    assert d["n_cross_citations"] == 1
    import json
    json.dumps(d)  # must not raise


def test_build_citation_network_warns_on_high_echo_chamber_ratio():
    """Warning fires when >50% of papers cite at least one other corpus paper,
    even when no single paper accounts for >50% of citations."""
    # Each paper cites one other — echo_ratio=1.0, but dominant_count/n = 1/3
    papers = [
        {"pmid": "111", "fulltext": _make_xml("222")},
        {"pmid": "222", "fulltext": _make_xml("333")},
        {"pmid": "333", "fulltext": _make_xml("111")},
    ]
    summary = build_citation_network(papers)
    assert summary.echo_chamber_ratio == pytest.approx(1.0)
    assert summary.warning is not None
