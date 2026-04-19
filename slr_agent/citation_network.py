"""Lightweight within-corpus citation graph for evidence independence analysis.

After full-text fetch, included papers' reference lists are parsed and
intersected with the corpus to detect circular citation patterns and
evidence inflation.
"""
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict


@dataclass
class CitationNetworkSummary:
    """Metrics describing within-corpus citation patterns."""
    n_papers: int
    n_cross_citations: int      # total within-corpus citation edges
    echo_chamber_ratio: float   # fraction of papers that cite ≥1 other corpus paper
    dominant_pmid: str | None   # most-cited corpus paper (None if no cross-citations)
    dominant_count: int         # how many corpus papers cite dominant_pmid
    warning: str | None         # human-readable flag; None if no concern

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_cited_pmids_from_xml(xml_str: str | None) -> list[str]:
    """Return PMIDs referenced in a PubMed XML full-text string.

    Parses all <ArticleId IdType="pubmed"> elements anywhere in the document.
    Returns an empty list on parse error or None input.
    """
    if not xml_str:
        return []
    try:
        root = ET.fromstring(xml_str)
        pmids: list[str] = []
        for el in root.iter("ArticleId"):
            if el.get("IdType") == "pubmed":
                text = (el.text or "").strip()
                if text.isdigit():
                    pmids.append(text)
        return pmids
    except ET.ParseError:
        return []


def build_citation_network(papers: list[dict]) -> CitationNetworkSummary:
    """Build a within-corpus citation graph from full-text XML reference lists.

    Args:
        papers: List of PaperRecord dicts. Only papers with a non-None
                ``fulltext`` field contribute edges. ``pmid`` is required.

    Returns:
        CitationNetworkSummary with cross-citation counts and a warning
        string when concerning patterns are detected.
    """
    n = len(papers)
    if n == 0:
        return CitationNetworkSummary(
            n_papers=0, n_cross_citations=0, echo_chamber_ratio=0.0,
            dominant_pmid=None, dominant_count=0, warning=None,
        )

    corpus_pmids = {p["pmid"] for p in papers}
    # cited_by[pmid] = number of other corpus papers that cite pmid
    cited_by: dict[str, int] = {pmid: 0 for pmid in corpus_pmids}
    n_papers_citing_corpus = 0
    n_cross_total = 0

    for paper in papers:
        fulltext = paper.get("fulltext")
        if not fulltext:
            continue
        cited = _extract_cited_pmids_from_xml(fulltext)
        within_corpus = [p for p in cited if p in corpus_pmids and p != paper["pmid"]]
        if within_corpus:
            n_papers_citing_corpus += 1
        for pmid in within_corpus:
            cited_by[pmid] += 1
            n_cross_total += 1

    echo_ratio = n_papers_citing_corpus / n

    # Identify the most-cited corpus paper
    dominant_pmid = max(cited_by, key=cited_by.get) if cited_by else None
    dominant_count = cited_by.get(dominant_pmid, 0) if dominant_pmid else 0
    # Suppress dominant_pmid if it has zero citations (no cross-citations at all)
    if dominant_count == 0:
        dominant_pmid = None

    warning = None
    if dominant_pmid and dominant_count / n > 0.5:
        pct = f"{dominant_count / n:.0%}"
        warning = (
            f"PMID {dominant_pmid} is cited by {dominant_count}/{n} included papers "
            f"({pct}). Evidence base may not be independent — multiple papers may "
            f"derive from the same original source."
        )
    elif echo_ratio > 0.5:
        pct = f"{echo_ratio:.0%}"
        warning = (
            f"{n_papers_citing_corpus}/{n} included papers ({pct}) cite at least one "
            f"other included paper. Check for circular citation patterns that could "
            f"inflate apparent evidence volume."
        )

    return CitationNetworkSummary(
        n_papers=n,
        n_cross_citations=n_cross_total,
        echo_chamber_ratio=echo_ratio,
        dominant_pmid=dominant_pmid,
        dominant_count=dominant_count,
        warning=warning,
    )
