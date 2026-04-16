# slr_agent/subgraphs/search.py
import re
import time
import xml.etree.ElementTree as ET
import httpx
from typing import Optional
from typing_extensions import TypedDict
from Bio import Entrez
from langgraph.graph import StateGraph, END
from slr_agent.db import Database, PaperRecord, GRADEScore
from slr_agent.state import SearchCounts, PICOResult


class SearchSubgraphState(TypedDict, total=False):
    run_id: str
    pico: PICOResult
    search_counts: Optional[SearchCounts]
    pubmed_api_key: Optional[str]
    max_results: int
    search_sources: list[str]
    # Internal intermediate keys
    _pubmed_pmids: list[str]
    _pubmed_count: int
    _biorxiv_count: int
    _arxiv_count: int


_DEFAULT_GRADE = GRADEScore(
    certainty="low", risk_of_bias="high", inconsistency="no",
    indirectness="no", imprecision="no",
    rationale="Not yet assessed",
)

def _esearch(term: str, retmax: int, date_from: str | None, date_to: str | None) -> list[str]:
    """Run a single PubMed esearch and return the PMID list."""
    kwargs: dict = {"db": "pubmed", "term": term, "retmax": retmax}
    if date_from and date_to:
        kwargs["mindate"] = date_from.replace("-", "/")
        kwargs["maxdate"] = date_to.replace("-", "/")
        kwargs["datetype"] = "pdat"
    handle = Entrez.esearch(**kwargs)
    record = Entrez.read(handle)
    handle.close()
    return record.get("IdList", [])


def _search_pubmed_node(state: dict, db: Database) -> dict:
    pico = state["pico"]
    api_key = state.get("pubmed_api_key")
    max_results = state.get("max_results", 500)
    run_id = state["run_id"]
    date_from = state.get("date_from")
    date_to = state.get("date_to")

    db.ensure_run(run_id)

    if api_key:
        Entrez.api_key = api_key
    Entrez.email = "slr-agent@local"

    # Collect PMIDs in relevance order per query (PubMed returns best matches first).
    # per_query_cap uses 2× headroom to compensate for overlap between PICO queries.
    seen: dict[str, None] = {}  # ordered set via dict keys
    n_queries = max(len(pico["query_strings"]), 1)
    per_query_cap = max(50, max_results // n_queries * 2)

    for query in pico["query_strings"]:
        for pmid in _esearch(query, per_query_cap, date_from, date_to):
            seen[pmid] = None
        if not api_key:
            time.sleep(0.34)

    # Fallback: if the LLM-generated queries returned very few results, try a simple
    # keyword-only query built directly from PICO terms. This guards against overly
    # restrictive queries with broken MeSH syntax or excessive Boolean complexity.
    fallback_threshold = max(10, max_results // 10)
    if len(seen) < fallback_threshold:
        import warnings
        warnings.warn(
            f"PICO queries returned only {len(seen)} PMIDs (threshold {fallback_threshold}). "
            "Running broad fallback query.",
            RuntimeWarning,
            stacklevel=2,
        )
        fallback = (
            f"({pico['intervention']}) AND ({pico['outcome']}) AND ({pico['population']})"
        )
        for pmid in _esearch(fallback, max_results, date_from, date_to):
            seen[pmid] = None
        if not api_key:
            time.sleep(0.34)

    return {"_pubmed_pmids": list(seen)[:max_results]}

def _fetch_pubmed_abstracts_node(state: dict, db: Database) -> dict:
    pmids = state["_pubmed_pmids"]
    run_id = state["run_id"]
    if not pmids:
        return {"_pubmed_count": 0}

    stored = 0
    batch_size = 200
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        handle = Entrez.efetch(
            db="pubmed", id=",".join(batch), rettype="xml", retmode="xml"
        )
        records = Entrez.read(handle)
        handle.close()

        for article in records.get("PubmedArticle", []):
            citation = article["MedlineCitation"]
            pmid = str(citation["PMID"])
            art = citation.get("Article", {})
            title = str(art.get("ArticleTitle", ""))
            abstract_obj = art.get("Abstract", {})
            abstract_raw = abstract_obj.get("AbstractText", "") if abstract_obj else ""
            # AbstractText is a list of StringElements for structured abstracts
            # (those with labeled sections like PURPOSE/METHODS/RESULTS)
            abstract = (
                " ".join(str(s) for s in abstract_raw)
                if isinstance(abstract_raw, list)
                else str(abstract_raw)
            )

            db.upsert_paper(PaperRecord(
                pmid=pmid, run_id=run_id, title=title, abstract=abstract,
                fulltext=None, page_image_paths=[], source="abstract",
                screening_decision="uncertain", screening_reason="",
                extracted_data={}, grade_score=_DEFAULT_GRADE,
                provenance=[], quarantined_fields=[],
            ))
            stored += 1
        if not state.get("pubmed_api_key"):
            time.sleep(0.34)

    return {"_pubmed_count": stored}

def _search_biorxiv_node(state: dict, db: Database) -> dict:
    """Fetch recent bioRxiv preprints via their date-range API.

    bioRxiv has no keyword search API — we fetch recent preprints and store them.
    Semantic filtering by PICO relevance happens in the Screening stage (stage 3)
    using Gemma 4, which is the appropriate place for LLM-based filtering.

    Self-gating: returns immediately if "biorxiv" is not in search_sources.
    """
    if "biorxiv" not in state.get("search_sources", []):
        return {"_biorxiv_count": 0}

    run_id = state["run_id"]
    max_results = state.get("max_results", 500)
    date_from = state.get("date_from", "2000-01-01")
    date_to = state.get("date_to") or "2099-12-31"
    biorxiv_count = 0

    try:
        resp = httpx.get(
            f"https://api.biorxiv.org/details/biorxiv/{date_from}/{date_to}/0/json",
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("collection", [])[:max_results]:
            doi = item.get("doi", "")
            title = item.get("title", "")
            abstract = item.get("abstract", "")
            if not doi:
                continue
            pmid_proxy = f"biorxiv:{doi}"
            db.upsert_paper(PaperRecord(
                pmid=pmid_proxy, run_id=run_id, title=title, abstract=abstract,
                fulltext=None, page_image_paths=[], source="abstract",
                screening_decision="uncertain", screening_reason="",
                extracted_data={}, grade_score=_DEFAULT_GRADE,
                provenance=[], quarantined_fields=[],
            ))
            biorxiv_count += 1
    except Exception as exc:
        import warnings
        warnings.warn(f"bioRxiv search failed: {exc}", RuntimeWarning, stacklevel=2)

    return {"_biorxiv_count": biorxiv_count}


_ARXIV_NS = "http://www.w3.org/2005/Atom"
_ARXIV_RATE_LIMIT_S = 3.0  # arXiv API terms require ≥3 s between requests


def _build_arxiv_query(query_strings: list[str]) -> str:
    """Convert PICO query strings (PubMed syntax) to arXiv all: keyword queries.

    Strips PubMed-specific syntax ([MeSH], [tiab], etc.) and joins with OR.
    """
    clean_terms = []
    for qs in query_strings:
        # Strip field tags like [MeSH Terms], [tiab], [Text Word], etc.
        cleaned = re.sub(r"\[[^\]]+\]", "", qs)
        # Collapse whitespace
        cleaned = " ".join(cleaned.split())
        if cleaned:
            clean_terms.append(f"all:{cleaned}")
    return " OR ".join(clean_terms) if clean_terms else "all:biomedical"


def _search_arxiv_node(state: dict, db: Database) -> dict:
    """Search arXiv preprints via the arXiv Atom API.

    Uses PICO query strings (with PubMed syntax stripped) to keyword-search arXiv.
    Self-gating: returns immediately if "arxiv" is not in search_sources.

    pmid proxy format: arxiv:{arxiv_id}  (e.g. arxiv:2301.12345)
    """
    if "arxiv" not in state.get("search_sources", []):
        return {"_arxiv_count": 0}

    run_id = state["run_id"]
    pico = state["pico"]
    max_results = state.get("max_results", 500)
    arxiv_count = 0

    query = _build_arxiv_query(pico.get("query_strings", []))

    try:
        resp = httpx.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": query, "start": 0, "max_results": max_results},
            timeout=60,
        )
        resp.raise_for_status()
        time.sleep(_ARXIV_RATE_LIMIT_S)

        root = ET.fromstring(resp.text)
        for entry in root.findall(f"{{{_ARXIV_NS}}}entry"):
            # arXiv ID: strip URL prefix and version suffix
            id_elem = entry.find(f"{{{_ARXIV_NS}}}id")
            if id_elem is None or not id_elem.text:
                continue
            raw_id = id_elem.text.strip()
            # e.g. "http://arxiv.org/abs/2301.12345v2" → "2301.12345"
            arxiv_id = re.sub(r"v\d+$", "", raw_id.split("/abs/")[-1])

            title_elem = entry.find(f"{{{_ARXIV_NS}}}title")
            title = " ".join((title_elem.text or "").split()) if title_elem is not None else ""

            summary_elem = entry.find(f"{{{_ARXIV_NS}}}summary")
            abstract = " ".join((summary_elem.text or "").split()) if summary_elem is not None else ""

            pmid_proxy = f"arxiv:{arxiv_id}"
            db.upsert_paper(PaperRecord(
                pmid=pmid_proxy, run_id=run_id, title=title, abstract=abstract,
                fulltext=None, page_image_paths=[], source="abstract",
                screening_decision="uncertain", screening_reason="",
                extracted_data={}, grade_score=_DEFAULT_GRADE,
                provenance=[], quarantined_fields=[],
            ))
            arxiv_count += 1
    except Exception as exc:
        import warnings
        warnings.warn(f"arXiv search failed: {exc}", RuntimeWarning, stacklevel=2)

    return {"_arxiv_count": arxiv_count}

def _merge_counts_node(state: dict) -> dict:
    pubmed_count = state.get("_pubmed_count", 0)
    biorxiv_count = state.get("_biorxiv_count", 0)
    arxiv_count = state.get("_arxiv_count", 0)
    total = pubmed_count + biorxiv_count + arxiv_count
    return {
        "search_counts": SearchCounts(
            n_retrieved=total,
            n_duplicates_removed=0,
            n_pubmed=pubmed_count,
            n_biorxiv=biorxiv_count,
            n_arxiv=arxiv_count,
        )
    }

def create_search_subgraph(db: Database):
    builder = StateGraph(SearchSubgraphState)

    builder.add_node("search_pubmed", lambda s: _search_pubmed_node(s, db))
    builder.add_node("fetch_pubmed_abstracts", lambda s: _fetch_pubmed_abstracts_node(s, db))
    # Both source nodes are self-gating: they check search_sources internally.
    builder.add_node("search_biorxiv", lambda s: _search_biorxiv_node(s, db))
    builder.add_node("search_arxiv", lambda s: _search_arxiv_node(s, db))
    builder.add_node("merge_counts", _merge_counts_node)

    builder.set_entry_point("search_pubmed")
    builder.add_edge("search_pubmed", "fetch_pubmed_abstracts")
    builder.add_edge("fetch_pubmed_abstracts", "search_biorxiv")
    builder.add_edge("search_biorxiv", "search_arxiv")
    builder.add_edge("search_arxiv", "merge_counts")
    builder.add_edge("merge_counts", END)

    return builder.compile()
