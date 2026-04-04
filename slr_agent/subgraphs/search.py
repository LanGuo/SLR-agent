# slr_agent/subgraphs/search.py
import time
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


_DEFAULT_GRADE = GRADEScore(
    certainty="low", risk_of_bias="high", inconsistency="no",
    indirectness="no", imprecision="no",
    rationale="Not yet assessed",
)

def _search_pubmed_node(state: dict, db: Database) -> dict:
    pico = state["pico"]
    api_key = state.get("pubmed_api_key")
    max_results = state.get("max_results", 500)
    run_id = state["run_id"]
    date_from = state.get("date_from")  # e.g. "2000-01-01"
    date_to = state.get("date_to")      # e.g. "2026-12-31"

    # Ensure the run exists in the DB before inserting papers (FK constraint)
    db.ensure_run(run_id)

    if api_key:
        Entrez.api_key = api_key
    Entrez.email = "slr-agent@local"

    # Collect PMIDs in relevance order per query (PubMed returns best matches first).
    # Use an ordered structure so we can cap the total while preferring top results.
    seen: dict[str, None] = {}  # ordered set via dict keys
    per_query_cap = max(1, max_results // max(len(pico["query_strings"]), 1))
    for query in pico["query_strings"]:
        search_kwargs: dict = {"db": "pubmed", "term": query, "retmax": per_query_cap}
        if date_from and date_to:
            # PubMed expects YYYY/MM/DD format
            search_kwargs["mindate"] = date_from.replace("-", "/")
            search_kwargs["maxdate"] = date_to.replace("-", "/")
            search_kwargs["datetype"] = "pdat"
        handle = Entrez.esearch(**search_kwargs)
        record = Entrez.read(handle)
        handle.close()
        for pmid in record.get("IdList", []):
            seen[pmid] = None
        if not api_key:
            time.sleep(0.34)  # 3 req/s limit without API key

    # Final cap: keep at most max_results unique PMIDs across all queries
    return {"_pubmed_pmids": list(seen)[:max_results]}

def _fetch_pubmed_abstracts_node(state: dict, db: Database) -> dict:
    pmids = state["_pubmed_pmids"]
    run_id = state["run_id"]
    if not pmids:
        return {"_pubmed_count": 0}

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
            abstract = str(abstract_obj.get("AbstractText", "")) if abstract_obj else ""

            db.upsert_paper(PaperRecord(
                pmid=pmid, run_id=run_id, title=title, abstract=abstract,
                fulltext=None, source="abstract",
                screening_decision="uncertain", screening_reason="",
                extracted_data={}, grade_score=_DEFAULT_GRADE,
                provenance=[], quarantined_fields=[],
            ))
        if not state.get("pubmed_api_key"):
            time.sleep(0.34)

    return {"_pubmed_count": len(pmids)}

def _search_biorxiv_node(state: dict, db: Database) -> dict:
    """Fetch recent bioRxiv preprints via their date-range API.

    bioRxiv has no keyword search API — we fetch recent preprints and store them.
    Semantic filtering by PICO relevance happens in the Screening stage (stage 3)
    using Gemma 4, which is the appropriate place for LLM-based filtering.
    """
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
                fulltext=None, source="abstract",
                screening_decision="uncertain", screening_reason="",
                extracted_data={}, grade_score=_DEFAULT_GRADE,
                provenance=[], quarantined_fields=[],
            ))
            biorxiv_count += 1
    except Exception:
        pass  # bioRxiv unavailable — continue with PubMed results only

    return {"_biorxiv_count": biorxiv_count}

def _merge_counts_node(state: dict) -> dict:
    search_sources = state.get("search_sources", ["pubmed"])
    pubmed_count = state.get("_pubmed_count", 0)
    biorxiv_count = state.get("_biorxiv_count", 0) if "biorxiv" in search_sources else 0
    total = pubmed_count + biorxiv_count
    return {
        "search_counts": SearchCounts(
            n_retrieved=total,
            n_duplicates_removed=0,
        )
    }

def _should_search_biorxiv(state: dict) -> str:
    sources = state.get("search_sources", ["pubmed"])
    return "biorxiv" if "biorxiv" in sources else "merge"

def create_search_subgraph(db: Database):
    builder = StateGraph(SearchSubgraphState)

    builder.add_node("search_pubmed", lambda s: _search_pubmed_node(s, db))
    builder.add_node("fetch_pubmed_abstracts", lambda s: _fetch_pubmed_abstracts_node(s, db))
    builder.add_node("search_biorxiv", lambda s: _search_biorxiv_node(s, db))
    builder.add_node("merge_counts", _merge_counts_node)

    builder.set_entry_point("search_pubmed")
    builder.add_edge("search_pubmed", "fetch_pubmed_abstracts")
    builder.add_conditional_edges(
        "fetch_pubmed_abstracts",
        _should_search_biorxiv,
        {"biorxiv": "search_biorxiv", "merge": "merge_counts"},
    )
    builder.add_edge("search_biorxiv", "merge_counts")
    builder.add_edge("merge_counts", END)

    return builder.compile()
