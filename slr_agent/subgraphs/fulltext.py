# slr_agent/subgraphs/fulltext.py
import time
from Bio import Entrez
from langgraph.graph import StateGraph, END
from slr_agent.db import Database
from slr_agent.grounding import ExtractionGrounder
from slr_agent.state import FulltextCounts


def fetch_pmc_fulltext(pmid: str) -> str | None:
    """Fetch full text from PubMed Central. Returns text or None if unavailable."""
    try:
        # First get PMC ID
        handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid)
        record = Entrez.read(handle)
        handle.close()
        link_sets = record[0].get("LinkSetDb", [])
        if not link_sets:
            return None
        pmc_ids = [lnk["Id"] for lnk in link_sets[0].get("Link", [])]
        if not pmc_ids:
            return None
        pmc_id = pmc_ids[0]

        # Fetch full text XML
        handle = Entrez.efetch(db="pmc", id=pmc_id, rettype="full", retmode="xml")
        text = handle.read()
        handle.close()
        return text.decode("utf-8") if isinstance(text, bytes) else text
    except Exception:
        return None


def _fetch_fulltext_node(state: dict, db: Database, llm) -> dict:
    run_id = state["run_id"]
    pico = state["pico"]
    included = db.get_papers_by_decision(run_id, "include")
    grounder = ExtractionGrounder()

    n_fetched = n_unavailable = n_excluded = 0

    for paper in included:
        pmid = paper["pmid"]
        fulltext = fetch_pmc_fulltext(pmid)

        if fulltext is None:
            n_unavailable += 1
            time.sleep(0.34)
            continue

        time.sleep(0.34)

        # Screen full text
        result = llm.chat([{
            "role": "user",
            "content": (
                f"Screen this full text for a systematic review.\n"
                f"PICO: P={pico['population']}, I={pico['intervention']}, "
                f"C={pico['comparator']}, O={pico['outcome']}\n\n"
                f"Full text (first 8000 chars):\n{fulltext[:8000]}\n\n"
                "Return JSON with fields: decision (include/exclude/uncertain), reason."
            ),
        }], schema={
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": ["include", "exclude", "uncertain"]},
                "reason": {"type": "string"},
            },
            "required": ["decision", "reason"],
        })

        dec = result["decision"]
        reason = result["reason"]

        # Ground reason against full text
        _, quarantined_fields = grounder.ground_extracted_data(
            {"screening_reason": reason},
            source_text=fulltext[:8000],
            pmid=pmid,
            source="fulltext",
            stage="fulltext_screening",
        )
        for qf in quarantined_fields:
            db.insert_quarantine(run_id, pmid, qf)

        paper["fulltext"] = fulltext
        paper["source"] = "fulltext"
        paper["screening_decision"] = dec
        paper["screening_reason"] = reason
        paper["quarantined_fields"] = quarantined_fields
        db.upsert_paper(paper)

        n_fetched += 1
        if dec == "exclude":
            n_excluded += 1

    return {
        "fulltext_counts": FulltextCounts(
            n_fetched=n_fetched,
            n_unavailable=n_unavailable,
            n_excluded=n_excluded,
        )
    }


def create_fulltext_subgraph(db: Database, llm):
    builder = StateGraph(dict)
    builder.add_node("fetch_fulltext", lambda s: _fetch_fulltext_node(s, db, llm))
    builder.set_entry_point("fetch_fulltext")
    builder.add_edge("fetch_fulltext", END)
    return builder.compile()
