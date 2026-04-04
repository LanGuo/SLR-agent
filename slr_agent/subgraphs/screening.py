# slr_agent/subgraphs/screening.py
from langgraph.graph import StateGraph, END
from slr_agent.db import Database
from slr_agent.grounding import ExtractionGrounder
from slr_agent.state import ScreeningCounts

_BATCH_SIZE = 5    # abstracts per LLM call — smaller batches = shorter prompts = more reliable JSON

def _screen_abstracts_node(state: dict, db: Database, llm) -> dict:
    run_id = state["run_id"]
    pico = state["pico"]
    papers = db.get_all_papers(run_id)
    grounder = ExtractionGrounder()

    n_included = n_excluded = n_uncertain = 0

    # Build criteria text once — same for every batch
    criteria = state.get("screening_criteria") or {}
    criteria_text = ""
    if criteria.get("study_designs"):
        criteria_text += f"Eligible study designs: {', '.join(criteria['study_designs'])}\n"
    if criteria.get("inclusion_criteria"):
        criteria_text += "Inclusion criteria:\n" + "".join(
            f"  - {c}\n" for c in criteria["inclusion_criteria"]
        )
    if criteria.get("exclusion_criteria"):
        criteria_text += "Exclusion criteria:\n" + "".join(
            f"  - {c}\n" for c in criteria["exclusion_criteria"]
        )

    schema = {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "pmid": {"type": "string"},
                        "decision": {"type": "string", "enum": ["include", "exclude", "uncertain"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["pmid", "decision", "reason"],
                },
            }
        },
        "required": ["decisions"],
    }

    for i in range(0, len(papers), _BATCH_SIZE):
        batch = papers[i : i + _BATCH_SIZE]
        batch_text = "\n\n".join(
            f"[PMID {p['pmid']}]\nTitle: {p['title']}\nAbstract: {p['abstract']}"
            for p in batch
        )

        result = llm.chat([{
            "role": "user",
            "content": (
                f"You are screening abstracts for a systematic review.\n"
                f"PICO: P={pico['population']}, I={pico['intervention']}, "
                f"C={pico['comparator']}, O={pico['outcome']}\n\n"
                f"{criteria_text}\n"
                f"For each abstract below, decide include/exclude/uncertain "
                f"and provide a brief reason referencing the abstract text.\n\n{batch_text}"
            ),
        }], schema=schema)

        paper_map = {p["pmid"]: p for p in batch}
        seen_pmids: set[str] = set()
        for decision in result.get("decisions", []):
            pmid = decision["pmid"]
            if pmid in seen_pmids:
                continue
            paper = paper_map.get(pmid)
            if not paper:
                continue
            seen_pmids.add(pmid)
            dec = decision["decision"]
            reason = decision["reason"]

            # Ground the reason against the abstract
            _, quarantined_fields = grounder.ground_extracted_data(
                {"screening_reason": reason},
                source_text=paper["abstract"],
                pmid=pmid,
                source="abstract",
                stage="screening",
            )
            for qf in quarantined_fields:
                db.insert_quarantine(run_id, pmid, qf)

            paper["screening_decision"] = dec
            paper["screening_reason"] = reason
            paper["quarantined_fields"] = quarantined_fields
            db.upsert_paper(paper)

            if dec == "include":
                n_included += 1
            elif dec == "exclude":
                n_excluded += 1
            else:
                n_uncertain += 1

    # Count papers the LLM never returned a decision for (stayed at default "uncertain").
    # These are real gaps — not explicit LLM "uncertain" votes — so surface them in counts.
    n_no_response = sum(
        1 for p in db.get_all_papers(run_id)
        if p["screening_decision"] == "uncertain" and not p.get("screening_reason")
    )
    return {
        "screening_counts": ScreeningCounts(
            n_included=n_included,
            n_excluded=n_excluded,
            n_uncertain=n_uncertain + n_no_response,
        )
    }

def create_screening_subgraph(db: Database, llm):
    builder = StateGraph(dict)
    builder.add_node("screen_abstracts", lambda s: _screen_abstracts_node(s, db, llm))
    builder.set_entry_point("screen_abstracts")
    builder.add_edge("screen_abstracts", END)
    return builder.compile()
