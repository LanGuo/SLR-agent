# slr_agent/subgraphs/screening.py
from langgraph.graph import StateGraph, END
from slr_agent.db import Database
from slr_agent.grounding import ExtractionGrounder
from slr_agent.state import ScreeningCounts

_BATCH_SIZE = 20   # abstracts per LLM call

def _screen_abstracts_node(state: dict, db: Database, llm) -> dict:
    run_id = state["run_id"]
    pico = state["pico"]
    papers = db.get_all_papers(run_id)
    grounder = ExtractionGrounder(threshold=85)

    n_included = n_excluded = n_uncertain = 0

    for i in range(0, len(papers), _BATCH_SIZE):
        batch = papers[i : i + _BATCH_SIZE]
        batch_text = "\n\n".join(
            f"[PMID {p['pmid']}]\nTitle: {p['title']}\nAbstract: {p['abstract']}"
            for p in batch
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
        result = llm.chat([{
            "role": "user",
            "content": (
                f"You are screening abstracts for a systematic review.\n"
                f"PICO: P={pico['population']}, I={pico['intervention']}, "
                f"C={pico['comparator']}, O={pico['outcome']}\n\n"
                f"Please screen the following abstracts. For each, decide include/exclude/uncertain "
                f"and provide a brief reason referencing the abstract text.\n\n{batch_text}"
            ),
        }], schema=schema)

        paper_map = {p["pmid"]: p for p in batch}
        for decision in result.get("decisions", []):
            pmid = decision["pmid"]
            paper = paper_map.get(pmid)
            if not paper:
                continue
            dec = decision["decision"]
            reason = decision["reason"]

            # Ground the reason against the abstract
            grounded, quarantined_fields = grounder.ground_extracted_data(
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
            paper["quarantined_fields"] = list(paper["quarantined_fields"]) + quarantined_fields
            db.upsert_paper(paper)

            if dec == "include":
                n_included += 1
            elif dec == "exclude":
                n_excluded += 1
            else:
                n_uncertain += 1

    return {
        "screening_counts": ScreeningCounts(
            n_included=n_included,
            n_excluded=n_excluded,
            n_uncertain=n_uncertain,
        )
    }

def create_screening_subgraph(db: Database, llm):
    builder = StateGraph(dict)
    builder.add_node("screen_abstracts", lambda s: _screen_abstracts_node(s, db, llm))
    builder.set_entry_point("screen_abstracts")
    builder.add_edge("screen_abstracts", END)
    return builder.compile()
