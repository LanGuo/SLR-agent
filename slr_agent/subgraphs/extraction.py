# slr_agent/subgraphs/extraction.py
from langgraph.graph import StateGraph, END
from slr_agent.db import Database, GRADEScore
from slr_agent.grounding import ExtractionGrounder
from slr_agent.state import ExtractionCounts


_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "sample_size": {"type": "string"},
        "intervention": {"type": "string"},
        "comparator": {"type": "string"},
        "primary_outcome": {"type": "string"},
        "result": {"type": "string"},
        "study_design": {"type": "string"},
        "follow_up_duration": {"type": "string"},
        "population_details": {"type": "string"},
    },
    "required": [
        "sample_size", "intervention", "comparator", "primary_outcome",
        "result", "study_design", "follow_up_duration", "population_details",
    ],
}


def _extract_node(state: dict, db: Database, llm) -> dict:
    run_id = state["run_id"]
    pico = state["pico"]
    included = db.get_papers_by_decision(run_id, "include")
    grounder = ExtractionGrounder()

    counts = ExtractionCounts(
        n_extracted=0, n_grade_high=0, n_grade_moderate=0,
        n_grade_low=0, n_grade_very_low=0, n_quarantined_fields=0,
    )

    for paper in included:
        pmid = paper["pmid"]
        source_text = paper.get("fulltext") or paper["abstract"]
        # Derive source type from actual content, not from stored column
        source_type = "fulltext" if paper.get("fulltext") else "abstract"
        extracted = llm.chat([{
            "role": "user",
            "content": (
                f"extract structured data from this paper for a systematic review.\n"
                f"PICO: P={pico['population']}, I={pico['intervention']}, "
                f"C={pico['comparator']}, O={pico['outcome']}\n\n"
                f"Source text:\n{source_text[:6000]}\n\n"
                "Return JSON with fields: sample_size, intervention, comparator, "
                "primary_outcome, result, study_design, follow_up_duration, population_details."
            ),
        }], schema=_EXTRACTION_SCHEMA)

        # Ground extracted fields against source text
        grounded, quarantined_fields = grounder.ground_extracted_data(
            extracted_data={k: v for k, v in extracted.items() if isinstance(v, str)},
            source_text=source_text,
            pmid=pmid,
            source=source_type,
            stage="extraction",
        )
        for qf in quarantined_fields:
            db.insert_quarantine(run_id, pmid, qf)

        # GRADE scoring
        grade_result = llm.chat([{
            "role": "user",
            "content": (
                f"assess the quality of evidence (GRADE) for this paper.\n"
                f"Study design: {extracted.get('study_design', 'unknown')}\n"
                f"Results: {extracted.get('result', 'unknown')}\n\n"
                "Return JSON with fields: certainty (high/moderate/low/very_low), "
                "risk_of_bias (low/some_concerns/high), inconsistency (no/some/serious), "
                "indirectness (no/some/serious), imprecision (no/some/serious), rationale."
            ),
        }], schema={
            "type": "object",
            "properties": {
                "certainty": {"type": "string"},
                "risk_of_bias": {"type": "string"},
                "inconsistency": {"type": "string"},
                "indirectness": {"type": "string"},
                "imprecision": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": ["certainty", "risk_of_bias", "inconsistency",
                         "indirectness", "imprecision", "rationale"],
        })

        grade = GRADEScore(
            certainty=grade_result["certainty"],
            risk_of_bias=grade_result["risk_of_bias"],
            inconsistency=grade_result["inconsistency"],
            indirectness=grade_result["indirectness"],
            imprecision=grade_result["imprecision"],
            rationale=grade_result["rationale"],
        )

        # Only store grounded fields in extracted_data; quarantined fields excluded
        paper["extracted_data"] = {k: v["value"] for k, v in grounded.items()}
        paper["grade_score"] = grade
        paper["provenance"] = [v["span"] for v in grounded.values() if v["span"]]
        # Accumulate quarantine entries from prior stages + this stage
        paper["quarantined_fields"] = list(paper["quarantined_fields"]) + quarantined_fields
        db.upsert_paper(paper)

        counts["n_extracted"] += 1
        counts["n_quarantined_fields"] += len(quarantined_fields)
        grade_key = f"n_grade_{grade['certainty']}"
        if grade_key in counts:
            counts[grade_key] += 1

    return {"extraction_counts": counts}


def create_extraction_subgraph(db: Database, llm):
    builder = StateGraph(dict)
    builder.add_node("extract", lambda s: _extract_node(s, db, llm))
    builder.set_entry_point("extract")
    builder.add_edge("extract", END)
    return builder.compile()
