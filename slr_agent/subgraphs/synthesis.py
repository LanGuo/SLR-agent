# slr_agent/subgraphs/synthesis.py
import json
import os
from langgraph.graph import StateGraph, END
from slr_agent.db import Database, QuarantinedField
from slr_agent.prisma import generate_prisma_mermaid


def _synthesise_node(state: dict, db: Database, llm, output_dir: str) -> dict:
    run_id = state["run_id"]
    pico = state["pico"]
    papers = db.get_papers_by_decision(run_id, "include")

    extractions = [
        {"pmid": p["pmid"], **p["extracted_data"]}
        for p in papers
    ]

    result = llm.chat([{
        "role": "user",
        "content": (
            f"synthesise the evidence from these {len(papers)} included studies "
            f"for a systematic review.\n"
            f"Research question: P={pico['population']}, I={pico['intervention']}, "
            f"C={pico['comparator']}, O={pico['outcome']}\n\n"
            f"Extracted data:\n" +
            "\n".join(
                f"[PMID {e['pmid']}]: {json.dumps(e)}"
                for e in extractions
            ) +
            "\n\nReturn JSON with fields:\n"
            "- claims: list of {text, supporting_pmids} — specific grounded claims\n"
            "- narrative: full narrative synthesis paragraph"
        ),
    }], schema={
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "supporting_pmids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["text", "supporting_pmids"],
                },
            },
            "narrative": {"type": "string"},
        },
        "required": ["claims", "narrative"],
    })

    # Claims with no supporting PMIDs are quarantined
    grounded_claims = []
    for claim in result.get("claims", []):
        if claim.get("supporting_pmids"):
            grounded_claims.append({
                "text": claim["text"],
                "supporting_pmids": claim["supporting_pmids"],
            })
        else:
            # "N/A" sentinel: synthesis-level quarantine not tied to a single paper
            db.insert_quarantine(run_id, "N/A", QuarantinedField(
                field_name="claim",
                value=claim["text"],
                stage="synthesis",
                reason="no supporting PMIDs identified by LLM",
            ))

    # Generate PRISMA diagram
    search_counts = state.get("search_counts") or {}
    screening_counts = state.get("screening_counts") or {}
    fulltext_counts = state.get("fulltext_counts")
    extraction_counts = state.get("extraction_counts") or {}

    prisma = generate_prisma_mermaid(
        n_retrieved=search_counts.get("n_retrieved", 0),
        n_duplicates=search_counts.get("n_duplicates_removed", 0),
        n_screened=search_counts.get("n_retrieved", 0) - search_counts.get("n_duplicates_removed", 0),
        n_excluded_abstract=screening_counts.get("n_excluded", 0),
        n_fulltext=fulltext_counts["n_fetched"] if fulltext_counts else None,
        n_excluded_fulltext=fulltext_counts["n_excluded"] if fulltext_counts else None,
        n_included=screening_counts.get("n_included", 0),
        n_quarantined=extraction_counts.get("n_quarantined_fields", 0),
    )

    # Write synthesis file
    os.makedirs(output_dir, exist_ok=True)
    synthesis_path = os.path.join(output_dir, f"{run_id}_synthesis.md")
    with open(synthesis_path, "w") as f:
        f.write("# Evidence Synthesis\n\n")
        f.write(f"## Narrative Summary\n\n{result.get('narrative', '')}\n\n")
        f.write("## Grounded Claims\n\n")
        for c in grounded_claims:
            pmids = ", ".join(c["supporting_pmids"])
            f.write(f"- {c['text']} [{pmids}]\n")
        f.write(f"\n## PRISMA Flow Diagram\n\n{prisma}\n")

    return {"synthesis_path": synthesis_path}


def create_synthesis_subgraph(db: Database, llm, output_dir: str):
    builder = StateGraph(dict)
    builder.add_node("synthesise", lambda s: _synthesise_node(s, db, llm, output_dir))
    builder.set_entry_point("synthesise")
    builder.add_edge("synthesise", END)
    return builder.compile()
