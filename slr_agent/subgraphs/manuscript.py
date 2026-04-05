# slr_agent/subgraphs/manuscript.py
import os
from langgraph.graph import StateGraph, END
from slr_agent.db import Database
from slr_agent.export import run_pandoc
from slr_agent.template import DEFAULT_PRISMA_TEMPLATE, score_rubric

_TEXT_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}


def _draft_manuscript_node(state: dict, db: Database, llm, output_dir: str) -> dict:
    run_id = state["run_id"]
    pico = state["pico"]
    synthesis_path = state.get("synthesis_path")
    output_language = pico.get("output_language", "en")
    template = state.get("template") or DEFAULT_PRISMA_TEMPLATE
    draft_version = state.get("manuscript_draft_version", 0) + 1

    synthesis_text = ""
    if synthesis_path and os.path.exists(synthesis_path):
        with open(synthesis_path) as f:
            synthesis_text = f.read()

    papers = db.get_papers_by_decision(run_id, "include")
    screening = state.get("screening_counts") or {}
    search = state.get("search_counts") or {}
    lang_suffix = f" Write in {output_language}." if output_language != "en" else ""

    # Build a factual search context block from actual run state so the LLM
    # does not hallucinate databases, dates, or PRISMA counts.
    sources = state.get("search_sources") or ["pubmed"]
    source_labels = {"pubmed": "PubMed/MEDLINE", "biorxiv": "bioRxiv"}
    sources_str = ", ".join(source_labels.get(s, s) for s in sources)
    date_from = state.get("date_from") or "2000-01-01"
    date_to = state.get("date_to") or "present"
    query_strings = (pico.get("query_strings") or [])
    queries_str = "\n".join(f"  - {q}" for q in query_strings)
    screening_criteria = state.get("screening_criteria") or {}
    inclusion_str = "; ".join(screening_criteria.get("inclusion_criteria") or [])
    exclusion_str = "; ".join(screening_criteria.get("exclusion_criteria") or [])

    search_context = (
        f"ACTUAL SEARCH AND PIPELINE DETAILS — use only these facts. "
        f"Do not invent reviewer names, software names, reference managers, "
        f"figure numbers, or placeholders in brackets.\n\n"
        f"Databases searched: {sources_str} (no other databases were searched)\n"
        f"Date range: {date_from} to {date_to}\n"
        f"Search queries used:\n{queries_str}\n"
        f"Records retrieved: {search.get('n_retrieved', len(papers))}\n"
        f"After screening: {len(papers)} studies included, "
        f"{screening.get('n_excluded', '?')} excluded, "
        f"{screening.get('n_uncertain', '?')} uncertain\n"
        f"Inclusion criteria: {inclusion_str or 'see PICO'}\n"
        f"Exclusion criteria: {exclusion_str or 'see PICO'}\n\n"
        f"HOW THIS REVIEW WAS CONDUCTED (AI-assisted pipeline — describe accurately):\n"
        f"- Search: automated via PubMed Entrez API and bioRxiv REST API; "
        f"deduplication by PMID (programmatic, no reference manager used)\n"
        f"- Screening: performed by a large language model (Gemma) applying the "
        f"inclusion/exclusion criteria above to titles and abstracts; "
        f"results reviewed and overrideable by a human operator at a checkpoint\n"
        f"- Data extraction: performed by the same LLM from abstracts "
        f"(full text where available); each extracted field verified against "
        f"source text via fuzzy string matching (rapidfuzz); unverifiable fields "
        f"quarantined and flagged for human review\n"
        f"- Risk of bias: assessed using GRADE (Grading of Recommendations "
        f"Assessment, Development and Evaluation), not Cochrane RoB 2.0 or NOS; "
        f"five GRADE domains assessed per study: risk of bias, inconsistency, "
        f"indirectness, imprecision, and overall certainty "
        f"(high/moderate/low/very low); assessed by the LLM from extracted data\n"
        f"- Synthesis: narrative synthesis only — no meta-analysis, no forest "
        f"plots, no statistical pooling; each synthesised claim was verified "
        f"against the source paper extractions (LLM grounding); claims without "
        f"supporting evidence were quarantined\n"
        f"- There were no named human reviewers, no third reviewer, no Cochrane "
        f"tools, no NOS scale, no Excel forms, and no reference management "
        f"software; do not invent these or leave bracketed placeholders\n"
    )

    # Draft each section from the template
    sections_md = []
    for section in template["sections"]:
        name = section["name"]
        instructions = section.get("instructions", "")
        style = template.get("style_notes", "")
        response = llm.chat([{
            "role": "user",
            "content": (
                f"write the {name.lower()} section of a systematic review manuscript. "
                f"Instructions: {instructions}\n\n"
                f"PICO: P={pico['population']}, I={pico['intervention']}, "
                f"C={pico['comparator']}, O={pico['outcome']}.\n\n"
                f"{search_context}\n"
                f"Synthesis:\n{synthesis_text[:2000]}\n\n"
                f"Style: {style}{lang_suffix} "
                "Return JSON with field 'text'."
            ),
        }], schema=_TEXT_SCHEMA)
        sections_md.append(f"## {name}\n\n{response['text']}")

    draft = (
        f"# Systematic Review: {pico['intervention']} in {pico['population']}\n\n"
        + "\n\n".join(sections_md)
    )

    # Write versioned draft
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    draft_path = os.path.join(run_dir, f"stage_7_draft_v{draft_version}.md")
    with open(draft_path, "w") as f:
        f.write(draft)

    # Final manuscript path (updated each revision)
    md_path = os.path.join(run_dir, f"{run_id}_manuscript.md")
    with open(md_path, "w") as f:
        f.write(draft)

    # Score rubric
    rubric_result = score_rubric(draft, template, llm)

    # Export to Word
    docx_path = md_path.replace(".md", ".docx")
    try:
        run_pandoc(md_path, docx_path)
    except RuntimeError:
        pass

    return {
        "manuscript_path": md_path,
        "manuscript_rubric": {**rubric_result, "template": template},
        "manuscript_draft_version": draft_version,
    }


def create_manuscript_subgraph(db: Database, llm, output_dir: str):
    builder = StateGraph(dict)
    builder.add_node("draft", lambda s: _draft_manuscript_node(s, db, llm, output_dir))
    builder.set_entry_point("draft")
    builder.add_edge("draft", END)
    return builder.compile()
