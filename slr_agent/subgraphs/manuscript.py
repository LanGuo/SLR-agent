# slr_agent/subgraphs/manuscript.py
import os
import re
from langgraph.graph import StateGraph, END
from slr_agent.db import Database
from slr_agent.export import run_pandoc
from slr_agent.template import DEFAULT_PRISMA_TEMPLATE, score_rubric


def _build_study_table(papers: list[dict]) -> str:
    """Generate a markdown study characteristics table directly from extracted data."""
    if not papers:
        return "_No included studies._"
    header = (
        "| PMID | Title | Study Design | Population | Intervention | "
        "Comparator | Primary Outcome | Sample Size | Follow-up |\n"
        "|------|-------|--------------|------------|--------------|"
        "------------|-----------------|-------------|----------|\n"
    )
    rows = []
    for p in papers:
        ed = p.get("extracted_data") or {}
        title = (p.get("title") or "")[:60]
        rows.append(
            f"| {p['pmid']} | {title} | {ed.get('study_design', '?')} | "
            f"{ed.get('population_details', '?')[:60]} | "
            f"{ed.get('intervention', '?')[:60]} | "
            f"{ed.get('comparator', '?')[:60]} | "
            f"{ed.get('primary_outcome', '?')[:60]} | "
            f"{ed.get('sample_size', '?')} | "
            f"{ed.get('follow_up_duration', '?')} |"
        )
    return header + "\n".join(rows)


def _build_grade_table(papers: list[dict]) -> str:
    """Generate a markdown GRADE summary table directly from grade_score data."""
    if not papers:
        return "_No included studies._"
    header = (
        "| PMID | Title | Certainty | Risk of Bias | Inconsistency | "
        "Indirectness | Imprecision |\n"
        "|------|-------|-----------|--------------|---------------|"
        "--------------|-------------|\n"
    )
    rows = []
    for p in papers:
        gs = p.get("grade_score") or {}
        title = (p.get("title") or "")[:60]
        rows.append(
            f"| {p['pmid']} | {title} | {gs.get('certainty', '?')} | "
            f"{gs.get('risk_of_bias', '?')} | {gs.get('inconsistency', '?')} | "
            f"{gs.get('indirectness', '?')} | {gs.get('imprecision', '?')} |"
        )
    return header + "\n".join(rows)

_ANCHOR_SCHEMA = {
    "type": "object",
    "properties": {
        "anchored": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "pmids": {"type": "array", "items": {"type": "string"}},
                    "section": {"type": "string"},
                },
                "required": ["claim", "pmids", "section"],
            },
        },
    },
    "required": ["anchored"],
}


def _verify_citations_node(draft: str, synthesis_path: str, llm) -> str:
    """Second pass: anchor synthesis claims to PMID citations in the draft.

    Reads grounded claims from the synthesis file (lines matching "- ... [PMIDs]"),
    asks the LLM which section of the draft each claim appears in, then injects
    "(PMID: X, Y)" markers at the end of the first matching line per claim.

    Returns the annotated draft string.
    """
    if not draft or not synthesis_path or not os.path.exists(synthesis_path):
        return draft

    # Parse grounded claims — lines of the form: "- claim text [PMID1, PMID2]"
    claims = []
    with open(synthesis_path) as f:
        for line in f:
            m = re.match(r"^- (.+) \[([^\]]+)\]$", line.strip())
            if m:
                claims.append({
                    "text": m.group(1).strip(),
                    "pmids": [p.strip() for p in m.group(2).split(",")],
                })

    if not claims:
        return draft

    claims_text = "\n".join(
        f"- \"{c['text']}\" → PMIDs: {', '.join(c['pmids'])}"
        for c in claims
    )

    try:
        result = llm.chat([{
            "role": "user",
            "content": (
                "Anchor citations for a systematic review manuscript.\n\n"
                "Below is the draft and grounded claims with their supporting PMIDs. "
                "For each claim, identify which section of the manuscript contains a "
                "sentence making this point, and return the claim, its PMIDs, and the section.\n\n"
                # Draft truncated to 6000 chars for LLM context budget.
                # Claims appearing in sections past this point fall back to section-header injection.
                f"DRAFT:\n{draft[:6000]}\n\n"
                f"CLAIMS WITH PMIDS:\n{claims_text}\n\n"
                "Return JSON with field 'anchored': list of {claim, pmids, section}."
            ),
        }], schema=_ANCHOR_SCHEMA)
    except Exception:
        # Citation anchoring is non-critical; degrade gracefully on LLM failure
        return draft

    # Inject "(PMID: X, Y)" after the first line containing the claim key (first 60 chars).
    # Fallback: if the claim text is not found verbatim, inject at the first non-empty
    # content line inside the named section.
    annotated = draft
    for anchor in result.get("anchored", []):
        claim_key = anchor["claim"][:60].lower().strip()
        pmid_tag = " (PMID: " + ", ".join(anchor["pmids"]) + ")"
        section_name = (anchor.get("section") or "").lower().strip()
        # Inject into the first line containing the claim text.
        # If two claims match the same line, the second falls through to the section fallback.
        lines = annotated.split("\n")
        injected = False
        # First try: find claim text directly
        for i, line in enumerate(lines):
            if claim_key in line.lower() and "(PMID:" not in line:
                lines[i] = line.rstrip() + pmid_tag
                injected = True
                break
        # Fallback: inject into the named section's first content line
        if not injected and section_name:
            in_section = False
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.lower() in (f"## {section_name}", f"# {section_name}"):
                    in_section = True
                    continue
                if in_section and stripped.startswith("#"):
                    break  # reached next section
                if in_section and stripped and "(PMID:" not in line:
                    lines[i] = line.rstrip() + pmid_tag
                    injected = True
                    break
        annotated = "\n".join(lines)

    return annotated


_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["FATAL", "MAJOR", "MINOR"]},
                    "section": {"type": "string"},
                    "issue": {"type": "string"},
                    "suggestion": {"type": "string"},
                    "rerun_stage": {"type": ["string", "null"]},
                },
                "required": ["severity", "section", "issue", "suggestion", "rerun_stage"],
            },
        },
    },
    "required": ["issues"],
}


def _adversarial_review_node(draft: str, synthesis_path: str, llm) -> dict:
    """Adversarial reviewer: critiques the draft with FATAL/MAJOR/MINOR severity.

    FATAL issues may name a prior pipeline stage to rerun (rerun_stage: screening |
    extraction | synthesis | null). All issues are returned as a dict for the
    orchestrator to surface at Gate 7 or trigger automatic reruns.

    Returns {"issues": [...]} — empty list on LLM failure (non-fatal degradation).
    """
    synthesis_text = ""
    if synthesis_path and os.path.exists(synthesis_path):
        with open(synthesis_path) as f:
            synthesis_text = f.read()

    try:
        result = llm.chat([{
            "role": "user",
            "content": (
                "You are an adversarial reviewer for a systematic review manuscript. "
                "Find every flaw, gap, and inconsistency in this draft before it reaches "
                "a human editor.\n\n"
                "Severity levels:\n"
                "- FATAL: the manuscript is wrong or misleading in a way that requires "
                "redoing a prior pipeline stage (name the stage in rerun_stage: "
                "screening | extraction | synthesis | null)\n"
                "- MAJOR: a significant flaw the human must address before submission\n"
                "- MINOR: a style, clarity, or minor accuracy issue\n\n"
                f"MANUSCRIPT DRAFT:\n{draft[:8000]}\n\n"
                f"SYNTHESIS:\n{synthesis_text[:2000]}\n\n"
                "Return JSON with field 'issues': list of "
                "{severity, section, issue, suggestion, rerun_stage}."
            ),
        }], schema=_REVIEW_SCHEMA, think=True)
    except Exception:
        return {"issues": []}

    return result if isinstance(result, dict) and "issues" in result else {"issues": []}


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
    fulltext = state.get("fulltext_counts") or {}
    extraction = state.get("extraction_counts") or {}
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
        f"EXACT PRISMA FLOW COUNTS (use only these numbers — do not invent any):\n"
        f"- Records identified via database search: {search.get('n_retrieved', len(papers))}\n"
        f"- Duplicates removed: {search.get('n_duplicates_removed', 0)}\n"
        f"- Records screened (titles/abstracts): {search.get('n_retrieved', len(papers)) - search.get('n_duplicates_removed', 0)}\n"
        f"- Records excluded after screening: {screening.get('n_excluded', '?')}\n"
        f"- Records uncertain / not assessed: {screening.get('n_uncertain', '?')}\n"
        f"- Full-text articles assessed: {fulltext.get('n_fetched', 'not fetched — abstracts only')}\n"
        f"- Full-text excluded: {fulltext.get('n_excluded', 'N/A')}\n"
        f"- Studies included in review: {len(papers)}\n"
        f"- Studies with quarantined extraction fields: {extraction.get('n_quarantined_fields', '?')}\n"
        f"- PRISMA flow diagram: generated as a separate file ({run_id}_prisma.md) — "
        f"do not write '[Figure would be included here]'; instead refer to it as "
        f"'the accompanying PRISMA flow diagram'\n\n"
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
        f"- Do not reference supplementary tables, supplementary materials, "
        f"or appendices — no such files exist\n"
    )

    # Pre-generate tables directly from DB — no LLM, exact data
    study_table = _build_study_table(papers)
    grade_table = _build_grade_table(papers)

    # Section names that should receive the pre-generated tables appended verbatim
    _STUDY_TABLE_SECTIONS = {"study characteristics", "characteristics of included studies"}
    _GRADE_TABLE_SECTIONS = {"risk of bias", "quality assessment", "grade assessment"}

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
                "Do NOT include inline citations (no PMID numbers, no author-year, "
                "no brackets). Citations will be added in a separate verification pass. "
                "Return JSON with field 'text'."
            ),
        }], schema=_TEXT_SCHEMA)
        section_md = f"## {name}\n\n{response['text']}"

        # Append pre-generated tables to the relevant sections
        name_lower = name.lower()
        if any(k in name_lower for k in _STUDY_TABLE_SECTIONS):
            section_md += f"\n\n**Table 1. Characteristics of Included Studies**\n\n{study_table}"
        if any(k in name_lower for k in _GRADE_TABLE_SECTIONS):
            section_md += f"\n\n**Table 2. GRADE Evidence Quality Assessment**\n\n{grade_table}"

        sections_md.append(section_md)

    draft = (
        f"# Systematic Review: {pico['intervention']} in {pico['population']}\n\n"
        + "\n\n".join(sections_md)
    )

    # Citation verifier pass — anchors synthesis claims to PMIDs in the draft
    draft = _verify_citations_node(draft, synthesis_path, llm)

    # Adversarial reviewer pass — structured critique before Gate 7
    adversarial_review = _adversarial_review_node(draft, synthesis_path, llm)

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
        "adversarial_review": adversarial_review,
    }


def create_manuscript_subgraph(db: Database, llm, output_dir: str):
    builder = StateGraph(dict)
    builder.add_node("draft", lambda s: _draft_manuscript_node(s, db, llm, output_dir))
    builder.set_entry_point("draft")
    builder.add_edge("draft", END)
    return builder.compile()
