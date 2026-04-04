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
    lang_suffix = f" Write in {output_language}." if output_language != "en" else ""

    # Draft each section from the template
    sections_md = []
    for section in template["sections"]:
        name = section["name"]
        instructions = section.get("instructions", "")
        style = template.get("style_notes", "")
        response = llm.chat([{
            "role": "user",
            "content": (
                f"write the {name} section of a systematic review manuscript. "
                f"Instructions: {instructions} "
                f"Context: P={pico['population']}, I={pico['intervention']}, "
                f"C={pico['comparator']}, O={pico['outcome']}. "
                f"{len(papers)} studies included. "
                f"Synthesis:\n{synthesis_text[:2000]}\n"
                f"Excluded: {screening.get('n_excluded', '?')}. "
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
