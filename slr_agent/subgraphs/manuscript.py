# slr_agent/subgraphs/manuscript.py
import os
from langgraph.graph import StateGraph, END
from slr_agent.db import Database
from slr_agent.export import run_pandoc

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

    synthesis_text = ""
    if synthesis_path and os.path.exists(synthesis_path):
        with open(synthesis_path) as f:
            synthesis_text = f.read()

    papers = db.get_papers_by_decision(run_id, "include")
    screening = state.get("screening_counts") or {}
    lang_suffix = f" Write in {output_language}." if output_language != "en" else ""

    # Draft Methods
    methods = llm.chat([{
        "role": "user",
        "content": (
            f"write the Methods section of a systematic review manuscript. "
            f"Research question: P={pico['population']}, I={pico['intervention']}, "
            f"C={pico['comparator']}, O={pico['outcome']}. "
            f"Database searched: PubMed. {len(papers)} studies included. "
            f"Follow PRISMA 2020 reporting standards.{lang_suffix} "
            "Return JSON with field 'text'."
        ),
    }], schema=_TEXT_SCHEMA)

    # Draft Results
    results_section = llm.chat([{
        "role": "user",
        "content": (
            f"write the Results section of a systematic review manuscript.\n"
            f"Included: {len(papers)}, excluded: {screening.get('n_excluded', '?')}.\n"
            f"Synthesis:\n{synthesis_text}\n\n"
            f"{lang_suffix} "
            "Return JSON with field 'text'."
        ),
    }], schema=_TEXT_SCHEMA)

    # Draft Discussion
    discussion = llm.chat([{
        "role": "user",
        "content": (
            f"write the Discussion section of a systematic review manuscript.\n"
            f"Findings summary:\n{synthesis_text}\n\n"
            f"Research question: P={pico['population']}, I={pico['intervention']}, "
            f"C={pico['comparator']}, O={pico['outcome']}.{lang_suffix} "
            "Return JSON with field 'text'."
        ),
    }], schema=_TEXT_SCHEMA)

    # Write manuscript markdown
    os.makedirs(output_dir, exist_ok=True)
    md_path = os.path.join(output_dir, f"{run_id}_manuscript.md")
    with open(md_path, "w") as f:
        f.write(f"# Systematic Review: {pico['intervention']} in {pico['population']}\n\n")
        f.write(f"## Methods\n\n{methods['text']}\n\n")
        f.write(f"## Results\n\n{results_section['text']}\n\n")
        if synthesis_text:
            f.write(f"### Evidence Synthesis\n\n{synthesis_text}\n\n")
        f.write(f"## Discussion\n\n{discussion['text']}\n")

    # Export to Word
    docx_path = md_path.replace(".md", ".docx")
    try:
        run_pandoc(md_path, docx_path)
    except RuntimeError:
        pass  # Pandoc unavailable — markdown is the fallback

    return {"manuscript_path": md_path}


def create_manuscript_subgraph(db: Database, llm, output_dir: str):
    builder = StateGraph(dict)
    builder.add_node("draft", lambda s: _draft_manuscript_node(s, db, llm, output_dir))
    builder.set_entry_point("draft")
    builder.add_edge("draft", END)
    return builder.compile()
