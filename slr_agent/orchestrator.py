# slr_agent/orchestrator.py
import os
import uuid
from typing import Any
from langgraph.graph import StateGraph, END

from slr_agent.broker import CheckpointBroker, NoOpHandler
from slr_agent.config import RunConfig, DEFAULT_CONFIG
from slr_agent.db import Database
from slr_agent.emitter import ProgressEmitter
from slr_agent.template import load_template
from slr_agent.subgraphs.pico import create_pico_subgraph
from slr_agent.subgraphs.search import create_search_subgraph
from slr_agent.subgraphs.screening import create_screening_subgraph
from slr_agent.subgraphs.fulltext import create_fulltext_subgraph
from slr_agent.subgraphs.extraction import create_extraction_subgraph
from slr_agent.subgraphs.synthesis import create_synthesis_subgraph
from slr_agent.subgraphs.manuscript import create_manuscript_subgraph


def _should_fetch_fulltext(state: dict) -> str:
    cfg = state.get("config", {})
    if cfg.get("fetch_fulltext", True):
        return "fulltext"
    return "extraction"


def create_orchestrator(
    db: Database,
    llm: Any,
    output_dir: str,
    config: RunConfig | None = None,
    db_path: str | None = None,
    broker: CheckpointBroker | None = None,
    emitter: ProgressEmitter | None = None,
):
    cfg = config or DEFAULT_CONFIG
    checkpoint_stages = cfg.get("checkpoint_stages", [])

    _broker = broker or CheckpointBroker(NoOpHandler())

    pico_sg = create_pico_subgraph(llm=llm)
    search_sg = create_search_subgraph(db=db)
    screening_sg = create_screening_subgraph(db=db, llm=llm)
    fulltext_sg = create_fulltext_subgraph(db=db, llm=llm)
    extraction_sg = create_extraction_subgraph(db=db, llm=llm)
    synthesis_sg = create_synthesis_subgraph(db=db, llm=llm, output_dir=output_dir)
    manuscript_sg = create_manuscript_subgraph(db=db, llm=llm, output_dir=output_dir)

    def _get_emitter(run_id: str) -> ProgressEmitter:
        if emitter is not None:
            return emitter
        return ProgressEmitter(output_dir=output_dir, run_id=run_id)

    def _maybe_pause(stage: int, stage_name: str, data: dict, run_id: str) -> dict:
        """Emit progress then pause if this stage is checkpointed."""
        _get_emitter(run_id).emit(stage, data)
        if stage in checkpoint_stages:
            return _broker.pause(stage, stage_name, data)
        return {**data, "action": "approve"}

    def pico_node(state: dict) -> dict:
        run_id = state.get("run_id") or str(uuid.uuid4())
        db.ensure_run(run_id)
        base = {
            **state,
            "run_id": run_id,
            "config": cfg,
            "pico": None,
            "search_counts": None,
            "screening_counts": None,
            "fulltext_counts": None,
            "extraction_counts": None,
            "synthesis_path": None,
            "manuscript_path": None,
            "template": None,
            "manuscript_draft_version": 1,
            "current_stage": "pico",
            "checkpoint_pending": False,
        }
        sub_result = pico_sg.invoke({
            "raw_question": state["raw_question"],
            "pico": None,
            "validation_errors": [],
        })
        pico_data = dict(sub_result.get("pico") or {})
        edited = _maybe_pause(1, "pico", pico_data, run_id)
        merged_pico = {**(sub_result.get("pico") or {}), **{
            k: v for k, v in edited.items() if k != "action"
        }}
        return {**base, **sub_result, "pico": merged_pico, "current_stage": "pico_done"}

    def search_node(state: dict) -> dict:
        _get_emitter(state["run_id"]).log("Searching PubMed...")
        sub_input = {
            **state,
            "pubmed_api_key": cfg.get("pubmed_api_key"),
            "max_results": cfg.get("max_results", 500),
            "search_sources": cfg.get("search_sources", ["pubmed"]),
        }
        result = search_sg.invoke(sub_input)
        run_id = state["run_id"]
        counts = result.get("search_counts", {})
        papers = db.get_all_papers(run_id)
        paper_list = [{"pmid": p["pmid"], "title": p["title"], "source": p["source"]} for p in papers]
        emit_data = {**dict(counts or {}), "papers": paper_list}
        edited = _maybe_pause(2, "search", emit_data, run_id)
        # Apply manual exclusions from gate
        for p in edited.get("papers", []):
            if p.get("excluded"):
                paper = db.get_paper(run_id, p["pmid"])
                if paper:
                    paper["screening_decision"] = "excluded_manual"
                    paper["screening_reason"] = "Excluded by user at search gate"
                    db.upsert_paper(paper)
        return {**state, **result, "current_stage": "search_done"}

    def screening_node(state: dict) -> dict:
        n_papers = len(db.get_all_papers(state["run_id"]))
        _get_emitter(state["run_id"]).log(f"Screening {n_papers} papers with LLM (may take several minutes)...")
        result = screening_sg.invoke(state)
        run_id = state["run_id"]
        papers = db.get_papers_by_decision(run_id, "include")
        excluded = db.get_papers_by_decision(run_id, "exclude")
        paper_list = [
            {"pmid": p["pmid"], "title": p["title"],
             "abstract": (p["abstract"] or "")[:300],
             "decision": p["screening_decision"],
             "reason": p["screening_reason"]}
            for p in papers + excluded
        ]
        emit_data = {**dict(result.get("screening_counts") or {}), "papers": paper_list}
        edited = _maybe_pause(3, "screening", emit_data, run_id)
        # Apply manual include/exclude overrides
        for p in edited.get("papers", []):
            if "decision" in p:
                record = db.get_paper(run_id, p["pmid"])
                if record and record["screening_decision"] != p["decision"]:
                    record["screening_decision"] = p["decision"]
                    record["screening_reason"] = p.get("reason", "User override")
                    db.upsert_paper(record)
        return {**state, **result, "current_stage": "screening_done"}

    def fulltext_node(state: dict) -> dict:
        result = fulltext_sg.invoke(state)
        emit_data = dict(result.get("fulltext_counts") or {})
        _get_emitter(state["run_id"]).emit(4, emit_data)
        return {**state, **result, "current_stage": "fulltext_done"}

    def extraction_node(state: dict) -> dict:
        n_included = len(db.get_papers_by_decision(state["run_id"], "include"))
        _get_emitter(state["run_id"]).log(f"Extracting data from {n_included} included papers...")
        result = extraction_sg.invoke(state)
        run_id = state["run_id"]
        papers = db.get_papers_by_decision(run_id, "include")
        paper_list = [
            {"pmid": p["pmid"], "title": p["title"],
             "extracted_data": p["extracted_data"],
             "quarantined_fields": p["quarantined_fields"],
             "grade_score": p["grade_score"]}
            for p in papers
        ]
        emit_data = {**dict(result.get("extraction_counts") or {}), "papers": paper_list}
        edited = _maybe_pause(5, "extraction", emit_data, run_id)
        # Apply manual field edits
        for p in edited.get("papers", []):
            record = db.get_paper(run_id, p["pmid"])
            if record and p.get("extracted_data"):
                record["extracted_data"] = p["extracted_data"]
                db.upsert_paper(record)
        return {**state, **result, "current_stage": "extraction_done"}

    def synthesis_node(state: dict) -> dict:
        _get_emitter(state["run_id"]).log("Synthesising evidence...")
        result = synthesis_sg.invoke(state)
        run_id = state["run_id"]
        synthesis_path = result.get("synthesis_path", "")
        synthesis_text = ""
        if synthesis_path and os.path.exists(synthesis_path):
            with open(synthesis_path) as f:
                synthesis_text = f.read()
        emit_data = {"synthesis_path": synthesis_path, "preview": synthesis_text[:500]}
        _maybe_pause(6, "synthesis", emit_data, run_id)
        # Note: stage 6 edits (synthesis text) are written directly to synthesis_path
        # by the Gradio panel's on_approve handler; the broker return is not needed here.
        return {**state, **result, "current_stage": "synthesis_done"}

    def manuscript_node(state: dict) -> dict:
        run_id = state["run_id"]
        _get_emitter(run_id).log("Generating manuscript draft...")
        # Load template if path provided
        template = state.get("template")
        template_path = cfg.get("template_path")
        if template is None and template_path:
            template = load_template(template_path, llm)

        result = manuscript_sg.invoke({**state, "template": template})
        current_state = {**state, **result, "template": template}

        # Stage 7 revision loop
        if 7 in checkpoint_stages:
            while True:
                manuscript_path = result.get("manuscript_path")
                if not manuscript_path or not os.path.exists(manuscript_path):
                    break
                with open(manuscript_path) as fh:
                    draft = fh.read()
                rubric = result.get("manuscript_rubric", {})
                checkpoint_data = {
                    "draft": draft,
                    "rubric": rubric,
                    "draft_version": result.get("manuscript_draft_version", 1),
                }
                _get_emitter(run_id).emit(7, {"rubric": rubric, "draft_version": checkpoint_data["draft_version"]})
                edited = _broker.pause(7, "manuscript", checkpoint_data)
                if edited.get("action") != "revise":
                    break
                revised_template = edited.get("template") or template
                revised_result = manuscript_sg.invoke({
                    **current_state,
                    "template": revised_template,
                    "manuscript_draft_version": result.get("manuscript_draft_version", 1),
                })
                result = revised_result
                current_state = {**current_state, **revised_result, "template": revised_template}
        else:
            _get_emitter(run_id).emit(7, result.get("manuscript_rubric", {}))

        return {**state, **result, "template": template, "current_stage": "done"}

    builder = StateGraph(dict)
    builder.add_node("pico", pico_node)
    builder.add_node("search", search_node)
    builder.add_node("screening", screening_node)
    builder.add_node("fulltext", fulltext_node)
    builder.add_node("extraction", extraction_node)
    builder.add_node("synthesis", synthesis_node)
    builder.add_node("manuscript", manuscript_node)

    builder.set_entry_point("pico")
    builder.add_edge("pico", "search")
    builder.add_edge("search", "screening")
    builder.add_conditional_edges("screening", _should_fetch_fulltext, {
        "fulltext": "fulltext",
        "extraction": "extraction",
    })
    builder.add_edge("fulltext", "extraction")
    builder.add_edge("extraction", "synthesis")
    builder.add_edge("synthesis", "manuscript")
    builder.add_edge("manuscript", END)

    if db_path:
        try:
            import sqlite3
            from langgraph.checkpoint.sqlite import SqliteSaver
            conn = sqlite3.connect(db_path, check_same_thread=False)
            checkpointer = SqliteSaver(conn)
            return builder.compile(checkpointer=checkpointer)
        except ImportError:
            import warnings
            warnings.warn(
                "langgraph-checkpoint-sqlite not installed — compiling without checkpointer.",
                RuntimeWarning,
                stacklevel=2,
            )

    return builder.compile()
