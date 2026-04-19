# slr_agent/orchestrator.py
import datetime
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
from slr_agent.citation_network import build_citation_network


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
    fulltext_sg = create_fulltext_subgraph(db=db, llm=llm, output_dir=output_dir)
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
        today = datetime.date.today().strftime("%Y-%m-%d")
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
            "date_from": cfg.get("date_from", "2000-01-01"),
            "date_to": cfg.get("date_to") or today,
            "search_sources": cfg.get("search_sources", ["pubmed"]),
            "max_results": cfg.get("max_results", 500),
            "screening_criteria": None,
            "citation_network": None,
            "current_stage": "pico",
            "checkpoint_pending": False,
        }
        sub_result = pico_sg.invoke({
            "raw_question": state["raw_question"],
            "pico": None,
            "validation_errors": [],
        })
        pico_fields = dict(sub_result.get("pico") or {})
        # Stage 1 gate includes PICO fields AND search configuration so user can edit both
        _SEARCH_KEYS = {"search_sources", "max_results", "date_from", "date_to", "action"}
        gate_data = {
            **pico_fields,
            "search_sources": base["search_sources"],
            "max_results": base["max_results"],
            "date_from": base["date_from"],
            "date_to": base["date_to"],
        }
        edited = _maybe_pause(1, "pico", gate_data, run_id)
        merged_pico = {**pico_fields, **{k: v for k, v in edited.items() if k not in _SEARCH_KEYS}}
        final_search_sources = edited.get("search_sources", base["search_sources"])
        final_max_results = int(edited.get("max_results", base["max_results"]))
        final_date_from = edited.get("date_from", base["date_from"])
        final_date_to = edited.get("date_to", base["date_to"])
        # Overwrite the stage_1_pico.json with user-edited values so the file matches what runs
        _get_emitter(run_id).emit(1, {
            **merged_pico,
            "search_sources": final_search_sources,
            "max_results": final_max_results,
            "date_from": final_date_from,
            "date_to": final_date_to,
        })
        return {
            **base,
            **sub_result,
            "pico": merged_pico,
            "search_sources": final_search_sources,
            "max_results": final_max_results,
            "date_from": final_date_from,
            "date_to": final_date_to,
            "current_stage": "pico_done",
        }

    def search_node(state: dict) -> dict:
        _get_emitter(state["run_id"]).log("Searching PubMed...")
        sub_input = {
            **state,
            "pubmed_api_key": cfg.get("pubmed_api_key"),
            # Use user-edited values from pico gate if present, else fall back to config
            "max_results": state.get("max_results") or cfg.get("max_results", 500),
            "search_sources": state.get("search_sources") or cfg.get("search_sources", ["pubmed"]),
            "date_from": state.get("date_from") or cfg.get("date_from", "2000-01-01"),
            "date_to": state.get("date_to") or datetime.date.today().strftime("%Y-%m-%d"),
        }
        result = search_sg.invoke(sub_input)
        run_id = state["run_id"]
        counts = result.get("search_counts", {})
        papers = db.get_all_papers(run_id)
        paper_list = [
            {"pmid": p["pmid"], "title": p["title"], "source": p["source"], "excluded": False}
            for p in papers
        ]
        emit_data = {
            **dict(counts or {}),
            "_instructions": (
                "Set excluded=true on any paper to remove it before screening. "
                "To add a paper by PMID set manual_add=true and fill pmid+title. "
                "Example add entry: {\"pmid\":\"12345678\",\"title\":\"\",\"excluded\":false,\"manual_add\":true}"
            ),
            "papers": paper_list,
        }
        edited = _maybe_pause(2, "search", emit_data, run_id)
        existing_pmids = {p["pmid"] for p in db.get_all_papers(run_id)}
        for p in edited.get("papers", []):
            pmid = p.get("pmid", "")
            if p.get("excluded"):
                paper = db.get_paper(run_id, pmid)
                if paper:
                    paper["screening_decision"] = "excluded_manual"
                    paper["screening_reason"] = "Excluded by user at search gate"
                    db.upsert_paper(paper)
            elif p.get("manual_add") and pmid and pmid not in existing_pmids:
                _get_emitter(run_id).log(f"Fetching manually added paper: {pmid}")
                db.add_paper_from_pmid(run_id, pmid, api_key=cfg.get("pubmed_api_key"))
        # Re-emit stage 2 with exclusion state applied so the on-disk file reflects
        # what actually proceeds to screening (same pattern as pico_node re-emit).
        all_papers_after = db.get_all_papers(run_id)
        _get_emitter(run_id).emit(2, {
            **dict(counts or {}),
            "papers": [
                {"pmid": p["pmid"], "title": p["title"], "source": p["source"],
                 "excluded": p["screening_decision"] == "excluded_manual"}
                for p in all_papers_after
            ],
        })
        return {**state, **result, "current_stage": "search_done"}

    def screening_node(state: dict) -> dict:
        run_id = state["run_id"]
        pico = state.get("pico") or {}

        # Generate explicit inclusion/exclusion criteria from PICO before screening
        _get_emitter(run_id).log("Generating screening criteria from PICO...")
        criteria_schema = {
            "type": "object",
            "properties": {
                "inclusion_criteria": {"type": "array", "items": {"type": "string"}},
                "exclusion_criteria": {"type": "array", "items": {"type": "string"}},
                "study_designs": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["inclusion_criteria", "exclusion_criteria", "study_designs"],
        }
        criteria_result = llm.chat([{
            "role": "user",
            "content": (
                "You are a systematic review methodologist. Generate explicit, specific inclusion "
                "and exclusion criteria for screening abstracts for this systematic review.\n\n"
                f"Population: {pico.get('population', '')}\n"
                f"Intervention: {pico.get('intervention', '')}\n"
                f"Comparator: {pico.get('comparator', '')}\n"
                f"Outcome: {pico.get('outcome', '')}\n\n"
                "Return:\n"
                "- inclusion_criteria: list of specific eligibility criteria (study design, "
                "population, intervention, outcome, language, publication type)\n"
                "- exclusion_criteria: list of explicit reasons to exclude a paper\n"
                "- study_designs: list of eligible study design types (e.g. RCT, cohort, meta-analysis)\n"
            ),
        }], schema=criteria_schema)
        screening_criteria = {
            "inclusion_criteria": criteria_result.get("inclusion_criteria", []),
            "exclusion_criteria": criteria_result.get("exclusion_criteria", []),
            "study_designs": criteria_result.get("study_designs", []),
        }

        # Pre-screening criteria gate (fires before any LLM screening)
        if 3 in checkpoint_stages:
            _get_emitter(run_id).log("Review screening criteria before screening begins...")
            criteria_edited = _broker.pause(3, "screening_criteria", screening_criteria)
            screening_criteria = {k: v for k, v in criteria_edited.items() if k != "action"}

        # Save criteria to disk with distinct filename
        _get_emitter(run_id).emit(3, screening_criteria, name="screening_criteria")

        n_papers = len(db.get_all_papers(run_id))
        _get_emitter(run_id).log(f"Screening {n_papers} papers with LLM (may take several minutes)...")
        result = screening_sg.invoke({**state, "screening_criteria": screening_criteria})
        papers = db.get_papers_by_decision(run_id, "include")
        excluded = db.get_papers_by_decision(run_id, "exclude")
        uncertain = db.get_papers_by_decision(run_id, "uncertain")
        paper_list = [
            {"pmid": p["pmid"], "title": p["title"],
             "abstract": (p["abstract"] or "")[:2000],
             "decision": p["screening_decision"],
             "reason": p["screening_reason"],
             "criterion_scores": p.get("criterion_scores") or []}
            for p in papers + excluded + uncertain
        ]
        emit_data = {**dict(result.get("screening_counts") or {}), "papers": paper_list}
        edited = _maybe_pause(3, "screening", emit_data, run_id)
        # Apply manual include/exclude overrides
        for p in edited.get("papers", []):
            if "decision" in p:
                record = db.get_paper(run_id, p["pmid"])
                if record and record["screening_decision"] != p["decision"]:
                    record["screening_decision"] = p["decision"]
                    record["screening_reason"] = p.get("reason") or "User override at HITL gate"
                    db.upsert_paper(record)
        return {**state, **result, "screening_criteria": screening_criteria, "current_stage": "screening_done"}

    def fulltext_node(state: dict) -> dict:
        run_id = state["run_id"]
        result = fulltext_sg.invoke(state)
        emit_data = dict(result.get("fulltext_counts") or {})
        _get_emitter(run_id).emit(4, emit_data)
        included = db.get_papers_by_decision(run_id, "include")
        cn_summary = build_citation_network(included)
        if cn_summary.warning:
            _get_emitter(run_id).log(cn_summary.warning)
        return {**state, **result, "citation_network": cn_summary.to_dict(), "current_stage": "fulltext_done"}

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
        for p in edited.get("papers", []):
            pmid = p.get("pmid", "")
            record = db.get_paper(run_id, pmid)
            if not record:
                continue
            if p.get("exclude"):
                record["screening_decision"] = "excluded_manual"
                record["screening_reason"] = "Excluded by user at extraction gate"
                db.upsert_paper(record)
            elif p.get("extracted_data"):
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
        # Compute citation network here if fulltext was skipped (fetch_fulltext=False)
        citation_network = state.get("citation_network")
        if citation_network is None:
            included = db.get_papers_by_decision(run_id, "include")
            cn_summary = build_citation_network(included)
            if cn_summary.warning:
                _get_emitter(run_id).log(cn_summary.warning)
            citation_network = cn_summary.to_dict()
        cn = citation_network or {}
        emit_data = {
            "synthesis_path": synthesis_path,
            "preview": synthesis_text[:500],
            "unresolved_questions": result.get("unresolved_questions") or [],
            "citation_network": cn,
        }
        _maybe_pause(6, "synthesis", emit_data, run_id)
        # Note: stage 6 edits (synthesis text) are written directly to synthesis_path
        # by the Gradio panel's on_approve handler; the broker return is not needed here.
        return {**state, **result, "citation_network": citation_network, "current_stage": "synthesis_done"}

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

        # Handle FATAL adversarial issues that require prior-stage reruns.
        # Bounded to one retry to prevent infinite loops.
        adversarial_review = result.get("adversarial_review", {})
        fatal_issues = [
            i for i in adversarial_review.get("issues", [])
            if i.get("severity") == "FATAL" and i.get("rerun_stage")
        ]
        if fatal_issues:
            rerun_stages = {i["rerun_stage"] for i in fatal_issues}
            _known_stages = {"screening", "extraction", "synthesis"}
            unknown = rerun_stages - _known_stages
            if unknown:
                _get_emitter(run_id).log(
                    f"Adversarial reviewer named unknown rerun stages (ignored): {unknown}"
                )
            rerun_stages = rerun_stages & _known_stages
            if not rerun_stages:
                fatal_issues = []  # nothing to rerun
        if fatal_issues:
            _get_emitter(run_id).log(
                f"Adversarial reviewer flagged FATAL issues — auto-rerunning: "
                f"{', '.join(sorted(rerun_stages))}"
            )
            # Only rerun the explicitly named stage(s). Pipeline dependencies are
            # respected by running in the correct order, but stages not named by
            # the reviewer are NOT re-run to avoid unnecessary cost.
            if "screening" in rerun_stages:
                screening_result = screening_sg.invoke(current_state)
                current_state = {**current_state, **screening_result}
            if "extraction" in rerun_stages:
                extraction_result = extraction_sg.invoke(current_state)
                current_state = {**current_state, **extraction_result}
            if "synthesis" in rerun_stages:
                synthesis_result = synthesis_sg.invoke(current_state)
                current_state = {**current_state, **synthesis_result}
            # Redraft once after rerun
            result = manuscript_sg.invoke({**current_state, "template": template})
            current_state = {**current_state, **result, "template": template}

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
                    "adversarial_review": result.get("adversarial_review", {"issues": []}),
                }
                _get_emitter(run_id).emit(7, {"rubric": rubric, "draft_version": checkpoint_data["draft_version"]})
                edited = _broker.pause(7, "manuscript", checkpoint_data)
                if edited.get("action") != "revise":
                    # Persist any direct edits or section rewrites the user made in the UI
                    edited_draft = edited.get("edited_draft", "")
                    if edited_draft and edited_draft != draft:
                        with open(manuscript_path, "w") as fh:
                            fh.write(edited_draft)
                        from slr_agent.export import run_pandoc
                        docx_path = manuscript_path.replace(".md", ".docx")
                        try:
                            run_pandoc(manuscript_path, docx_path)
                        except RuntimeError:
                            pass
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
