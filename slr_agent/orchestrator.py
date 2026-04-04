# slr_agent/orchestrator.py
import uuid
from typing import Any
from langgraph.graph import StateGraph, END

from slr_agent.config import RunConfig, DEFAULT_CONFIG
from slr_agent.db import Database
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
):
    cfg = config or DEFAULT_CONFIG

    pico_sg = create_pico_subgraph(llm=llm)
    search_sg = create_search_subgraph(db=db)
    screening_sg = create_screening_subgraph(db=db, llm=llm)
    fulltext_sg = create_fulltext_subgraph(db=db, llm=llm)
    extraction_sg = create_extraction_subgraph(db=db, llm=llm)
    synthesis_sg = create_synthesis_subgraph(db=db, llm=llm, output_dir=output_dir)
    manuscript_sg = create_manuscript_subgraph(db=db, llm=llm, output_dir=output_dir)

    def pico_node(state: dict) -> dict:
        run_id = state.get("run_id") or str(uuid.uuid4())
        db.ensure_run(run_id)
        # Initialize all OrchestratorState fields so they are always present
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
            "current_stage": "pico",
            "checkpoint_pending": False,
        }
        sub_result = pico_sg.invoke({
            "raw_question": state["raw_question"],
            "pico": None,
            "validation_errors": [],
        })
        return {**base, **sub_result, "current_stage": "pico_done"}

    def search_node(state: dict) -> dict:
        sub_input = {
            **state,
            "pubmed_api_key": cfg.get("pubmed_api_key"),
            "max_results": cfg.get("max_results", 500),
            "search_sources": cfg.get("search_sources", ["pubmed"]),
        }
        result = search_sg.invoke(sub_input)
        return {**state, **result, "current_stage": "search_done"}

    def screening_node(state: dict) -> dict:
        result = screening_sg.invoke(state)
        return {**state, **result, "current_stage": "screening_done"}

    def fulltext_node(state: dict) -> dict:
        result = fulltext_sg.invoke(state)
        return {**state, **result, "current_stage": "fulltext_done"}

    def extraction_node(state: dict) -> dict:
        result = extraction_sg.invoke(state)
        return {**state, **result, "current_stage": "extraction_done"}

    def synthesis_node(state: dict) -> dict:
        result = synthesis_sg.invoke(state)
        return {**state, **result, "current_stage": "synthesis_done"}

    def manuscript_node(state: dict) -> dict:
        result = manuscript_sg.invoke(state)
        return {**state, **result, "current_stage": "done"}

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

    # Add SQLite checkpointer only when db_path is provided (enables resume/HITL)
    if db_path:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            checkpointer = SqliteSaver.from_conn_string(db_path)
            return builder.compile(checkpointer=checkpointer)
        except ImportError:
            import warnings
            warnings.warn(
                f"langgraph.checkpoint.sqlite not available — compiling without checkpointer. "
                f"Resume and HITL interrupts will not persist.",
                RuntimeWarning,
                stacklevel=2,
            )

    return builder.compile()
