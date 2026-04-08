# slr_agent/subgraphs/screening.py
from langgraph.graph import StateGraph, END
from slr_agent.db import Database
from slr_agent.grounding import ExtractionGrounder
from slr_agent.state import ScreeningCounts

_DEFAULT_BATCH_SIZE = 3    # abstracts per LLM call; smaller = shorter prompts = more reliable JSON

# Criterion score values
_MET = "yes"
_NOT_MET = "no"
_UNCLEAR = "unclear"


def _build_criteria_list(criteria: dict) -> list[dict]:
    """Flatten criteria dict into a typed list for prompt injection and schema use."""
    items = []
    for c in criteria.get("inclusion_criteria", []):
        items.append({"criterion": c, "type": "inclusion"})
    for c in criteria.get("exclusion_criteria", []):
        items.append({"criterion": c, "type": "exclusion"})
    for c in criteria.get("study_designs", []):
        items.append({"criterion": c, "type": "study_design"})
    return items


def _derive_decision(criterion_scores: list[dict]) -> str:
    """Derive include/exclude/uncertain from criterion scores.

    Rules:
    - Any exclusion criterion met (yes) → exclude
    - Any inclusion criterion not met (no) → exclude
    - Study designs present and none matched → exclude
    - Any score is unclear → uncertain (unless already excluded)
    - All met → include
    """
    inclusion = [s for s in criterion_scores if s["type"] == "inclusion"]
    exclusion = [s for s in criterion_scores if s["type"] == "exclusion"]
    designs = [s for s in criterion_scores if s["type"] == "study_design"]

    if any(s["met"] == _MET for s in exclusion):
        return "exclude"
    if any(s["met"] == _NOT_MET for s in inclusion):
        return "exclude"
    if designs and not any(s["met"] == _MET for s in designs):
        return "exclude"
    if any(s["met"] == _UNCLEAR for s in inclusion + exclusion + designs):
        return "uncertain"
    return "include"


def _screen_abstracts_node(state: dict, db: Database, llm) -> dict:
    run_id = state["run_id"]
    pico = state["pico"]
    papers = db.get_all_papers(run_id)
    grounder = ExtractionGrounder()
    batch_size = state.get("config", {}).get("screening_batch_size", _DEFAULT_BATCH_SIZE)

    criteria = state.get("screening_criteria") or {}
    criteria_list = _build_criteria_list(criteria)

    # Build criteria section for the prompt
    def _criteria_prompt_section() -> str:
        if not criteria_list:
            return ""
        lines = []
        inc = [c for c in criteria_list if c["type"] == "inclusion"]
        exc = [c for c in criteria_list if c["type"] == "exclusion"]
        des = [c for c in criteria_list if c["type"] == "study_design"]
        if des:
            lines.append("Eligible study designs (paper must match at least one):")
            for i, c in enumerate(des, 1):
                lines.append(f"  SD{i}. {c['criterion']}")
        if inc:
            lines.append("Inclusion criteria (ALL must be met):")
            for i, c in enumerate(inc, 1):
                lines.append(f"  IN{i}. {c['criterion']}")
        if exc:
            lines.append("Exclusion criteria (ANY one met → exclude):")
            for i, c in enumerate(exc, 1):
                lines.append(f"  EX{i}. {c['criterion']}")
        return "\n".join(lines) + "\n"

    criteria_section = _criteria_prompt_section()

    # JSON schema — criterion_scores items are fixed by the criteria list
    criterion_score_item = {
        "type": "object",
        "properties": {
            "criterion": {"type": "string"},
            "type": {"type": "string", "enum": ["inclusion", "exclusion", "study_design"]},
            "met": {"type": "string", "enum": [_MET, _NOT_MET, _UNCLEAR]},
            "note": {"type": "string"},
        },
        "required": ["criterion", "type", "met", "note"],
    }
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
                        "criterion_scores": {
                            "type": "array",
                            "items": criterion_score_item,
                        },
                    },
                    "required": ["pmid", "decision", "reason", "criterion_scores"],
                },
            }
        },
        "required": ["decisions"],
    }

    # Build criteria list instruction for prompt
    def _criteria_score_instruction() -> str:
        if not criteria_list:
            return ""
        lines = ["Score EACH of the following criteria for each paper:"]
        for c in criteria_list:
            lines.append(f'  - criterion: "{c["criterion"]}", type: "{c["type"]}"')
        lines.append(
            'Use met="yes" (clearly satisfied), "no" (clearly not satisfied), '
            '"unclear" (abstract lacks enough info).'
        )
        return "\n".join(lines) + "\n\n"

    criteria_score_instr = _criteria_score_instruction()

    def _build_prompt(batch_papers: list[dict]) -> str:
        batch_text = "\n\n".join(
            f"[PMID {p['pmid']}]\nTitle: {p['title']}\nAbstract: {p['abstract']}"
            for p in batch_papers
        )
        pmid_list = ", ".join(p["pmid"] for p in batch_papers)
        return (
            f"You are screening abstracts for a systematic review.\n"
            f"PICO: P={pico['population']}, I={pico['intervention']}, "
            f"C={pico['comparator']}, O={pico['outcome']}\n\n"
            f"{criteria_section}"
            "STRICT RULES (apply before deciding):\n"
            f"1. {pico['intervention']} must be the PRIMARY intervention — "
            "if only mentioned incidentally → EXCLUDE.\n"
            f"2. Outcomes must include observed {pico['outcome']} — "
            "surrogate endpoints only → EXCLUDE.\n"
            "3. Protocol/design papers with no reported results → EXCLUDE.\n"
            "4. Narrative reviews and non-systematic guideline summaries → EXCLUDE.\n"
            "5. Use 'uncertain' only when the abstract genuinely lacks enough information. "
            "Default to EXCLUDE when in doubt.\n\n"
            f"{criteria_score_instr}"
            "For each abstract, return:\n"
            "  - criterion_scores: one entry per criterion above\n"
            "  - decision: derived from scores — "
            "include (all inclusion + study_design met, no exclusion met), "
            "exclude (any exclusion met OR any inclusion not met), "
            "uncertain (abstract lacks info to judge)\n"
            "  - reason: one-sentence summary referencing the scores\n"
            f"You MUST return a decision for EVERY PMID listed: {pmid_list}.\n\n"
            f"{batch_text}"
        )

    def _apply_decisions(result: dict, paper_map: dict) -> set[str]:
        seen: set[str] = set()
        for decision in result.get("decisions", []):
            pmid = decision["pmid"]
            if pmid in seen:
                continue
            paper = paper_map.get(pmid)
            if not paper:
                continue
            seen.add(pmid)

            criterion_scores = decision.get("criterion_scores", [])
            # Override LLM decision with score-derived decision for consistency
            dec = _derive_decision(criterion_scores) if criterion_scores else decision["decision"]
            reason = decision["reason"]

            _, quarantined_fields = grounder.ground_extracted_data(
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
            paper["criterion_scores"] = criterion_scores
            paper["quarantined_fields"] = quarantined_fields
            db.upsert_paper(paper)
        return seen

    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]
        paper_map = {p["pmid"]: p for p in batch}

        result = llm.chat([{"role": "user", "content": _build_prompt(batch)}],
                          schema=schema, think=True)
        seen = _apply_decisions(result, paper_map)

        # Coverage validation: retry individually for any PMID the LLM skipped
        for missing_paper in [p for p in batch if p["pmid"] not in seen]:
            retry_result = llm.chat(
                [{"role": "user", "content": _build_prompt([missing_paper])}],
                schema=schema, think=True,
            )
            _apply_decisions(retry_result, {missing_paper["pmid"]: missing_paper})

    # Tally final decisions from DB (post-retry)
    n_included = n_excluded = n_uncertain = 0
    for p in db.get_all_papers(run_id):
        dec = p["screening_decision"]
        if dec == "include":
            n_included += 1
        elif dec == "exclude":
            n_excluded += 1
        elif dec not in ("excluded_manual",):
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
