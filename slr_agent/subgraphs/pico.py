from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from slr_agent.state import PICOResult


class PICOSubgraphState(TypedDict):
    raw_question: str
    pico: PICOResult | None
    validation_errors: list[str]
    _detected_language: str
    _translated_question: str
    _source_language: str
    _pico_components: dict
    _query_strings: list[str]


def _detect_language_node(state: dict, llm) -> dict:
    result = llm.chat([{
        "role": "user",
        "content": (
            f"detect the language of this text and return JSON with field "
            f"'language_code' (ISO 639-1).\n\nText: {state['raw_question']}"
        ),
    }], schema={"type": "object", "properties": {"language_code": {"type": "string"}},
                "required": ["language_code"]})
    return {"_detected_language": result["language_code"]}


def _translate_node(state: dict, llm) -> dict:
    lang = state.get("_detected_language", "en")
    if lang == "en":
        return {"_translated_question": state["raw_question"], "_source_language": "en"}
    result = llm.chat([{
        "role": "user",
        "content": (
            f"translate the following research question to English. "
            f"Return JSON with field 'translated'.\n\n{state['raw_question']}"
        ),
    }], schema={"type": "object", "properties": {"translated": {"type": "string"}},
                "required": ["translated"]})
    return {
        "_translated_question": result["translated"],
        "_source_language": lang,
    }


def _expand_pico_node(state: dict, llm) -> dict:
    question = state.get("_translated_question") or state["raw_question"]
    result = llm.chat([{
        "role": "user",
        "content": (
            f"expand this research question into PICO components. Return JSON with "
            f"fields: population, intervention, comparator, outcome.\n\n{question}"
        ),
    }], schema={
        "type": "object",
        "properties": {
            "population": {"type": "string"},
            "intervention": {"type": "string"},
            "comparator": {"type": "string"},
            "outcome": {"type": "string"},
        },
        "required": ["population", "intervention", "comparator", "outcome"],
    })
    return {"_pico_components": result}


def _generate_queries_node(state: dict, llm) -> dict:
    pico = state["_pico_components"]
    result = llm.chat([{
        "role": "user",
        "content": (
            f"Generate 3-4 PubMed search query strings for this PICO:\n"
            f"P (population): {pico['population']}\n"
            f"I (intervention): {pico['intervention']}\n"
            f"C (comparator): {pico['comparator']}\n"
            f"O (outcome): {pico['outcome']}\n\n"
            f"Rules:\n"
            f"- Use ONLY plain Boolean operators: AND, OR, NOT, and parentheses.\n"
            f"- Do NOT use any field tags such as [MeSH], [tiab], [pt], [Title/Abstract], etc.\n"
            f"- EVERY query string MUST contain terms from ALL FOUR PICO components "
            f"(P AND I AND C AND O). Never omit a component.\n"
            f"- Vary coverage by using different synonyms and phrasings across queries, "
            f"not by dropping components.\n"
            f"- Each query must be broad enough to retrieve hundreds of papers.\n"
            f"- Include synonyms and related terms using OR within each component block.\n"
            f"- Structure each query as: (P terms) AND (I terms) AND (C terms) AND (O terms)\n"
            f"- Example: (adults OR patients) AND (aspirin OR acetylsalicylic acid) AND "
            f"(placebo OR standard care) AND (myocardial infarction OR stroke OR death)\n\n"
            f"Return JSON with field 'query_strings' (list of strings)."
        ),
    }], schema={
        "type": "object",
        "properties": {"query_strings": {"type": "array", "items": {"type": "string"}}},
        "required": ["query_strings"],
    })
    return {"_query_strings": result["query_strings"]}


def _validate_node(state: dict) -> dict:
    errors = []
    pico = state.get("_pico_components", {})
    for field in ["population", "intervention", "comparator", "outcome"]:
        if not pico.get(field, "").strip():
            errors.append(f"PICO field '{field}' is empty")
    queries = state.get("_query_strings", [])
    if not queries:
        errors.append("No query strings generated")
    for q in queries:
        if q.count("(") != q.count(")"):
            errors.append(f"Unbalanced parentheses in query: {q}")

    if errors:
        return {"validation_errors": errors, "pico": None}

    source_lang = state.get("_source_language", "en")
    pico_result = PICOResult(
        population=pico["population"],
        intervention=pico["intervention"],
        comparator=pico["comparator"],
        outcome=pico["outcome"],
        query_strings=state["_query_strings"],
        source_language=source_lang,
        search_language="en",
        output_language=source_lang,
    )
    return {"pico": pico_result, "validation_errors": []}


def create_pico_subgraph(llm):
    builder = StateGraph(PICOSubgraphState)

    builder.add_node("detect_language", lambda s: _detect_language_node(s, llm))
    builder.add_node("translate", lambda s: _translate_node(s, llm))
    builder.add_node("expand_pico", lambda s: _expand_pico_node(s, llm))
    builder.add_node("generate_queries", lambda s: _generate_queries_node(s, llm))
    builder.add_node("validate", _validate_node)

    builder.set_entry_point("detect_language")
    builder.add_edge("detect_language", "translate")
    builder.add_edge("translate", "expand_pico")
    builder.add_edge("expand_pico", "generate_queries")
    builder.add_edge("generate_queries", "validate")
    builder.add_edge("validate", END)

    return builder.compile()
