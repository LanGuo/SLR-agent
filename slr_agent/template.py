import json
from typing import Any

DEFAULT_PRISMA_TEMPLATE: dict = {
    "sections": [
        {
            "name": "Abstract",
            "instructions": (
                "Provide a structured abstract with Background, Objectives, Data Sources, "
                "Study Eligibility Criteria, Participants, Interventions, Study Appraisal "
                "and Synthesis Methods, Results, Limitations, Conclusions."
            ),
            "rubric_criteria": [
                "Includes PICO elements in structured format",
                "Reports number of studies and participants",
                "States main findings with effect size",
                "Mentions limitations",
            ],
        },
        {
            "name": "Introduction",
            "instructions": (
                "Describe the rationale for the review, state the explicit research question "
                "in PICO format, and explain why this systematic review is needed."
            ),
            "rubric_criteria": [
                "States explicit research question in PICO format",
                "Justifies need for the review with evidence gap",
                "Describes expected benefits of the intervention",
            ],
        },
        {
            "name": "Methods",
            "instructions": (
                "Describe the protocol, eligibility criteria, information sources, search strategy, "
                "study selection process, data extraction, risk of bias assessment, and synthesis methods. "
                "Follow PRISMA 2020 checklist items 5-16."
            ),
            "rubric_criteria": [
                "Specifies inclusion and exclusion criteria",
                "Names all databases searched with date ranges",
                "Provides at least one full search string",
                "Describes risk of bias assessment tool used",
                "Explains data synthesis approach (meta-analysis or narrative)",
            ],
        },
        {
            "name": "Results",
            "instructions": (
                "Report study selection (PRISMA flow), study characteristics, risk of bias results, "
                "results of individual studies, and results of syntheses."
            ),
            "rubric_criteria": [
                "Includes PRISMA flow counts (retrieved, screened, included)",
                "Summarises characteristics of included studies",
                "Reports effect sizes with confidence intervals",
                "Presents risk of bias assessment results",
            ],
        },
        {
            "name": "Discussion",
            "instructions": (
                "Interpret results in context of existing evidence, discuss limitations of the review, "
                "and discuss implications for practice and research."
            ),
            "rubric_criteria": [
                "Interprets findings in context of prior evidence",
                "Discusses at least two limitations of the review",
                "Addresses clinical or policy implications",
            ],
        },
        {
            "name": "Conclusions",
            "instructions": (
                "Provide a brief, clear conclusion about the main findings and their implications. "
                "Do not introduce new evidence."
            ),
            "rubric_criteria": [
                "Directly answers the research question",
                "Does not introduce new information",
                "States implications for practice or future research",
            ],
        },
    ],
    "style_notes": "Follow PRISMA 2020 reporting guidelines. Use past tense for methods and results.",
}

_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "criterion": {"type": "string"},
                    "score": {"type": "string", "enum": ["met", "partial", "not met"]},
                    "explanation": {"type": "string"},
                },
                "required": ["criterion", "score", "explanation"],
            },
        }
    },
    "required": ["scores"],
}

_PDF_SCHEMA = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "instructions": {"type": "string"},
                    "rubric_criteria": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "instructions", "rubric_criteria"],
            },
        },
        "style_notes": {"type": "string"},
    },
    "required": ["sections", "style_notes"],
}


def load_template(path: str, llm: Any | None = None) -> dict:
    """Load and normalize a template from JSON schema or PDF.

    Returns normalized dict: {sections: [{name, instructions, rubric_criteria}], style_notes}.
    Raises ValueError for unsupported file formats.
    """
    if path.endswith(".json"):
        return _load_json_template(path)
    if path.lower().endswith(".pdf"):
        if llm is None:
            raise ValueError("LLM required to parse PDF templates")
        return _load_pdf_template(path, llm)
    raise ValueError(f"Unsupported template format: {path}. Use .json or .pdf")


def _load_json_template(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    sections = []
    for s in data.get("sections", []):
        sections.append({
            "name": s.get("name", ""),
            "instructions": s.get("instructions", ""),
            "rubric_criteria": s.get("rubric_criteria", []),
        })
    return {
        "sections": sections,
        "style_notes": data.get("style_notes", ""),
    }


def _load_pdf_template(path: str, llm: Any) -> dict:
    import fitz
    with fitz.open(path) as doc:
        text = "\n".join(page.get_text() for page in doc)[:8000]
    result = llm.chat([{
        "role": "user",
        "content": (
            "analyze this systematic review paper and extract the manuscript structure. "
            "For each section identify: the section name, instructions for what it should contain, "
            "and 3-5 measurable rubric criteria. Also note any style conventions. "
            "Return JSON with fields: sections (list of {name, instructions, rubric_criteria}), "
            f"style_notes.\n\nPaper text:\n{text}"
        ),
    }], schema=_PDF_SCHEMA)
    return result


def score_rubric(draft: str, template: dict, llm: Any) -> dict:
    """Score a manuscript draft against all rubric criteria in the template.

    Returns a dict with 'scores' list of {criterion, score, explanation}.
    score is one of: 'met', 'partial', 'not met'.
    """
    all_criteria = [
        c
        for section in template.get("sections", [])
        for c in section.get("rubric_criteria", [])
    ]
    if not all_criteria:
        return {"scores": []}

    result = llm.chat([{
        "role": "user",
        "content": (
            "score the following systematic review draft against each rubric criterion. "
            "For each criterion return: criterion (exact text), score ('met', 'partial', or 'not met'), "
            "and a one-sentence explanation.\n\n"
            f"Criteria:\n" + "\n".join(f"- {c}" for c in all_criteria) +
            f"\n\nDraft:\n{draft[:6000]}"
        ),
    }], schema=_SCORE_SCHEMA)
    return result
