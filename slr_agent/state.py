from typing import Literal
from typing_extensions import TypedDict
from slr_agent.config import RunConfig

class PICOResult(TypedDict):
    population: str
    intervention: str
    comparator: str
    outcome: str
    query_strings: list[str]   # editable at PICO checkpoint
    source_language: str       # ISO 639-1, e.g. "fr"
    search_language: str       # always "en"
    output_language: str       # same as source_language

class SearchCounts(TypedDict):
    n_retrieved: int
    n_duplicates_removed: int

class ScreeningCounts(TypedDict):
    n_included: int
    n_excluded: int
    n_uncertain: int

class FulltextCounts(TypedDict):
    n_fetched: int
    n_unavailable: int
    n_excluded: int

class ExtractionCounts(TypedDict):
    n_extracted: int
    n_grade_high: int
    n_grade_moderate: int
    n_grade_low: int
    n_grade_very_low: int
    n_quarantined_fields: int

class OrchestratorState(TypedDict):
    run_id: str
    config: RunConfig
    pico: PICOResult | None
    search_counts: SearchCounts | None
    screening_counts: ScreeningCounts | None
    fulltext_counts: FulltextCounts | None
    extraction_counts: ExtractionCounts | None
    synthesis_path: str | None
    manuscript_path: str | None
    current_stage: str
    checkpoint_pending: bool
