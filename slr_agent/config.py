from typing import Literal
from typing_extensions import TypedDict

class RunConfig(TypedDict, total=False):
    checkpoint_stages: list[int]   # stage numbers 1–7; default [1, 2, 3, 5, 7]
    fetch_fulltext: bool           # default True
    output_format: Literal["markdown", "word", "both"]  # default "both"
    pubmed_api_key: str | None     # raises PubMed rate limit to 10 req/s
    max_results: int               # total cap per source; distributed across query strings for PubMed
    search_sources: list[Literal["pubmed", "biorxiv"]]  # default ["pubmed", "biorxiv"]
    template_path: str | None      # path to JSON schema or PDF template
    hitl_mode: Literal["cli", "ui", "none"]  # default "cli"
    date_from: str | None          # search date range start, e.g. "2000-01-01"
    date_to: str | None            # search date range end, e.g. "2026-12-31"
    model: str                     # Ollama model tag, e.g. "gemma4:e4b"
    screening_batch_size: int      # abstracts per LLM call during screening (default 3)

DEFAULT_CONFIG: RunConfig = {
    "checkpoint_stages": [1, 2, 3, 5, 7],
    "fetch_fulltext": True,
    "output_format": "both",
    "pubmed_api_key": None,
    "max_results": 500,
    "search_sources": ["pubmed", "biorxiv"],
    "template_path": None,
    "hitl_mode": "cli",
    "date_from": "2000-01-01",
    "date_to": None,               # None = no upper limit (current date at runtime)
    "model": "gemma4:e4b",         # default: Gemma 4 E4B (9.6 GB, fits 16 GB unified memory)
    "screening_batch_size": 3,     # abstracts per LLM screening call; smaller = more reliable JSON
}
