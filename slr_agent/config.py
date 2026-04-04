from typing import Literal
from typing_extensions import TypedDict

class RunConfig(TypedDict, total=False):
    checkpoint_stages: list[int]   # stage numbers 1–7; default [1, 3, 5]
    fetch_fulltext: bool           # default True
    output_format: Literal["markdown", "word", "both"]  # default "both"
    pubmed_api_key: str | None     # raises PubMed rate limit to 10 req/s
    max_results: int               # per source, default 500
    search_sources: list[Literal["pubmed", "biorxiv"]]  # default ["pubmed", "biorxiv"]
    template_path: str | None      # path to JSON schema or PDF template
    hitl_mode: Literal["cli", "ui", "none"]  # default "cli"

DEFAULT_CONFIG: RunConfig = {
    "checkpoint_stages": [1, 3, 5],
    "fetch_fulltext": True,
    "output_format": "both",
    "pubmed_api_key": None,
    "max_results": 500,
    "search_sources": ["pubmed", "biorxiv"],
    "template_path": None,
    "hitl_mode": "cli",
}
