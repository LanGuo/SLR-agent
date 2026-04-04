import re
from typing import Literal
from typing_extensions import TypedDict
from rapidfuzz import fuzz
from slr_agent.db import Span, QuarantinedField

# Threshold by source type: abstracts paraphrase full text, so legitimate
# extractions score lower than verbatim full-text matches.
_THRESHOLD_ABSTRACT = 75
_THRESHOLD_FULLTEXT = 85

# Values shorter than this are grounded by exact substring search instead of
# fuzzy matching — fuzzy scores on short strings (e.g. "96") are unreliable.
_SHORT_VALUE_CHARS = 20


class GroundedField(TypedDict):
    value: str
    span: Span | None
    confidence: float
    status: Literal["grounded", "quarantined"]


class ExtractionGrounder:
    def __init__(self, threshold: int | None = None):
        # threshold param kept for backward compatibility; source-adaptive thresholds
        # are used instead when None (see _THRESHOLD_ABSTRACT / _THRESHOLD_FULLTEXT).
        self._override_threshold = threshold

    def _threshold_for(self, source: Literal["abstract", "fulltext"]) -> int:
        if self._override_threshold is not None:
            return self._override_threshold
        return _THRESHOLD_ABSTRACT if source == "abstract" else _THRESHOLD_FULLTEXT

    def ground_field(
        self,
        field_name: str,
        value: str,
        source_text: str,
        pmid: str,
        source: Literal["abstract", "fulltext"],
    ) -> GroundedField:
        threshold = self._threshold_for(source)
        value_lower = value.lower()
        source_lower = source_text.lower()

        # Short values (numbers, short phrases): exact substring search is more
        # reliable than fuzzy matching, which is noise-dominated for short strings.
        if len(value) < _SHORT_VALUE_CHARS:
            if value_lower in source_lower:
                idx = source_lower.index(value_lower)
                return GroundedField(
                    value=value,
                    span=Span(pmid=pmid, source=source,
                              char_start=idx, char_end=idx + len(value),
                              text=source_text[idx: idx + len(value)]),
                    confidence=100.0,
                    status="grounded",
                )
            return GroundedField(value=value, span=None, confidence=0.0, status="quarantined")

        # Longer values: token_set_ratio handles word-order differences (paraphrasing)
        # better than partial_ratio, which is character-order sensitive.
        overall_score = fuzz.token_set_ratio(value_lower, source_lower)
        if overall_score < threshold:
            return GroundedField(value=value, span=None, confidence=overall_score, status="quarantined")

        # Locate best-matching span using sentence-level chunks (fast) rather than
        # a character-by-character sliding window (O(n) calls for a 6000-char text).
        sentences = re.split(r"(?<=[.!?])\s+", source_text)
        # Build sentence start offsets
        offsets: list[int] = []
        pos = 0
        for s in sentences:
            offsets.append(pos)
            pos += len(s) + 1  # +1 for the space consumed by the split

        # Score each sentence; use a context window of up to 3 sentences for
        # multi-sentence extractions (e.g. full result descriptions).
        best_score = 0.0
        best_start = 0
        best_end = min(len(value) * 2, len(source_text))
        window_size = 3
        for i in range(len(sentences)):
            chunk = " ".join(sentences[i: i + window_size])
            score = fuzz.token_set_ratio(value_lower, chunk.lower())
            if score > best_score:
                best_score = score
                best_start = offsets[i]
                end_idx = min(i + window_size - 1, len(sentences) - 1)
                best_end = offsets[end_idx] + len(sentences[end_idx])

        return GroundedField(
            value=value,
            span=Span(pmid=pmid, source=source,
                      char_start=best_start, char_end=best_end,
                      text=source_text[best_start:best_end]),
            confidence=overall_score,
            status="grounded",
        )

    def ground_extracted_data(
        self,
        extracted_data: dict,
        source_text: str,
        pmid: str,
        source: Literal["abstract", "fulltext"],
        stage: str,
    ) -> tuple[dict[str, GroundedField], list[QuarantinedField]]:
        grounded: dict[str, GroundedField] = {}
        quarantined: list[QuarantinedField] = []

        for field_name, value in extracted_data.items():
            if not isinstance(value, str) or not value.strip():
                continue
            result = self.ground_field(field_name, value, source_text, pmid, source)
            if result["status"] == "grounded":
                grounded[field_name] = result
            else:
                quarantined.append(QuarantinedField(
                    field_name=field_name,
                    value=value,
                    stage=stage,
                    reason=f"no matching span (score={result['confidence']:.1f})",
                ))

        return grounded, quarantined


class GroundedClaim(TypedDict):
    claim: str
    supporting_pmids: list[str]
    status: Literal["grounded", "quarantined"]
    confidence: str


class SynthesisGrounder:
    def __init__(self, llm):
        self.llm = llm

    def ground_claim(
        self,
        claim: str,
        paper_extractions: list[dict],
    ) -> GroundedClaim:
        extractions_text = "\n".join(
            f"[PMID {p['pmid']}]: {p.get('result', '')}"
            for p in paper_extractions
        )
        messages = [
            {
                "role": "user",
                "content": (
                    f"Is this claim supported by the paper extractions below?\n\n"
                    f"CLAIM: {claim}\n\n"
                    f"EXTRACTIONS:\n{extractions_text}\n\n"
                    "Return JSON with fields: supporting_pmids (list of PMIDs that "
                    "directly support this claim), confidence (high/moderate/low/none)."
                ),
            }
        ]
        schema = {
            "type": "object",
            "properties": {
                "supporting_pmids": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "string"},
            },
            "required": ["supporting_pmids", "confidence"],
        }
        result = self.llm.chat(messages, schema=schema)
        pmids = result.get("supporting_pmids", [])
        status: Literal["grounded", "quarantined"] = (
            "grounded" if pmids else "quarantined"
        )
        return GroundedClaim(
            claim=claim,
            supporting_pmids=pmids,
            status=status,
            confidence=result.get("confidence", "none"),
        )
