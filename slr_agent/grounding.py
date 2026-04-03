from typing import Literal
from typing_extensions import TypedDict
from rapidfuzz import fuzz
from slr_agent.db import Span, QuarantinedField


class GroundedField(TypedDict):
    value: str
    span: Span | None
    confidence: float
    status: Literal["grounded", "quarantined"]


class ExtractionGrounder:
    def __init__(self, threshold: int = 85):
        self.threshold = threshold

    def ground_field(
        self,
        field_name: str,
        value: str,
        source_text: str,
        pmid: str,
        source: Literal["abstract", "fulltext"],
    ) -> GroundedField:
        # First check overall score against full source text using partial_ratio
        # (partial_ratio finds the best matching substring internally)
        overall_score = fuzz.partial_ratio(value.lower(), source_text.lower())

        if overall_score < self.threshold:
            return GroundedField(value=value, span=None, confidence=overall_score, status="quarantined")

        # Locate the best-matching span by sliding a window of ~len(value) chars
        window = max(len(value), 30)
        best_score = 0.0
        best_start = 0

        for i in range(0, max(1, len(source_text) - window + 1), 1):
            chunk = source_text[i : i + window]
            score = fuzz.partial_ratio(value.lower(), chunk.lower())
            if score > best_score:
                best_score = score
                best_start = i

        end = min(best_start + window, len(source_text))
        return GroundedField(
            value=value,
            span=Span(
                pmid=pmid,
                source=source,
                char_start=best_start,
                char_end=end,
                text=source_text[best_start:end],
            ),
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
