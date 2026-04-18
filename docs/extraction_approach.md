# Extraction Approach — Design Notes and Comparison with LangExtract

This document describes how structured data extraction works in the SLR pipeline, how it differs from LangExtract, and what would be needed to isolate the extraction layer as a standalone package.

---

## What we built

Extraction in this pipeline is a four-step sequence per paper:

```
LLM call (constrained decoding)
  → fuzzy grounding against source text
    → auto LLM re-verification for failed fields
      → quarantine (with HITL resolution at Gate 5)
```

Each step is independent and the output of each feeds the next. The LLM is only responsible for semantic interpretation; correctness of the output is enforced structurally rather than by trusting the model.

---

## Step 1 — LLM → structured output via constrained decoding

**How it works:**

`LLMClient.chat()` passes a JSON Schema dict as `format=schema` to Ollama (`llm.py:41`). Ollama converts this to a GBNF (Generalized Backus-Naur Form) grammar and applies it as a hard constraint during token sampling via llama.cpp. At each sampling step, any token that would produce an invalid JSON continuation is masked out. The model always sees only valid continuations.

The output arrives as a JSON string and is `json.loads`'d directly. No post-processing parser, no regex fallback.

```python
# llm.py — the entirety of structured output enforcement
if schema:
    kwargs["format"] = schema
response = ollama.chat(**kwargs)
result = json.loads(response["message"]["content"])
```

**On parse failure (rare):**

If `json.loads` still fails (malformed UTF-8, truncated response), the bad output is injected back as an assistant message and a correction turn is appended. This retries up to `max_retries` times with exponential backoff. In practice with constrained decoding this path is almost never hit.

**What the model needs:**

Nothing special. Constrained decoding is enforced by the runtime (Ollama/llama.cpp), not the model weights. Any model Ollama serves — base models included — will produce structurally valid output. The model only needs to understand the *semantics* of what to put in each field.

**Extraction schema (`extraction.py`):**

Eight PICO-aligned fields, all strings:

```
sample_size, intervention, comparator, primary_outcome,
result, study_design, follow_up_duration, population_details
```

The prompt injects the review's PICO (P/I/C/O) so the model interprets ambiguous fields — e.g., which arm is the "intervention" — in the context of the specific review question.

**Multimodal:**

When PDF page images are available (from the full-text fetch stage), they are base64-encoded and passed as `images=` in the message dict. The model can then read tables and figures that don't parse cleanly into plain text.

---

## Step 2 — Fuzzy grounding against source text

**Why:**

The LLM output is treated as a *draft*, not a final answer. Every field value is verified against the source text (abstract or full-text) before being stored. This catches hallucinated values, values from the wrong paper (copy-paste confabulation), or values that are technically valid JSON but not actually present in the source.

**How:**

`ExtractionGrounder` (in `grounding.py`) runs rapidfuzz `token_set_ratio` on each field value against the source text. Thresholds are source-adaptive:

| Source | Threshold | Rationale |
|---|---|---|
| Abstract | 75 | Abstracts are summaries — values may be paraphrased |
| Full text | 85 | Full text is verbatim — higher bar appropriate |

Short values under 20 characters use exact substring search instead of fuzzy matching.

On match, a `Span` is stored recording `char_start`, `char_end`, and the matched text, tagged with `provenance_type`:

| Type | Meaning |
|---|---|
| `direct` | Exact substring — value appears verbatim |
| `paraphrased` | Fuzzy match above threshold |
| `inferred` | Failed fuzzy but LLM-confirmed (see Step 3) |

Fields that pass grounding enter `extracted_data`. Fields that fail go to Step 3.

---

## Step 3 — Auto LLM re-verification before quarantine

**Why:**

Fuzzy matching is conservative by design. It rejects valid extractions that are heavily paraphrased, abbreviated, or numerically expressed differently from the source (e.g. `"96"` vs `"ninety-six"`, `"MI"` vs `"myocardial infarction"`). Without a second pass, all of these would need human resolution.

**How:**

`_auto_llm_ground()` sends each quarantined field back to the LLM with the question: *"Does the source text support this value, even if phrased differently?"* The LLM returns `{supported: bool, span: string}`.

- `supported: true` → field is promoted into `extracted_data` with `confidence=80.0` and `provenance_type="inferred"`
- `supported: false` → field proceeds to quarantine

The whole block is wrapped in try/except — a grounding failure leaves the field quarantined rather than aborting the extraction run.

---

## Step 4 — Quarantine

Fields that failed both fuzzy grounding and LLM re-verification are written to a `quarantine` SQLite table. They are:

- **Kept**, not dropped — the value is preserved for human inspection
- **Surfaced at Gate 5** — the reviewer sees each quarantined field and can accept, edit, or discard it
- **Auditable** — the full quarantine table persists in the database

---

## Step 5 — GRADE evidence quality scoring

A separate LLM call per paper with `think=True` (Gemma 4 extended reasoning) assesses evidence quality across five GRADE dimensions: certainty, risk of bias, inconsistency, indirectness, imprecision. This is structurally identical to Step 1 (constrained decoding into a fixed schema) but uses the thinking chain for careful judgment.

---

## Comparison with LangExtract

LangExtract is LangChain's framework for extracting structured data from text. The comparison below reflects LangChain's extraction approach as of early 2026.

### Structured output mechanism

| | This pipeline | LangExtract |
|---|---|---|
| Mechanism | Ollama `format=` — constrained decoding at sampling time (GBNF grammar via llama.cpp) | Function calling / tool use (OpenAI, Anthropic) or prompt + output parser |
| Schema definition | Raw JSON Schema dict | Pydantic model → auto-converted to function schema |
| Enforcement | Hard — invalid tokens masked at sampling | Soft on non-FC models — parser can fail, triggers `OutputFixingParser` |
| Model requirement | None — any model Ollama serves | Requires fine-tuned function-calling support for reliable structured output |
| Retry on failure | Inject bad output + correction turn, exponential backoff | `OutputFixingParser` re-prompts; `PydanticOutputParser` raises on failure |
| Provider coupling | Ollama only (local, any model) | OpenAI / Anthropic API or LangChain-supported providers |

### Post-extraction verification

| | This pipeline | LangExtract |
|---|---|---|
| Source grounding | Mandatory fuzzy match against source text | None — LLM output is trusted |
| Provenance | Character-level spans with `provenance_type` | Not tracked |
| Failed field handling | Auto LLM re-verification → quarantine → HITL | No equivalent — field is returned or absent |
| Audit trail | SQLite quarantine table + `llm_trace.jsonl` | Not built in |

### Other capabilities

| | This pipeline | LangExtract |
|---|---|---|
| Thinking chain | `think=True` flag, reasoning saved to trace | No equivalent |
| Multimodal | `images=` in message dict (base64 PNGs) | Separate multimodal chain setup required |
| Domain quality scoring | GRADE per paper (separate LLM call) | Not built in |
| HITL gate | Gate 5 — human reviews quarantined fields | Not built in |

### Summary

LangExtract is a general-purpose "schema → LLM → structured object" pipeline. The extraction layer here is "schema → LLM → structured object → provenance verification → auto re-verification → quarantine → evidence quality scoring → human gate". The LLM output is treated as a draft requiring corroboration, not a final answer. The grounding layer is where most of the correctness guarantee comes from — constrained decoding ensures the *shape* is right, grounding ensures the *content* is traceable to the source.

---

## What isolation as a standalone package would require

The extraction-plus-grounding layer is relatively self-contained. The dependencies that would need to be severed or abstracted:

**Keep (core extraction logic):**
- `llm.py` — `LLMClient` (Ollama wrapper with constrained decoding + retry)
- `grounding.py` — `ExtractionGrounder` (rapidfuzz grounding + span tracking)
- `subgraphs/extraction.py` — schema, `_auto_llm_ground`, `_extract_node`
- `db.py` types — `Span`, `GroundedField`, `QuarantinedField` TypedDicts (no DB needed)

**Decouple or make optional:**
- `Database` — the extraction node writes directly to SQLite; a standalone package should accept a callback or return results rather than writing to a DB
- `GRADEScore` — domain-specific to systematic reviews; should be optional or pluggable
- LangGraph — `StateGraph` is used only for the subgraph wiring; extraction logic itself has no graph dependency, it's all in `_extract_node`
- `ExtractionCounts` / `OrchestratorState` — pipeline-specific state; a standalone package should return a plain dataclass or dict

**The core API a standalone package would expose:**

```python
result = extract(
    text=source_text,
    schema={"sample_size": str, "intervention": str, ...},
    llm=llm_client,
    images=[],           # optional base64 PNGs
    grounding=True,      # whether to run fuzzy grounding
    auto_reground=True,  # whether to run LLM re-verification on failures
)
# result.extracted   — grounded field values
# result.quarantined — fields that failed grounding
# result.provenance  — Span objects with char_start/char_end/provenance_type
```

The grounding layer is the most novel and reusable part — the combination of fuzzy match → LLM re-verification → typed provenance spans is not present in any existing extraction library and would be the primary value proposition of a standalone package.
