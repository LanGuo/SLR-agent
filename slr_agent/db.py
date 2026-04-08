import json
import sqlite3
import os
from contextlib import contextmanager
from typing import Literal
from typing_extensions import TypedDict

import fitz  # PyMuPDF
from Bio import Entrez


class Span(TypedDict):
    pmid: str
    source: Literal["abstract", "fulltext"]
    char_start: int
    char_end: int
    text: str


class QuarantinedField(TypedDict):
    field_name: str
    value: str
    stage: str
    reason: str


class GRADEScore(TypedDict):
    certainty: Literal["high", "moderate", "low", "very_low"]
    risk_of_bias: Literal["low", "some_concerns", "high"]
    inconsistency: Literal["no", "some", "serious"]
    indirectness: Literal["no", "some", "serious"]
    imprecision: Literal["no", "some", "serious"]
    rationale: str


class PaperRecord(TypedDict):
    pmid: str
    run_id: str
    title: str
    abstract: str
    fulltext: str | None
    page_image_paths: list[str]   # paths to PNG page renders for multimodal extraction
    source: Literal["abstract", "fulltext"]
    screening_decision: Literal["include", "exclude", "uncertain"]
    screening_reason: str
    criterion_scores: list[dict]   # per-criterion scores from screening LLM
    extracted_data: dict
    grade_score: GRADEScore
    provenance: list[Span]
    quarantined_fields: list[QuarantinedField]


_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT DEFAULT (datetime('now')),
    current_stage TEXT DEFAULT 'pico'
);
CREATE TABLE IF NOT EXISTS papers (
    pmid TEXT NOT NULL,
    run_id TEXT NOT NULL,
    title TEXT,
    abstract TEXT,
    fulltext TEXT,
    page_image_paths TEXT DEFAULT '[]',
    source TEXT DEFAULT 'abstract',
    screening_decision TEXT DEFAULT 'uncertain',
    screening_reason TEXT DEFAULT '',
    criterion_scores TEXT DEFAULT '[]',
    extracted_data TEXT DEFAULT '{}',
    grade_score TEXT DEFAULT '{}',
    provenance TEXT DEFAULT '[]',
    quarantined_fields TEXT DEFAULT '[]',
    PRIMARY KEY (pmid, run_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
CREATE TABLE IF NOT EXISTS quarantine (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    pmid TEXT NOT NULL,
    field_name TEXT NOT NULL,
    value TEXT NOT NULL,
    stage TEXT NOT NULL,
    reason TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
"""


class Database:
    def __init__(self, path: str):
        self.path = path
        with self.connect() as conn:
            conn.executescript(_SCHEMA)
            # Migrate existing databases: add page_image_paths if not present
            cols = {r[1] for r in conn.execute("PRAGMA table_info(papers)").fetchall()}
            if "page_image_paths" not in cols:
                conn.execute("ALTER TABLE papers ADD COLUMN page_image_paths TEXT DEFAULT '[]'")
            if "criterion_scores" not in cols:
                conn.execute("ALTER TABLE papers ADD COLUMN criterion_scores TEXT DEFAULT '[]'")

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def upsert_paper(self, paper: PaperRecord) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO papers
                   (pmid, run_id, title, abstract, fulltext, page_image_paths, source,
                    screening_decision, screening_reason, criterion_scores, extracted_data,
                    grade_score, provenance, quarantined_fields)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(pmid, run_id) DO UPDATE SET
                     title=excluded.title, abstract=excluded.abstract,
                     fulltext=excluded.fulltext,
                     page_image_paths=excluded.page_image_paths,
                     source=excluded.source,
                     screening_decision=excluded.screening_decision,
                     screening_reason=excluded.screening_reason,
                     criterion_scores=excluded.criterion_scores,
                     extracted_data=excluded.extracted_data,
                     grade_score=excluded.grade_score,
                     provenance=excluded.provenance,
                     quarantined_fields=excluded.quarantined_fields""",
                (
                    paper["pmid"], paper["run_id"], paper["title"],
                    paper["abstract"], paper.get("fulltext"),
                    json.dumps(paper.get("page_image_paths") or []),
                    paper["source"], paper["screening_decision"],
                    paper["screening_reason"],
                    json.dumps(paper.get("criterion_scores") or []),
                    json.dumps(paper["extracted_data"]),
                    json.dumps(paper["grade_score"]),
                    json.dumps(paper["provenance"]),
                    json.dumps(paper["quarantined_fields"]),
                ),
            )

    def get_paper(self, run_id: str, pmid: str) -> PaperRecord | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM papers WHERE run_id=? AND pmid=?",
                (run_id, pmid),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_paper(row)

    def get_papers_by_decision(
        self, run_id: str, decision: Literal["include", "exclude", "uncertain"]
    ) -> list[PaperRecord]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM papers WHERE run_id=? AND screening_decision=?",
                (run_id, decision),
            ).fetchall()
        return [self._row_to_paper(r) for r in rows]

    def get_all_papers(self, run_id: str) -> list[PaperRecord]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM papers WHERE run_id=?", (run_id,)
            ).fetchall()
        return [self._row_to_paper(r) for r in rows]

    def insert_quarantine(
        self, run_id: str, pmid: str, field: QuarantinedField
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO quarantine (run_id,pmid,field_name,value,stage,reason)"
                " VALUES (?,?,?,?,?,?)",
                (run_id, pmid, field["field_name"], field["value"],
                 field["stage"], field["reason"]),
            )

    def get_quarantine(self, run_id: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM quarantine WHERE run_id=?", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def add_paper_from_pmid(
        self, run_id: str, pmid: str, api_key: str | None = None
    ) -> "PaperRecord | None":
        """Fetch a paper from PubMed by PMID and insert it. Returns None on failure."""
        if api_key:
            Entrez.api_key = api_key
        Entrez.email = "slr-agent@local"
        try:
            handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="xml")
            try:
                records = Entrez.read(handle)
            finally:
                handle.close()
            articles = records.get("PubmedArticle", [])
            if not articles:
                return None
            citation = articles[0]["MedlineCitation"]
            art = citation.get("Article", {})
            title = str(art.get("ArticleTitle", ""))
            abstract_obj = art.get("Abstract", {})
            abstract_raw = abstract_obj.get("AbstractText", "") if abstract_obj else ""
            abstract = " ".join(str(s) for s in abstract_raw) if isinstance(abstract_raw, list) else str(abstract_raw)
        except Exception:
            return None

        paper = PaperRecord(
            pmid=str(pmid),
            run_id=run_id,
            title=title,
            abstract=abstract,
            fulltext=None,
            page_image_paths=[],
            source="manual",
            screening_decision="uncertain",
            screening_reason="Manually added by user",
            criterion_scores=[],
            extracted_data={},
            grade_score=GRADEScore(
                certainty="low", risk_of_bias="high",
                inconsistency="no", indirectness="no",
                imprecision="no", rationale="Not yet assessed",
            ),
            provenance=[],
            quarantined_fields=[],
        )
        self.upsert_paper(paper)
        return paper

    def add_paper_from_pdf(
        self, run_id: str, pdf_path: str, metadata: dict
    ) -> "PaperRecord":
        """Extract fulltext from a PDF and insert as a manually added paper."""
        with fitz.open(pdf_path) as doc:
            fulltext = "\n".join(page.get_text() for page in doc)

            paper = PaperRecord(
                pmid=metadata.get("pmid", f"pdf:{os.path.basename(pdf_path)}"),
                run_id=run_id,
                title=metadata.get("title", os.path.basename(pdf_path)),
                abstract=metadata.get("abstract", fulltext[:500]),
                fulltext=fulltext,
                page_image_paths=[],
                source="manual",
                screening_decision="uncertain",
                screening_reason="Manually uploaded PDF",
                criterion_scores=[],
                extracted_data={},
                grade_score=GRADEScore(
                    certainty="low", risk_of_bias="high",
                    inconsistency="no", indirectness="no",
                    imprecision="no", rationale="Not yet assessed",
                ),
                provenance=[],
                quarantined_fields=[],
            )
            self.upsert_paper(paper)
            return paper

    def ensure_run(self, run_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO runs (run_id) VALUES (?)", (run_id,)
            )

    def _row_to_paper(self, row: sqlite3.Row) -> PaperRecord:
        d = dict(row)
        d["extracted_data"] = json.loads(d["extracted_data"])
        d["grade_score"] = json.loads(d["grade_score"])
        d["provenance"] = json.loads(d["provenance"])
        d["quarantined_fields"] = json.loads(d["quarantined_fields"])
        d["page_image_paths"] = json.loads(d.get("page_image_paths") or "[]")
        d["criterion_scores"] = json.loads(d.get("criterion_scores") or "[]")
        return d  # type: ignore[return-value]
