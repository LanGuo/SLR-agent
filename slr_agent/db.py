import json
import sqlite3
from contextlib import contextmanager
from typing import Literal
from typing_extensions import TypedDict


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
    source: Literal["abstract", "fulltext"]
    screening_decision: Literal["include", "exclude", "uncertain"]
    screening_reason: str
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
    source TEXT DEFAULT 'abstract',
    screening_decision TEXT DEFAULT 'uncertain',
    screening_reason TEXT DEFAULT '',
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
                   (pmid, run_id, title, abstract, fulltext, source,
                    screening_decision, screening_reason, extracted_data,
                    grade_score, provenance, quarantined_fields)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(pmid, run_id) DO UPDATE SET
                     title=excluded.title, abstract=excluded.abstract,
                     fulltext=excluded.fulltext, source=excluded.source,
                     screening_decision=excluded.screening_decision,
                     screening_reason=excluded.screening_reason,
                     extracted_data=excluded.extracted_data,
                     grade_score=excluded.grade_score,
                     provenance=excluded.provenance,
                     quarantined_fields=excluded.quarantined_fields""",
                (
                    paper["pmid"], paper["run_id"], paper["title"],
                    paper["abstract"], paper.get("fulltext"),
                    paper["source"], paper["screening_decision"],
                    paper["screening_reason"],
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
        return d  # type: ignore[return-value]
