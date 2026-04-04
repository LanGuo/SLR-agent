import pytest
from unittest.mock import patch, MagicMock
from slr_agent.db import Database, PaperRecord


def test_add_paper_from_pmid_inserts_record(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.ensure_run("run-1")

    with patch("slr_agent.db.Entrez") as mock_entrez:
        mock_entrez.efetch.return_value = MagicMock()
        mock_entrez.read.return_value = {
            "PubmedArticle": [{
                "MedlineCitation": {
                    "PMID": type("P", (), {"__str__": lambda s: "12345"})(),
                    "Article": {
                        "ArticleTitle": "Aspirin trial",
                        "Abstract": {"AbstractText": "Background: aspirin reduces events."},
                    },
                }
            }]
        }
        paper = db.add_paper_from_pmid("run-1", "12345")

    assert paper is not None
    assert paper["title"] == "Aspirin trial"
    assert paper["pmid"] == "12345"
    assert paper["source"] == "manual"
    stored = db.get_paper("run-1", "12345")
    assert stored is not None


def test_add_paper_from_pmid_returns_none_on_error(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.ensure_run("run-1")
    with patch("slr_agent.db.Entrez") as mock_entrez:
        mock_entrez.efetch.side_effect = Exception("network error")
        result = db.add_paper_from_pmid("run-1", "00000")
    assert result is None


def test_add_paper_from_pdf_inserts_record(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.ensure_run("run-1")

    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    with patch("slr_agent.db.fitz") as mock_fitz:
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Full text of paper about aspirin..."
        mock_doc.__iter__ = lambda s: iter([mock_page])
        mock_fitz.open.return_value = mock_doc

        paper = db.add_paper_from_pdf(
            "run-1",
            str(fake_pdf),
            metadata={"pmid": "pdf-001", "title": "Manual PDF Paper"},
        )

    assert paper["pmid"] == "pdf-001"
    assert paper["title"] == "Manual PDF Paper"
    assert "aspirin" in paper["fulltext"]
    assert paper["source"] == "manual"
