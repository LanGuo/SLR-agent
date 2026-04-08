# slr_agent/subgraphs/fulltext.py
import os
import time
import xml.etree.ElementTree as ET
from Bio import Entrez
from langgraph.graph import StateGraph, END
from slr_agent.db import Database
from slr_agent.grounding import ExtractionGrounder
from slr_agent.state import FulltextCounts


def fetch_pmc_fulltext(pmid: str) -> tuple[str | None, str | None]:
    """Fetch full text from PubMed Central.

    Returns (fulltext_xml_str, pmc_id) or (None, None) if unavailable.
    pmc_id is returned so callers can attempt PDF download without a second elink call.
    """
    try:
        handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid)
        record = Entrez.read(handle)
        handle.close()
        link_sets = record[0].get("LinkSetDb", [])
        if not link_sets:
            return None, None
        pmc_ids = [lnk["Id"] for lnk in link_sets[0].get("Link", [])]
        if not pmc_ids:
            return None, None
        pmc_id = pmc_ids[0]

        handle = Entrez.efetch(db="pmc", id=pmc_id, rettype="full", retmode="xml")
        text = handle.read()
        handle.close()
        fulltext = text.decode("utf-8") if isinstance(text, bytes) else text
        return fulltext, pmc_id
    except Exception:
        return None, None


def fetch_pmc_pdf_images(
    pmc_id: str, output_dir: str, run_id: str, pmid: str, max_pages: int = 10
) -> list[str]:
    """Download the open-access PDF from PMC and convert the first N pages to PNG images.

    Uses the PMC OA API to locate the PDF link, downloads it, then renders each page
    at 150 DPI using PyMuPDF. Images are saved to ``output_dir/<run_id>/pages/``.

    Returns a list of absolute PNG file paths, or [] if the PDF is unavailable.
    """
    try:
        import httpx
        import fitz  # PyMuPDF

        oa_url = (
            f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
            f"?id=PMC{pmc_id}&format=pdf"
        )
        resp = httpx.get(oa_url, timeout=15, follow_redirects=True)
        root = ET.fromstring(resp.text)
        link_el = root.find('.//link[@format="pdf"]')
        if link_el is None:
            return []
        pdf_href = link_el.get("href", "")
        if not pdf_href:
            return []
        # PMC OA returns FTP URLs; convert to HTTPS for httpx
        if pdf_href.startswith("ftp://"):
            pdf_href = "https://" + pdf_href[6:]

        pdf_resp = httpx.get(pdf_href, timeout=90, follow_redirects=True)
        pdf_resp.raise_for_status()

        pages_dir = os.path.join(output_dir, run_id, "pages")
        os.makedirs(pages_dir, exist_ok=True)

        doc = fitz.open(stream=pdf_resp.content, filetype="pdf")
        image_paths: list[str] = []
        for page_num in range(min(max_pages, len(doc))):
            pix = doc[page_num].get_pixmap(dpi=150)
            img_path = os.path.join(pages_dir, f"{pmid}_p{page_num}.png")
            pix.save(img_path)
            image_paths.append(img_path)
        doc.close()
        return image_paths
    except Exception:
        return []


def _fetch_fulltext_node(state: dict, db: Database, llm, output_dir: str) -> dict:
    run_id = state["run_id"]
    pico = state["pico"]
    included = db.get_papers_by_decision(run_id, "include")
    grounder = ExtractionGrounder()

    n_fetched = n_unavailable = n_excluded = 0

    for paper in included:
        pmid = paper["pmid"]
        fulltext, pmc_id = fetch_pmc_fulltext(pmid)

        if fulltext is None:
            n_unavailable += 1
            time.sleep(0.34)
            continue

        time.sleep(0.34)

        # Attempt to fetch PDF page images for multimodal extraction
        page_image_paths: list[str] = []
        if pmc_id:
            page_image_paths = fetch_pmc_pdf_images(
                pmc_id=pmc_id,
                output_dir=output_dir,
                run_id=run_id,
                pmid=pmid,
            )

        # Screen full text with thinking enabled for careful reasoning
        result = llm.chat([{
            "role": "user",
            "content": (
                f"Screen this full text for a systematic review.\n"
                f"PICO: P={pico['population']}, I={pico['intervention']}, "
                f"C={pico['comparator']}, O={pico['outcome']}\n\n"
                f"Full text (first 8000 chars):\n{fulltext[:8000]}\n\n"
                "Return JSON with fields: decision (include/exclude/uncertain), reason."
            ),
        }], schema={
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": ["include", "exclude", "uncertain"]},
                "reason": {"type": "string"},
            },
            "required": ["decision", "reason"],
        }, think=True)

        dec = result["decision"]
        reason = result["reason"]

        _, quarantined_fields = grounder.ground_extracted_data(
            {"screening_reason": reason},
            source_text=fulltext[:8000],
            pmid=pmid,
            source="fulltext",
            stage="fulltext_screening",
        )
        for qf in quarantined_fields:
            db.insert_quarantine(run_id, pmid, qf)

        paper["fulltext"] = fulltext
        paper["page_image_paths"] = page_image_paths
        paper["source"] = "fulltext"
        paper["screening_decision"] = dec
        paper["screening_reason"] = reason
        paper["quarantined_fields"] = quarantined_fields
        db.upsert_paper(paper)

        n_fetched += 1
        if dec == "exclude":
            n_excluded += 1

    return {
        "fulltext_counts": FulltextCounts(
            n_fetched=n_fetched,
            n_unavailable=n_unavailable,
            n_excluded=n_excluded,
        )
    }


def create_fulltext_subgraph(db: Database, llm, output_dir: str = "outputs"):
    builder = StateGraph(dict)
    builder.add_node(
        "fetch_fulltext",
        lambda s: _fetch_fulltext_node(s, db, llm, output_dir),
    )
    builder.set_entry_point("fetch_fulltext")
    builder.add_edge("fetch_fulltext", END)
    return builder.compile()
