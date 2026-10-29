from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from hrbot.ingest import chunk_pages, load_pdf


def test_ac_2_4_chunk_metadata_fields_and_doc_title_fallback(tmp_path: Path) -> None:
    # PDF 1: with explicit metadata title
    titled_pdf = tmp_path / "with_title.pdf"
    c1 = canvas.Canvas(str(titled_pdf), pagesize=A4)
    c1.setTitle("Official Remote Work Policy")
    c1.drawString(72, 750, "Employees can work remotely up to 3 days per week.")
    c1.showPage()
    c1.save()

    # PDF 2: without explicit metadata title (fallback to file name)
    untitled_pdf = tmp_path / "no_title.pdf"
    c2 = canvas.Canvas(str(untitled_pdf), pagesize=A4)
    c2.drawString(72, 750, "Expenses must be submitted within 30 days.")
    c2.showPage()
    c2.save()

    res1 = load_pdf(titled_pdf)
    res2 = load_pdf(untitled_pdf)
    chunks = chunk_pages(res1.pages + res2.pages, chunk_size=1000, chunk_overlap=150)

    assert len(chunks) == 2

    meta1 = chunks[0].metadata
    assert set(meta1.keys()) == {"source", "page", "chunk_index", "doc_title", "chunk_id"}
    assert meta1["source"] == "with_title.pdf"
    assert meta1["page"] == 1
    assert meta1["chunk_index"] == 0
    assert meta1["doc_title"] == "Official Remote Work Policy"
    assert isinstance(meta1["chunk_id"], str) and len(meta1["chunk_id"]) == 40

    meta2 = chunks[1].metadata
    assert meta2["source"] == "no_title.pdf"
    assert meta2["page"] == 1
    assert meta2["chunk_index"] == 0
    assert meta2["doc_title"] == "no_title.pdf"
    assert isinstance(meta2["chunk_id"], str) and len(meta2["chunk_id"]) == 40

