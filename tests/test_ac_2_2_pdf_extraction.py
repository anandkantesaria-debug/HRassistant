import logging
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from hrbot.ingest import load_pdf


def test_ac_2_2_extract_pages_and_warn_on_empty_page(
    tmp_path: Path, caplog: logging.LogCaptureFixture
) -> None:
    pdf_path = tmp_path / "test_policy.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    # Page 1: has text
    c.drawString(72, 750, "Page one content about annual leave.")
    c.showPage()
    # Page 2: blank (no extractable text)
    c.showPage()
    # Page 3: has text
    c.drawString(72, 750, "Page three content about carry-over rules.")
    c.showPage()
    c.save()

    with caplog.at_level(logging.WARNING, logger="hrbot.ingest"):
        result = load_pdf(pdf_path)

    assert result.total_pages == 3
    assert result.skipped_pages == 1
    assert len(result.pages) == 2

    assert result.pages[0].page == 1
    assert "Page one content" in result.pages[0].text
    assert result.pages[1].page == 3
    assert "Page three content" in result.pages[1].text

    assert any(
        "page 2" in record.message.lower() and "test_policy.pdf" in record.message
        for record in caplog.records
    )

