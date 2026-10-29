from hrbot.ingest import PageData, chunk_pages


def test_ac_2_3_chunking_respects_size_overlap_and_page_boundaries() -> None:
    page1_text = "Page1Sentence. " * 120  # ~1800 chars -> 2 chunks with chunk_size=1000
    page2_text = "Page2UniqueMarker only on second page."

    pages = [
        PageData(source="policy.pdf", page=1, doc_title="Policy Doc", text=page1_text),
        PageData(source="policy.pdf", page=2, doc_title="Policy Doc", text=page2_text),
    ]

    chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=150)

    page1_chunks = [c for c in chunks if c.metadata["page"] == 1]
    page2_chunks = [c for c in chunks if c.metadata["page"] == 2]

    assert len(page1_chunks) >= 2
    assert len(page2_chunks) == 1

    for c in chunks:
        assert len(c.page_content) <= 1000

    # Verify chunks never span across pages
    for c in page1_chunks:
        assert "Page2UniqueMarker" not in c.page_content
    for c in page2_chunks:
        assert "Page1Sentence" not in c.page_content

    # Verify overlap exists between consecutive chunks on page 1
    tail_of_first = page1_chunks[0].page_content[-80:]
    assert tail_of_first in page1_chunks[1].page_content

