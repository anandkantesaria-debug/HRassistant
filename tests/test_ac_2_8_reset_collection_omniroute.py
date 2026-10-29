from pathlib import Path

from langchain_core.embeddings import Embeddings
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from hrbot.config import Settings
from hrbot.ingest import index_chunks, ingest_directory


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.3, 0.1, 0.4, 0.1] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.3, 0.1, 0.4, 0.1]


def _write_single_page_pdf(path: Path, text: str) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)
    c.drawString(72, 750, text)
    c.showPage()
    c.save()


def test_ac_2_8_reset_deletes_removed_pdfs_from_index(tmp_path: Path) -> None:
    policies_dir = tmp_path / "policies"
    policies_dir.mkdir()

    pdf_a = policies_dir / "policy_a.pdf"
    pdf_b = policies_dir / "policy_b.pdf"
    _write_single_page_pdf(pdf_a, "Policy A content about annual vacation.")
    _write_single_page_pdf(pdf_b, "Policy B content about travel reimbursement.")

    settings = Settings(
        OMNIROUTE_API_KEY="test-key",
        LLM_MODEL="test-model",
        CHROMA_DIR=str(tmp_path / "chroma"),
        COLLECTION_NAME="hr_policies",
        _env_file=None,
    )
    fake_emb = FakeEmbeddings()

    # Initial ingest with both PDFs
    ingest_directory(policies_dir, reset=False, settings=settings, embeddings=fake_emb)
    vs1 = index_chunks([], settings=settings, embeddings=fake_emb, reset=False)
    items1 = vs1._collection.get()
    sources1 = {m["source"] for m in items1["metadatas"]}
    assert sources1 == {"policy_a.pdf", "policy_b.pdf"}

    # Remove policy_b.pdf and re-ingest with reset=True
    pdf_b.unlink()
    ingest_directory(policies_dir, reset=True, settings=settings, embeddings=fake_emb)

    vs2 = index_chunks([], settings=settings, embeddings=fake_emb, reset=False)
    items2 = vs2._collection.get()
    sources2 = {m["source"] for m in items2["metadatas"]}
    assert sources2 == {"policy_a.pdf"}
    assert len(items2["ids"]) == 1

