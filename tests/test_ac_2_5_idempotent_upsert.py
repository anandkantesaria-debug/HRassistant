from pathlib import Path

from langchain_core.embeddings import Embeddings
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from hrbot.config import Settings
from hrbot.ingest import compute_chunk_id, index_chunks, ingest_directory


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 10 + 1), 0.5, 0.25, 0.125] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 10 + 1), 0.5, 0.25, 0.125]


def test_ac_2_5_deterministic_chunk_id_and_idempotent_reingest(tmp_path: Path) -> None:
    id1 = compute_chunk_id("leave.pdf", 1, 0, "20 days of annual leave")
    id2 = compute_chunk_id("leave.pdf", 1, 0, "20 days of annual leave")
    id_diff_page = compute_chunk_id("leave.pdf", 2, 0, "20 days of annual leave")
    id_diff_text = compute_chunk_id("leave.pdf", 1, 0, "21 days of annual leave")

    assert id1 == id2
    assert len(id1) == 40
    assert id1 != id_diff_page
    assert id1 != id_diff_text

    policies_dir = tmp_path / "policies"
    policies_dir.mkdir()
    pdf_path = policies_dir / "leave.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    c.drawString(72, 750, "Full-time staff receive 20 days of paid annual leave.")
    c.showPage()
    c.save()

    chroma_dir = tmp_path / "chroma"
    settings = Settings(
        GROQ_API_KEY="test-key",
        LLM_MODEL="test-model",
        CHROMA_DIR=str(chroma_dir),
        COLLECTION_NAME="hr_policies",
        _env_file=None,
    )
    fake_emb = FakeEmbeddings()

    summary1 = ingest_directory(policies_dir, reset=False, settings=settings, embeddings=fake_emb)
    summary2 = ingest_directory(policies_dir, reset=False, settings=settings, embeddings=fake_emb)

    assert summary1.chunks == 1
    assert summary2.chunks == 1

    vs = index_chunks([], settings=settings, embeddings=fake_emb, reset=False)
    assert vs._collection.count() == 1

