from pathlib import Path
from unittest.mock import patch

from hrbot.ingest import build_parser, find_pdfs, main


def test_ac_2_1_find_pdfs_recursively_and_cli_args(tmp_path: Path) -> None:
    sub1 = tmp_path / "dept_a"
    sub2 = tmp_path / "dept_b" / "nested"
    sub1.mkdir(parents=True)
    sub2.mkdir(parents=True)

    pdf_root = tmp_path / "root_policy.pdf"
    pdf_sub1 = sub1 / "leave_policy.pdf"
    pdf_sub2 = sub2 / "expense_policy.PDF"
    txt_file = sub1 / "notes.txt"
    md_file = tmp_path / "readme.md"

    for f in (pdf_root, pdf_sub1, pdf_sub2, txt_file, md_file):
        f.write_bytes(b"placeholder")

    found = find_pdfs(tmp_path)
    assert found == sorted([pdf_root, pdf_sub1, pdf_sub2])

    parser = build_parser()
    args = parser.parse_args(["--path", str(tmp_path), "--reset"])
    assert args.path == str(tmp_path)
    assert args.reset is True

    with patch("hrbot.ingest.ingest_directory") as mock_ingest:
        rc = main(["--path", str(tmp_path), "--reset"])
        assert rc == 0
        mock_ingest.assert_called_once_with(path=str(tmp_path), reset=True)

