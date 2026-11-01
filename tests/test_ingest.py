from __future__ import annotations

from pathlib import Path

from rag.config import POLICY_DIR
from rag.ingest import Document, chunk_document, load_file, load_documents
from rag.pipeline import KnowledgeBase, _active_policy_documents


def test_unbroken_text_never_exceeds_chunk_size():
    doc = Document("doc", "Title", "long.txt", "x" * 3000, kind="txt")
    chunks = chunk_document(doc, size=500, overlap=0)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 500 for chunk in chunks)


def test_split_table_fragments_repeat_the_table_header():
    text = (
        "# Rates\n\n## Products\n\n"
        "| Product | Rate |\n|---|---|\n"
        "| Savings account | 3.25% |\n"
        "| Personal loan | 12.50% |\n"
        "| Vehicle loan | 9.75% |\n"
    )
    chunks = chunk_document(Document("doc", "Rates", "rates.md", text), size=65, overlap=0)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 65 for chunk in chunks)
    assert all("| Product | Rate |" in chunk.text for chunk in chunks)


def test_document_ids_include_file_location_and_markdown_metadata(tmp_path: Path):
    first = tmp_path / "policies"
    second = tmp_path / "uploads"
    first.mkdir()
    second.mkdir()
    content = (
        "---\npolicy_id: home-loans\nversion: 2.1\neffective_date: 2026-03-01\n"
        "supersedes: 2.0\n---\n# Home Loans\n\n## Rates\nRate details.\n"
    )
    (first / "policy.md").write_text(content, encoding="utf-8")
    (second / "policy.md").write_text(content, encoding="utf-8")

    bundled = load_file(first / "policy.md")
    uploaded = load_file(second / "policy.md")

    assert bundled is not None and uploaded is not None
    assert bundled.doc_id != uploaded.doc_id
    assert bundled.policy_id == "home-loans"
    assert bundled.version == "2.1"
    assert bundled.effective_date == "2026-03-01"
    assert bundled.supersedes == ["2.0"]
    chunk = chunk_document(bundled)[0]
    assert chunk.version == "2.1"
    assert "effective 2026-03-01" in chunk.label


def test_bundled_policy_header_metadata_is_attached_to_chunks():
    document = load_file(POLICY_DIR / "05_home_loans.md")

    assert document is not None
    assert document.policy_id == "PD-05"
    assert document.version == "7.1"
    assert document.effective_date == "2026-04-01"
    assert all(chunk.version == "7.1" for chunk in chunk_document(document))


def test_scanned_and_malformed_pdfs_are_skipped(tmp_path: Path, monkeypatch, capsys):
    scanned = tmp_path / "scanned.pdf"
    malformed = tmp_path / "malformed.pdf"
    scanned.write_bytes(b"placeholder")
    malformed.write_bytes(b"placeholder")
    def fake_read(path):
        if path == scanned:
            return [""]
        raise ValueError("invalid PDF")

    monkeypatch.setattr("rag.ingest.read_pdf_pages", fake_read)
    assert load_file(scanned) is None
    errors = []
    documents = load_documents([tmp_path], errors)
    assert documents == []
    assert any("no extractable text" in error for error in errors)
    assert any("invalid PDF" in error for error in errors)
    assert "invalid PDF" in capsys.readouterr().out


def test_policy_version_resolution_uses_latest_date_and_rejects_ambiguity():
    from rag.ingest import Document

    old = Document("old", "Policy", "old.md", "old", policy_id="same",
                   version="1", effective_date="2025-01-01")
    current = Document("current", "Policy", "current.md", "new", policy_id="same",
                       version="2", effective_date="2026-01-01")
    active, conflicts = _active_policy_documents([old, current])
    assert active == [current]
    assert conflicts == []

    undated_a = Document("a", "Policy", "a.md", "a", policy_id="ambiguous")
    undated_b = Document("b", "Policy", "b.md", "b", policy_id="ambiguous")
    active, conflicts = _active_policy_documents([undated_a, undated_b])
    assert active == []
    assert len(conflicts) == 1

    replaced = Document("replaced", "Policy", "old.md", "old", policy_id="replaced",
                        version="1")
    replacement = Document("replacement", "Policy", "new.md", "new", policy_id="replaced",
                          version="2", supersedes=["1"])
    active, conflicts = _active_policy_documents([replaced, replacement])
    assert active == [replacement]
    assert conflicts == []


def test_ambiguous_policy_family_is_excluded_without_breaking_empty_index(tmp_path: Path):
    (tmp_path / "old.md").write_text(
        "---\npolicy_id: shared-policy\nversion: 1\n---\n# Shared policy\nOld rule.\n",
        encoding="utf-8",
    )
    (tmp_path / "new.md").write_text(
        "---\npolicy_id: shared-policy\nversion: 2\n---\n# Shared policy\nNew rule.\n",
        encoding="utf-8",
    )

    knowledge_base = KnowledgeBase([tmp_path], embedder_preference="tfidf").build()

    assert knowledge_base.policy_conflicts
    result = knowledge_base.retriever.retrieve("What is the policy?")
    assert not result.answerable
