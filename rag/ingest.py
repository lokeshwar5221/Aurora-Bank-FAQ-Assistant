"""Module 1 - Policy Document Ingestion.

Loads .md / .txt / .pdf policy documents from one or more folders and splits
them into retrieval-friendly chunks.

Chunking strategy (important for RAG quality):
  * Markdown documents are split on their headings first, so a chunk never
    mixes two unrelated topics (e.g. "Home loan fees" and "Home loan tax").
  * Each section is then packed into chunks of at most CHUNK_SIZE characters,
    keeping paragraphs and tables together, with a small overlap.
  * Every chunk remembers its document title, section and source file, so the
    answer generator can cite exactly where a fact came from.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .config import CHUNK_OVERLAP, CHUNK_SIZE, SUPPORTED_EXTENSIONS

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*?)\s*$")


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------
@dataclass
class Document:
    doc_id: str
    title: str
    source: str            # file name
    text: str
    kind: str = "md"       # md | txt | pdf


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    source: str
    section: str
    text: str
    page: int | None = None

    @property
    def embed_text(self) -> str:
        """Text that is embedded: title + section give the chunk extra context."""
        return f"{self.doc_title} - {self.section}\n{self.text}"

    @property
    def label(self) -> str:
        loc = f" (page {self.page})" if self.page else ""
        return f"{self.doc_title} > {self.section}{loc}"

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _prettify(stem: str) -> str:
    stem = re.sub(r"^\d+[_\-\s]*", "", stem)
    return stem.replace("_", " ").replace("-", " ").strip().title() or "Untitled"


def _make_id(source: str) -> str:
    return hashlib.md5(source.encode("utf-8")).hexdigest()[:8]


def read_pdf_pages(data: bytes | Path) -> list[str]:
    """Extract text page by page from a PDF given as a path or raw bytes."""
    import io

    from pypdf import PdfReader  # lazy import: only needed for PDFs

    source = str(data) if isinstance(data, Path) else io.BytesIO(data)
    reader = PdfReader(source)
    return [(page.extract_text() or "") for page in reader.pages]


def load_file(path: Path) -> Document | None:
    """Load one file. Returns None for unsupported / empty files."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return None
    if suffix == ".pdf":
        pages = read_pdf_pages(path)
        # Keep page markers so chunks can cite page numbers.
        text = "\n".join(f"<<PAGE {i}>>\n{p}" for i, p in enumerate(pages, start=1))
        kind = "pdf"
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
        kind = "md" if suffix == ".md" else "txt"
    if not text.strip():
        return None

    title = _prettify(path.stem)
    if kind == "md":  # prefer the first "# Heading" as the title
        for line in text.splitlines():
            m = HEADING_RE.match(line)
            if m and len(m.group(1)) == 1:
                title = m.group(2)
                break
    return Document(doc_id=_make_id(path.name), title=title, source=path.name, text=text, kind=kind)


def load_documents(folders: Iterable[Path]) -> list[Document]:
    docs: list[Document] = []
    for folder in folders:
        folder = Path(folder)
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and not path.name.startswith("."):
                try:
                    doc = load_file(path)
                except Exception as exc:  # a broken PDF must not crash the whole app
                    print(f"[ingest] skipped {path.name}: {exc}")
                    continue
                if doc:
                    docs.append(doc)
    return docs


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def _split_sections(doc: Document) -> list[tuple[str, int | None, str]]:
    """Return [(section_title, page, body_text)] for a document."""
    if doc.kind == "pdf":
        sections = []
        parts = re.split(r"<<PAGE (\d+)>>\n", doc.text)
        # parts = ['', '1', 'text', '2', 'text', ...]
        for i in range(1, len(parts), 2):
            page = int(parts[i])
            body = parts[i + 1].strip()
            if body:
                sections.append((f"Page {page}", page, body))
        return sections

    sections: list[tuple[str, int | None, str]] = []
    path: list[str] = []
    buf: list[str] = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(path) if path else "Overview", None, body))
        buf.clear()

    for line in doc.text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            level, heading = len(m.group(1)), m.group(2)
            flush()
            if level == 1:       # document title - not a section by itself
                path.clear()
            else:
                del path[level - 2:]
                path.append(heading)
        else:
            buf.append(line)
    flush()
    return sections


def _split_long(paragraph: str, size: int) -> list[str]:
    """Break an over-long paragraph on lines, then on sentences."""
    pieces: list[str] = []
    for line in paragraph.split("\n"):
        if len(line) <= size:
            pieces.append(line)
        else:
            pieces.extend(s for s in re.split(r"(?<=[.!?])\s+", line) if s)
    return pieces


def _pack(paragraphs: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    cur = ""
    for para in paragraphs:
        pieces = [para] if len(para) <= size else _split_long(para, size)
        for piece in pieces:
            joiner = "\n\n" if cur and "\n" not in piece else "\n"
            if cur and len(cur) + len(joiner) + len(piece) > size:
                chunks.append(cur.strip())
                tail = cur[-overlap:] if overlap else ""
                tail = tail[tail.find(" ") + 1:] if " " in tail else ""
                cur = tail
                joiner = "\n" if cur else ""
            cur = f"{cur}{joiner}{piece}" if cur else piece
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def chunk_document(doc: Document, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    chunks: list[Chunk] = []
    n = 0
    for section, page, body in _split_sections(doc):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        for text in _pack(paragraphs, size, overlap):
            n += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}-{n:03d}",
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    source=doc.source,
                    section=section,
                    text=text,
                    page=page,
                )
            )
    return chunks


def build_chunks(docs: list[Document]) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc))
    return out
