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
from dataclasses import asdict, dataclass, field
from datetime import datetime
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
    policy_id: str | None = None
    version: str | None = None
    effective_date: str | None = None
    supersedes: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    source: str
    section: str
    text: str
    page: int | None = None
    policy_id: str | None = None
    version: str | None = None
    effective_date: str | None = None

    @property
    def embed_text(self) -> str:
        """Text that is embedded: title + section give the chunk extra context."""
        return f"{self.doc_title} - {self.section}\n{self.text}"

    @property
    def label(self) -> str:
        loc = f" (page {self.page})" if self.page else ""
        revision = ""
        if self.version or self.effective_date:
            details = [f"version {self.version}" if self.version else "",
                       f"effective {self.effective_date}" if self.effective_date else ""]
            revision = f" ({', '.join(part for part in details if part)})"
        return f"{self.doc_title} > {self.section}{loc}{revision}"

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _prettify(stem: str) -> str:
    stem = re.sub(r"^\d+[_\-\s]*", "", stem)
    return stem.replace("_", " ").replace("-", " ").strip().title() or "Untitled"


def _make_id(source: Path) -> str:
    identity = str(source.resolve()).casefold()
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]


_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|$)", re.S)
_POLICY_HEADER_RE = re.compile(
    r"^Document ID:\s*([^|]+)\|\s*Version\s*([^|]+)\|\s*Effective from\s*([^|]+)",
    re.I | re.M,
)


def _normalise_date(value: str) -> str:
    value = value.strip()
    for pattern in ("%d %B %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(value, pattern).date().isoformat()
        except ValueError:
            continue
    return value


def _parse_metadata(text: str) -> tuple[dict[str, str | list[str]], str]:
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text

    metadata: dict[str, str | list[str]] = {}
    for line in match.group(1).splitlines():
        key, separator, value = line.partition(":")
        key, value = key.strip().lower(), value.strip().strip("\"'")
        if not separator or key not in {"policy_id", "version", "effective_date", "supersedes"}:
            continue
        if key == "supersedes":
            metadata[key] = [item.strip() for item in value.split(",") if item.strip()]
        elif value:
            metadata[key] = value
    return metadata, text[match.end():]


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
        if not any(page.strip() for page in pages):
            return None
        # Keep page markers so chunks can cite page numbers.
        text = "\n".join(f"<<PAGE {i}>>\n{p}" for i, p in enumerate(pages, start=1))
        kind = "pdf"
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
        kind = "md" if suffix == ".md" else "txt"
    metadata: dict[str, str | list[str]] = {}
    if kind == "md":
        metadata, text = _parse_metadata(text)
        header = _POLICY_HEADER_RE.search(text)
        if header:
            metadata.setdefault("policy_id", header.group(1).strip())
            metadata.setdefault("version", header.group(2).strip())
            metadata.setdefault("effective_date", _normalise_date(header.group(3)))
    if not text.strip():
        return None

    title = _prettify(path.stem)
    if kind == "md":  # prefer the first "# Heading" as the title
        for line in text.splitlines():
            m = HEADING_RE.match(line)
            if m and len(m.group(1)) == 1:
                title = m.group(2)
                break
    effective_date = metadata.get("effective_date")
    if isinstance(effective_date, str):
        effective_date = _normalise_date(effective_date)
    return Document(
        doc_id=_make_id(path), title=title, source=path.name, text=text, kind=kind,
        policy_id=metadata.get("policy_id") if isinstance(metadata.get("policy_id"), str) else None,
        version=metadata.get("version") if isinstance(metadata.get("version"), str) else None,
        effective_date=effective_date if isinstance(effective_date, str) else None,
        supersedes=metadata.get("supersedes") if isinstance(metadata.get("supersedes"), list) else [],
    )


def load_documents(folders: Iterable[Path], errors: list[str] | None = None) -> list[Document]:
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
                    message = f"{path.name}: {exc}"
                    print(f"[ingest] skipped {message}")
                    if errors is not None:
                        errors.append(message)
                    continue
                if doc:
                    docs.append(doc)
                elif errors is not None:
                    errors.append(f"{path.name}: no extractable text (scanned PDFs require OCR).")
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


def _split_fallback(text: str, size: int) -> list[str]:
    """Split text that has no useful sentence boundary, including long tokens."""
    pieces: list[str] = []
    current = ""
    for word in re.findall(r"\S+", text):
        if len(word) > size:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(word[i:i + size] for i in range(0, len(word), size))
            continue
        candidate = f"{current} {word}" if current else word
        if len(candidate) > size:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _split_long(paragraph: str, size: int) -> list[str]:
    """Break long prose and tables while keeping table headers with each fragment."""
    lines = paragraph.split("\n")
    pieces: list[str] = []
    index = 0

    def split_prose(line: str) -> list[str]:
        result: list[str] = []
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            if not sentence:
                continue
            result.extend([sentence] if len(sentence) <= size else _split_fallback(sentence, size))
        return result

    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            pieces.extend(split_prose(lines[index]))
            index += 1
            continue

        end = index
        while end < len(lines) and lines[end].lstrip().startswith("|"):
            end += 1
        table = lines[index:end]
        header_lines = table[:2] if len(table) > 1 and re.search(r"-{3,}", table[1]) else table[:1]
        rows = table[len(header_lines):]
        header = "\n".join(header_lines)
        current = header
        for row in rows:
            candidate = f"{current}\n{row}"
            if len(candidate) <= size:
                current = candidate
                continue
            if current:
                pieces.append(current)
            row_room = size - len(header) - 1
            if row_room > 0 and len(row) <= row_room:
                current = f"{header}\n{row}"
            elif row_room > 0:
                row_pieces = _split_fallback(row, row_room)
                pieces.extend(f"{header}\n{part}" for part in row_pieces[:-1])
                current = f"{header}\n{row_pieces[-1]}"
            else:
                pieces.extend(_split_fallback(header, size))
                pieces.extend(_split_fallback(row, size))
                current = ""
        if current:
            pieces.append(current)
        index = end
    return pieces


def _pack(paragraphs: list[str], size: int, overlap: int) -> list[str]:
    if size < 1:
        raise ValueError("chunk size must be at least 1")
    chunks: list[str] = []
    cur = ""
    for para in paragraphs:
        pieces = [para] if len(para) <= size else _split_long(para, size)
        pieces = [part for piece in pieces for part in
                  ([piece] if len(piece) <= size else _split_fallback(piece, size))]
        for piece in pieces:
            joiner = "\n\n" if cur and "\n" not in piece else "\n"
            if cur and len(cur) + len(joiner) + len(piece) > size:
                chunks.append(cur.strip())
                available = max(0, size - len(piece) - 1)
                carry = min(max(overlap, 0), available)
                tail = cur[-carry:] if carry and not piece.lstrip().startswith("|") else ""
                tail = tail[tail.find(" ") + 1:] if " " in tail else ""
                cur = tail if len(tail) + len(piece) + 1 <= size else ""
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
                    policy_id=doc.policy_id,
                    version=doc.version,
                    effective_date=doc.effective_date,
                )
            )
    return chunks


def build_chunks(docs: list[Document]) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc))
    return out
