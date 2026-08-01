from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_SUFFIXES = {".md", ".txt", ".html", ".htm"}
TAG_RE = re.compile(r"<[^>]+>")
HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")
ASCII_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+.-]{1,}")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]+")


@dataclass(frozen=True)
class KnowledgeChunk:
    source: str
    title: str
    text: str
    terms: frozenset[str]


@dataclass(frozen=True)
class Citation:
    source: str
    title: str
    excerpt: str
    score: float


def _terms(text: str) -> frozenset[str]:
    lowered = text.lower()
    tokens = set(ASCII_TOKEN_RE.findall(lowered))
    for run in CHINESE_RE.findall(lowered):
        tokens.update(run[index:index + 2] for index in range(max(1, len(run) - 1)))
        if len(run) <= 4:
            tokens.add(run)
    return frozenset(tokens)


def _plain_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in {".html", ".htm"}:
        text = TAG_RE.sub(" ", text)
        text = html.unescape(text)
    return text.replace("\r\n", "\n")


def _split_document(source: str, fallback_title: str, text: str, max_chars: int) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    title = fallback_title
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            chunks.append(KnowledgeChunk(source, title, body, _terms(f"{title}\n{body}")))
        buffer.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        heading = HEADING_RE.match(line)
        if heading:
            flush()
            title = heading.group(1)
            continue
        if not line:
            continue
        if sum(len(part) for part in buffer) + len(line) > max_chars:
            flush()
        buffer.append(line)
    flush()
    return chunks


class KnowledgeIndex:
    def __init__(self, roots: list[Path], max_chunk_chars: int = 1800) -> None:
        self.roots = [root.resolve() for root in roots]
        self.chunks: list[KnowledgeChunk] = []
        for root in self.roots:
            if not root.is_dir():
                raise FileNotFoundError(f"knowledge root does not exist: {root}")
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                    continue
                source = self._display_path(path)
                self.chunks.extend(_split_document(source, path.stem, _plain_text(path), max_chunk_chars))
        if not self.chunks:
            raise ValueError("knowledge roots contain no supported documents")

    def _display_path(self, path: Path) -> str:
        for root in self.roots:
            try:
                return f"{root.name}/{path.relative_to(root).as_posix()}"
            except ValueError:
                continue
        return path.name

    @property
    def document_count(self) -> int:
        return len({chunk.source for chunk in self.chunks})

    def search(self, query: str, limit: int = 4) -> list[Citation]:
        query_terms = _terms(query)
        if not query_terms:
            return []
        ranked: list[tuple[float, KnowledgeChunk]] = []
        for chunk in self.chunks:
            overlap = query_terms & chunk.terms
            if not overlap:
                continue
            title_terms = _terms(chunk.title)
            score = sum(2.0 if term in title_terms else 1.0 for term in overlap)
            score /= max(1.0, len(query_terms) ** 0.5)
            ranked.append((score, chunk))
        ranked.sort(key=lambda item: (-item[0], item[1].source, item[1].title))
        return [
            Citation(chunk.source, chunk.title, chunk.text[:360].replace("\n", " "), score)
            for score, chunk in ranked[:limit]
        ]

