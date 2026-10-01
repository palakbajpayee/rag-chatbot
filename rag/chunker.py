"""Split knowledge-base markdown files into small, overlapping, heading-aware chunks."""
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str   # file name, e.g. "returns_refunds.md"
    section: str  # nearest heading, e.g. "Refund timelines"
    text: str


_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    """Return (heading, body) pairs; text before the first heading gets an empty heading."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in markdown.splitlines():
        match = _HEADING.match(line.strip())
        if match:
            sections.append((match.group(2).strip(), []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(body).strip()) for h, body in sections if "\n".join(body).strip()]


def _window(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if len(words) <= size:
        return [words]
    step = max(1, size - overlap)
    return [words[i:i + size] for i in range(0, len(words) - overlap, step)]


def load_chunks(kb_dir: Path, chunk_words: int = 120, overlap: int = 30) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")) + sorted(kb_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        doc_title = path.stem.replace("_", " ").title()
        for s_idx, (heading, body) in enumerate(_split_sections(text)):
            for w_idx, words in enumerate(_window(body.split(), chunk_words, overlap)):
                chunks.append(Chunk(
                    id=f"{path.stem}#{s_idx}.{w_idx}",
                    source=path.name,
                    section=heading or doc_title,
                    text=" ".join(words),
                ))
    return chunks
