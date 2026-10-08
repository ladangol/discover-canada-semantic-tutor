"""The indexed Discover Canada chunks, loaded straight from ChromaDB.

Nothing is re-embedded: the stored MiniLM vectors are the single source of
truth for neighbours, diversity selection, and the UMAP atlas.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import chromadb
import numpy as np
import pandas as pd

from src.config import CHROMA_PATH, COLLECTION_NAME

PREVIEW_CHARS = 160


def split_breadcrumb(text: str) -> tuple[list[str], str]:
    """Split a chunk into (breadcrumb lines, body).

    Chunks usually start with "Chapter\\nSection\\n\\nbody...". A chunk with a
    single heading line (the section title only) has no chapter breadcrumb.
    """
    head, sep, body = text.partition("\n\n")
    if not sep:
        return [], text.strip()
    return [line.strip() for line in head.split("\n") if line.strip()], body.strip()


def derive_chapter(text: str, section: str) -> str:
    """Top-level chapter = first breadcrumb line when there are two heading
    lines; otherwise the chunk's own section title (e.g. "Understanding the
    Oath", where the source has no chapter above the section)."""
    crumbs, _ = split_breadcrumb(text)
    return crumbs[0] if len(crumbs) >= 2 else section


def _preview(body: str) -> str:
    flat = " ".join(body.split())
    return flat if len(flat) <= PREVIEW_CHARS else flat[: PREVIEW_CHARS - 1] + "…"


@dataclass(frozen=True)
class Corpus:
    chunks: pd.DataFrame        # one row per chunk, same order as `embeddings`
    embeddings: np.ndarray      # (n, 384) float32, L2-normalized

    def __len__(self) -> int:
        return len(self.chunks)

    def index_of(self, chunk_id: str) -> int:
        return int(self._index[chunk_id])

    @property
    def _index(self) -> dict[str, int]:
        return dict(zip(self.chunks["chunk_id"], range(len(self.chunks))))

    def row(self, chunk_id: str) -> pd.Series:
        return self.chunks.iloc[self.index_of(chunk_id)]

    @property
    def chapters(self) -> list[str]:
        return list(dict.fromkeys(self.chunks["chapter"]))

    def sections(self, chapter: str | None = None) -> list[str]:
        df = self.chunks if chapter is None else self.chunks[self.chunks["chapter"] == chapter]
        return list(dict.fromkeys(df["section"]))

    def scope_indices(self, chapter: str | None = None, section: str | None = None) -> list[int]:
        mask = np.ones(len(self.chunks), dtype=bool)
        if chapter:
            mask &= (self.chunks["chapter"] == chapter).to_numpy()
        if section:
            mask &= (self.chunks["section"] == section).to_numpy()
        return [int(i) for i in np.flatnonzero(mask)]

    def neighbours(
        self,
        chunk_id: str,
        n: int = 5,
        *,
        exclude_same_section: bool = False,
    ) -> list[tuple[int, float]]:
        """Nearest chunks by cosine similarity of the ORIGINAL embeddings.

        Vectors are normalized, so cosine == dot product. `exclude_same_section`
        drops overlapping-window siblings so neighbours are *other* concepts.
        """
        i = self.index_of(chunk_id)
        sims = self.embeddings @ self.embeddings[i]
        sections = self.chunks["section"].to_numpy()
        out: list[tuple[int, float]] = []
        for j in np.argsort(-sims):
            if j == i or (exclude_same_section and sections[j] == sections[i]):
                continue
            out.append((int(j), float(sims[j])))
            if len(out) == n:
                break
        return out


@lru_cache(maxsize=1)
def load_corpus() -> Corpus:
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
    result = collection.get(include=["documents", "metadatas", "embeddings"])

    rows = []
    for chunk_id, text, meta in zip(result["ids"], result["documents"], result["metadatas"]):
        _, body = split_breadcrumb(text)
        rows.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "body": body,
                "page": str(meta["page"]),
                "page_start": int(meta.get("page_start", meta["page"])),
                "section": str(meta["section"]),
                "block_type": str(meta["block_type"]),
                "chapter": derive_chapter(text, str(meta["section"])),
                "preview": _preview(body),
                "word_count": len(body.split()),
            }
        )
    chunks = pd.DataFrame(rows)
    embeddings = np.asarray(result["embeddings"], dtype=np.float32)
    return Corpus(chunks=chunks, embeddings=embeddings)
