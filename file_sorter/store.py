from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .scanner import FileInfo

logger = logging.getLogger(__name__)

_COLLECTION = 'files'


def _file_id(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]


@dataclass
class StoredFile:
    id: str
    path: str
    filename: str
    file_type: str
    size: int
    mtime: float
    extension: str
    embedding: Optional[np.ndarray] = None

    @classmethod
    def from_chroma(cls, id_: str, meta: dict, embedding: Optional[list] = None) -> 'StoredFile':
        return cls(
            id=id_,
            path=meta['path'],
            filename=meta['filename'],
            file_type=meta['file_type'],
            size=int(meta['size']),
            mtime=float(meta['mtime']),
            extension=meta['extension'],
            embedding=np.array(embedding) if embedding is not None else None,
        )


@dataclass
class SimilarFile:
    stored: StoredFile
    score: float


class VectorStore:
    def __init__(self, db_path: str = '.file_sorter_db'):
        import chromadb
        self._client = chromadb.PersistentClient(path=db_path)
        self._col = self._client.get_or_create_collection(
            name=_COLLECTION,
            metadata={'hnsw:space': 'cosine'},
        )

    def is_processed(self, path: Path, mtime: float) -> bool:
        result = self._col.get(ids=[_file_id(path)], include=['metadatas'])
        if not result['ids']:
            return False
        stored_mtime = result['metadatas'][0].get('mtime', 0)
        return abs(float(stored_mtime) - mtime) < 1.0

    def add_file(self, info: FileInfo, embedding: np.ndarray) -> None:
        self._col.upsert(
            ids=[_file_id(info.path)],
            embeddings=[embedding.tolist()],
            documents=[info.name],
            metadatas=[{
                'path':      str(info.path.resolve()),
                'filename':  info.name,
                'file_type': info.file_type,
                'size':      info.size,
                'mtime':     info.mtime,
                'extension': info.extension,
            }],
        )

    def get_similar(
        self,
        embedding: np.ndarray,
        n: int = 10,
        exclude_path: Optional[Path] = None,
    ) -> list[SimilarFile]:
        total = self._col.count()
        if total == 0:
            return []

        fetch = min(n + (1 if exclude_path else 0), total)
        result = self._col.query(
            query_embeddings=[embedding.tolist()],
            n_results=fetch,
            include=['metadatas', 'distances'],
        )

        out: list[SimilarFile] = []
        exclude_str = str(exclude_path.resolve()) if exclude_path else None
        for id_, meta, dist in zip(
            result['ids'][0], result['metadatas'][0], result['distances'][0]
        ):
            if exclude_str and meta['path'] == exclude_str:
                continue
            out.append(SimilarFile(
                stored=StoredFile.from_chroma(id_, meta),
                score=1.0 - float(dist),
            ))
            if len(out) == n:
                break

        return out

    def get_all(self, include_embeddings: bool = False) -> list[StoredFile]:
        include = ['metadatas'] + (['embeddings'] if include_embeddings else [])
        result = self._col.get(include=include)
        files = []
        embeddings = result.get('embeddings') or []
        for i, (id_, meta) in enumerate(zip(result['ids'], result['metadatas'])):
            emb = embeddings[i] if embeddings else None
            files.append(StoredFile.from_chroma(id_, meta, emb))
        return files

    def count(self) -> int:
        return self._col.count()

    def remove_missing(self) -> int:
        """Delete DB entries whose files no longer exist on disk. Returns count removed."""
        all_files = self.get_all()
        to_remove = [f.id for f in all_files if not Path(f.path).exists()]
        if to_remove:
            self._col.delete(ids=to_remove)
        return len(to_remove)
