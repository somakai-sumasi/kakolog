"""保存・検索のビジネスロジック。server.pyから分離。"""

from pathlib import Path

from .chunker import chunk_session
from .config import is_excluded
from .db import Memory, connection
from .embedder import embed_documents
from .repository import MemoryToSave, find_memory_by_qa, insert_memory, update_memory
from .transcript import SessionMeta, read_session_meta

_EXCLUDED_ENTRYPOINTS = frozenset({"sdk-cli"})
_EXCLUDED_PATH_PARTS = frozenset({"subagents"})


def is_excluded_session(transcript_path: str, entrypoint: str | None) -> bool:
    if any(part in _EXCLUDED_PATH_PARTS for part in Path(transcript_path).parts):
        return True
    if entrypoint in _EXCLUDED_ENTRYPOINTS:
        return True
    return False


def save_session(
    session_id: str, transcript_path: str, project_path: str | None = None
) -> int:
    meta = read_session_meta(transcript_path)
    if is_excluded_session(transcript_path, meta.entrypoint):
        return 0
    resolved_project_path = project_path if project_path is not None else meta.cwd
    if is_excluded(resolved_project_path):
        return 0
    chunks = chunk_session(transcript_path)
    if not chunks:
        return 0

    texts = [c.to_text() for c in chunks]
    embeddings = embed_documents(texts)

    with connection() as conn:
        count = 0
        for chunk, emb in zip(chunks, embeddings):
            ts = chunk.timestamp or meta.first_timestamp
            existing = find_memory_by_qa(conn, chunk.question, chunk.answer, resolved_project_path)
            if existing:
                update_memory(conn, Memory(
                    id=existing.id,
                    question=existing.question,
                    answer=existing.answer,
                    created_at=ts if ts is not None else existing.created_at,
                    project_path=existing.project_path,
                ))
                continue
            insert_memory(conn, MemoryToSave(
                session_id=session_id,
                question=chunk.question,
                answer=chunk.answer,
                embedding=emb,
                project_path=resolved_project_path,
                created_at=ts,
            ))
            count += 1

    return count
