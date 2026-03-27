"""保存・検索のビジネスロジック。server.pyから分離。"""

import sys

from .chunker import chunk_session
from .config import is_excluded
from .db import connection, init_db
from .repository import insert_memory, touch_if_exists
from .embedder import embed_documents


def save_session(session_id: str, transcript_path: str, project_path: str | None = None) -> int:
    if is_excluded(project_path):
        return 0
    chunks = chunk_session(transcript_path)
    if not chunks:
        return 0

    texts = [f"{c.question}\n{c.answer}" for c in chunks]
    embeddings = embed_documents(texts)

    with connection() as conn:
        init_db(conn)
        count = 0
        for chunk, emb in zip(chunks, embeddings):
            if touch_if_exists(conn, chunk.question, chunk.answer, project_path):
                continue
            insert_memory(conn, session_id, chunk.question, chunk.answer, emb, project_path)
            count += 1

    return count
