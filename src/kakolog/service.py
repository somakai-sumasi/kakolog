"""保存・検索のビジネスロジック。server.pyから分離。"""

import sys

from .chunker import chunk_session
from .db import connection, init_db, insert_memory, memory_exists
from .embedder import embed_documents


def save_session(session_id: str, transcript_path: str, project_path: str | None = None) -> int:
    chunks = chunk_session(transcript_path)
    if not chunks:
        return 0

    texts = [f"{c.question}\n{c.answer}" for c in chunks]
    embeddings = embed_documents(texts)

    with connection() as conn:
        init_db(conn)
        count = 0
        for chunk, emb in zip(chunks, embeddings):
            if memory_exists(conn, chunk.question, chunk.answer):
                continue
            insert_memory(conn, session_id, chunk.question, chunk.answer, emb, project_path)
            count += 1

    return count
