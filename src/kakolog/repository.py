"""メモリのデータ操作（CRUD）"""

from dataclasses import dataclass

import numpy as np
from sqlite_vec import serialize_float32

from .db import get_conn
from .db_util import columns_of, from_row
from .models import Memory


def find_memory_by_content(
    content: str,
    project_path: str | None = None,
) -> Memory | None:
    """content+project_pathをキーにメモリを検索する。"""
    conn = get_conn()
    row = conn.execute(
        f"SELECT {columns_of(Memory)} FROM memories"
        " WHERE content = ? AND project_path IS ? LIMIT 1",
        [content, project_path],
    ).fetchone()
    if row is None:
        return None
    return from_row(row, Memory)


def update_memory(memory: Memory) -> None:
    """メモリをIDをキーに更新する。"""
    get_conn().execute(
        "UPDATE memories"
        " SET user_turn = ?,"
        "     agent_turn = ?,"
        "     content = ?,"
        "     last_accessed_at = ?,"
        "     project_path = ?"
        " WHERE id = ?",
        [
            memory.user_turn,
            memory.agent_turn,
            memory.content,
            memory.last_accessed_at.isoformat(),
            memory.project_path,
            memory.id,
        ],
    )


@dataclass(frozen=True)
class MemoryToSave:
    session_id: str
    user_turn: str
    agent_turn: str
    content: str
    embedding: list[float]
    project_path: str | None = None
    last_accessed_at: str | None = None


def insert_memory(memory: MemoryToSave) -> int:
    conn = get_conn()
    cursor = conn.execute(
        "INSERT INTO memories("
        "  session_id, user_turn, agent_turn,"
        "  content, project_path,"
        "  created_at, last_accessed_at"
        ") VALUES ("
        "  ?, ?, ?, ?, ?,"
        "  COALESCE(?, CURRENT_TIMESTAMP),"
        "  COALESCE(?, CURRENT_TIMESTAMP)"
        ")",
        [
            memory.session_id,
            memory.user_turn,
            memory.agent_turn,
            memory.content,
            memory.project_path,
            memory.last_accessed_at,
            memory.last_accessed_at,
        ],
    )
    memory_id = cursor.lastrowid
    assert memory_id is not None
    conn.execute(
        "INSERT INTO vec_memories(memory_id, embedding)"
        " VALUES (?, ?)",
        [memory_id, serialize_float32(memory.embedding)],
    )
    return memory_id


@dataclass(frozen=True)
class Stats:
    memories: int
    sessions: int


def get_existing_session_ids() -> set[str]:
    rows = get_conn().execute(
        "SELECT DISTINCT session_id FROM memories"
    ).fetchall()
    return {r[0] for r in rows}


def get_stats() -> Stats:
    conn = get_conn()
    memories = conn.execute(
        "SELECT COUNT(*) FROM memories"
    ).fetchone()[0]
    sessions = conn.execute(
        "SELECT COUNT(DISTINCT session_id) FROM memories"
    ).fetchone()[0]
    return Stats(memories=memories, sessions=sessions)


def fetch_memories_by_ids(
    ids: list[int],
    project_path: str | None = None,
) -> list[Memory]:
    """指定IDのメモリをDBから取得。project_path指定時はフィルタリング。"""
    if not ids:
        return []
    conn = get_conn()
    placeholders = ",".join("?" * len(ids))
    query_sql = (
        f"SELECT {columns_of(Memory)}"
        f" FROM memories WHERE id IN ({placeholders})"
    )
    params: list = list(ids)
    if project_path:
        query_sql += " AND project_path = ?"
        params.append(project_path)
    rows = conn.execute(query_sql, params).fetchall()
    return [from_row(row, Memory) for row in rows]


def search_fts(
    terms: list[str],
    limit: int = 50,
) -> tuple[list[int], dict[int, int]]:
    """FTS5でタームごとに検索し、ヒット数でソートしたIDを返す。"""
    conn = get_conn()
    doc_hits: dict[int, int] = {}
    for term in terms:
        try:
            rows = conn.execute(
                "SELECT rowid"
                " FROM fts_memories"
                " WHERE fts_memories MATCH ?"
                " LIMIT ?",
                [term, limit],
            ).fetchall()
            for r in rows:
                doc_hits[r[0]] = doc_hits.get(r[0], 0) + 1
        except Exception:
            continue

    sorted_ids = sorted(
        doc_hits.keys(), key=lambda x: doc_hits[x], reverse=True
    )
    return sorted_ids[:limit], doc_hits


def search_vec(
    embedding: list[float],
    limit: int = 50,
) -> list[int]:
    """sqlite-vecでベクトル近傍検索を行いIDリストを返す。"""
    rows = get_conn().execute(
        "SELECT memory_id"
        " FROM vec_memories"
        " WHERE embedding MATCH ? AND k = ?",
        [serialize_float32(embedding), limit],
    ).fetchall()
    return [r[0] for r in rows]


def fetch_embeddings_by_ids(
    memory_ids: list[int],
) -> dict[int, np.ndarray]:
    """vec_memoriesから指定IDのベクトルを取得"""
    if not memory_ids:
        return {}
    placeholders = ",".join("?" * len(memory_ids))
    rows = get_conn().execute(
        "SELECT memory_id, embedding"
        " FROM vec_memories"
        f" WHERE memory_id IN ({placeholders})",
        memory_ids,
    ).fetchall()
    return {row[0]: np.frombuffer(row[1], dtype=np.float32) for row in rows}
