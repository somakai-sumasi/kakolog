"""メモリのデータ操作（CRUD）"""

import sqlite3
from dataclasses import dataclass

import numpy as np
from sqlite_vec import serialize_float32

from .db import Memory


def find_memory_by_qa(
    conn: sqlite3.Connection,
    question: str,
    answer: str,
    project_path: str | None = None,
) -> Memory | None:
    """Q&A+project_pathをキーにメモリを検索する。"""
    row = conn.execute(
        "SELECT id, question, answer, created_at, project_path FROM memories"
        " WHERE question = ? AND answer = ? AND project_path IS ? LIMIT 1",
        [question, answer, project_path],
    ).fetchone()
    if row is None:
        return None
    return Memory(
        id=row["id"],
        question=row["question"],
        answer=row["answer"],
        created_at=row["created_at"],
        project_path=row["project_path"],
    )


def update_memory(conn: sqlite3.Connection, memory: Memory) -> None:
    """メモリをIDをキーに更新する。"""
    conn.execute(
        "UPDATE memories SET question = ?, answer = ?, created_at = ?, project_path = ? WHERE id = ?",
        [
            memory.question,
            memory.answer,
            memory.created_at,
            memory.project_path,
            memory.id,
        ],
    )


@dataclass(frozen=True)
class MemoryToSave:
    session_id: str
    question: str
    answer: str
    embedding: list[float]
    project_path: str | None = None
    created_at: str | None = None


def insert_memory(conn: sqlite3.Connection, memory: MemoryToSave) -> int:
    cursor = conn.execute(
        "INSERT INTO memories(session_id, question, answer, project_path, created_at)"
        " VALUES (?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))",
        [
            memory.session_id,
            memory.question,
            memory.answer,
            memory.project_path,
            memory.created_at,
        ],
    )
    memory_id = cursor.lastrowid
    assert memory_id is not None
    conn.execute(
        "INSERT INTO vec_memories(memory_id, embedding) VALUES (?, ?)",
        [memory_id, serialize_float32(memory.embedding)],
    )
    return memory_id


@dataclass(frozen=True)
class Stats:
    memories: int
    sessions: int


def get_existing_session_ids(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT DISTINCT session_id FROM memories").fetchall()
    return {r[0] for r in rows}


def get_stats(conn: sqlite3.Connection) -> Stats:
    memories = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    sessions = conn.execute(
        "SELECT COUNT(DISTINCT session_id) FROM memories"
    ).fetchone()[0]
    return Stats(memories=memories, sessions=sessions)


def fetch_memories_by_ids(
    conn: sqlite3.Connection,
    ids: list[int],
    project_path: str | None = None,
) -> list[Memory]:
    """指定IDのメモリをDBから取得。project_path指定時はフィルタリング。"""
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    query_sql = f"SELECT id, question, answer, created_at, project_path FROM memories WHERE id IN ({placeholders})"
    params: list = list(ids)
    if project_path:
        query_sql += " AND project_path = ?"
        params.append(project_path)
    rows = conn.execute(query_sql, params).fetchall()
    return [
        Memory(
            id=row["id"],
            question=row["question"],
            answer=row["answer"],
            created_at=row["created_at"],
            project_path=row["project_path"],
        )
        for row in rows
    ]


def fetch_embeddings_by_ids(
    conn: sqlite3.Connection, memory_ids: list[int]
) -> dict[int, np.ndarray]:
    """vec_memoriesから指定IDのベクトルを取得"""
    if not memory_ids:
        return {}
    placeholders = ",".join("?" * len(memory_ids))
    rows = conn.execute(
        f"SELECT memory_id, embedding FROM vec_memories WHERE memory_id IN ({placeholders})",
        memory_ids,
    ).fetchall()
    return {row[0]: np.frombuffer(row[1], dtype=np.float32) for row in rows}
