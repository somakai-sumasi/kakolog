"""メモリのデータ操作（CRUD）"""

import sqlite3
from dataclasses import dataclass

import numpy as np
from sqlite_vec import serialize_float32

from .db import Memory


def find_memory_by_turns(
    conn: sqlite3.Connection,
    user_turn: str,
    agent_turn: str,
    project_path: str | None = None,
) -> Memory | None:
    """ユーザー/エージェントターン+project_pathをキーにメモリを検索する。"""
    row = conn.execute(
        "SELECT id, user_turn, agent_turn, last_accessed_at, project_path FROM memories"
        " WHERE user_turn = ? AND agent_turn = ? AND project_path IS ? LIMIT 1",
        [user_turn, agent_turn, project_path],
    ).fetchone()
    if row is None:
        return None
    return Memory(
        id=row["id"],
        user_turn=row["user_turn"],
        agent_turn=row["agent_turn"],
        last_accessed_at=row["last_accessed_at"],
        project_path=row["project_path"],
    )


def update_memory(conn: sqlite3.Connection, memory: Memory) -> None:
    """メモリをIDをキーに更新する。"""
    conn.execute(
        "UPDATE memories SET user_turn = ?, agent_turn = ?, last_accessed_at = ?, project_path = ? WHERE id = ?",
        [
            memory.user_turn,
            memory.agent_turn,
            memory.last_accessed_at,
            memory.project_path,
            memory.id,
        ],
    )


@dataclass(frozen=True)
class MemoryToSave:
    session_id: str
    user_turn: str
    agent_turn: str
    embedding: list[float]
    project_path: str | None = None
    last_accessed_at: str | None = None


def insert_memory(conn: sqlite3.Connection, memory: MemoryToSave) -> int:
    cursor = conn.execute(
        "INSERT INTO memories(session_id, user_turn, agent_turn, project_path, last_accessed_at)"
        " VALUES (?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))",
        [
            memory.session_id,
            memory.user_turn,
            memory.agent_turn,
            memory.project_path,
            memory.last_accessed_at,
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
    query_sql = (
        f"SELECT id, user_turn, agent_turn, last_accessed_at, project_path"
        f" FROM memories WHERE id IN ({placeholders})"
    )
    params: list = list(ids)
    if project_path:
        query_sql += " AND project_path = ?"
        params.append(project_path)
    rows = conn.execute(query_sql, params).fetchall()
    return [
        Memory(
            id=row["id"],
            user_turn=row["user_turn"],
            agent_turn=row["agent_turn"],
            last_accessed_at=row["last_accessed_at"],
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
