from datetime import datetime

import numpy as np

from kakolog.db import EMBEDDING_DIM
from kakolog.models import Memory, SearchScope
from kakolog.repository import (
    MemoryToSave,
    fetch_embeddings_by_ids,
    fetch_memories_by_ids,
    find_memory_by_content,
    get_stats,
    insert_memory,
    update_memory,
)


def _make_embedding():
    vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    return vec.tolist()


def _make_memory(
    session_id="sess1", user_turn="U", agent_turn="A", **kwargs
) -> MemoryToSave:
    content = kwargs.pop("content", f"U: {user_turn}\nA: {agent_turn}")
    return MemoryToSave(
        session_id=session_id,
        user_turn=user_turn,
        agent_turn=agent_turn,
        content=content,
        embedding=_make_embedding(),
        **kwargs,
    )


class TestInsert:
    def test_insert_returns_id(self, db_conn):
        mid = insert_memory(_make_memory())
        assert mid > 0


class TestFindByContent:
    def test_find_existing(self, db_conn):
        insert_memory(_make_memory())
        result = find_memory_by_content("U: U\nA: A")
        assert result is not None
        assert result.user_turn == "U"
        assert result.agent_turn == "A"
        assert result.content == "U: U\nA: A"

    def test_find_nonexistent(self, db_conn):
        assert find_memory_by_content("U: U\nA: A") is None

    def test_find_with_project_path(self, db_conn):
        insert_memory(_make_memory(project_path="/proj"))
        assert find_memory_by_content("U: U\nA: A", project_path="/proj") is not None
        assert find_memory_by_content("U: U\nA: A", project_path="/other") is None


class TestUpdateMemory:
    def test_update_last_accessed_at(self, db_conn):
        insert_memory(_make_memory())
        existing = find_memory_by_content("U: U\nA: A")
        updated = Memory(
            id=existing.id,
            session_id=existing.session_id,
            user_turn=existing.user_turn,
            agent_turn=existing.agent_turn,
            content=existing.content,
            created_at=existing.created_at,
            last_accessed_at=datetime(2026, 1, 1),
            access_count=existing.access_count + 1,
            project_path=existing.project_path,
        )
        update_memory(updated)
        result = find_memory_by_content("U: U\nA: A")
        assert result.last_accessed_at == datetime(2026, 1, 1)


class TestFetch:
    def test_fetch_memories_by_ids(self, db_conn):
        id1 = insert_memory(_make_memory("s1", "U1", "A1"))
        id2 = insert_memory(_make_memory("s1", "U2", "A2"))
        memories = fetch_memories_by_ids([id1, id2])
        assert len(memories) == 2
        assert all(isinstance(m, Memory) for m in memories)

    def test_fetch_empty_ids(self, db_conn):
        assert fetch_memories_by_ids([]) == []

    def test_fetch_with_project_filter(self, db_conn):
        id1 = insert_memory(_make_memory("s1", "U1", "A1", project_path="/proj"))
        id2 = insert_memory(_make_memory("s1", "U2", "A2", project_path="/other"))
        memories = fetch_memories_by_ids([id1, id2], SearchScope(project_path="/proj"))
        assert len(memories) == 1
        assert memories[0].project_path == "/proj"

    def test_fetch_with_session_filter(self, db_conn):
        id1 = insert_memory(_make_memory("s1", "U1", "A1"))
        id2 = insert_memory(_make_memory("s2", "U2", "A2"))
        memories = fetch_memories_by_ids([id1, id2], SearchScope(session_id="s1"))
        assert len(memories) == 1
        assert memories[0].session_id == "s1"

    def test_fetch_embeddings(self, db_conn):
        mid = insert_memory(_make_memory())
        embs = fetch_embeddings_by_ids([mid])
        assert mid in embs
        assert len(embs[mid]) == EMBEDDING_DIM

    def test_fetch_embeddings_empty(self, db_conn):
        assert fetch_embeddings_by_ids([]) == {}


class TestStats:
    def test_empty_db(self, db_conn):
        s = get_stats()
        assert s.memories == 0
        assert s.sessions == 0

    def test_after_insert(self, db_conn):
        insert_memory(_make_memory("s1", "U1", "A1"))
        insert_memory(_make_memory("s2", "U2", "A2"))
        s = get_stats()
        assert s.memories == 2
        assert s.sessions == 2
