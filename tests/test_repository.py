import numpy as np

from kakolog.db import EMBEDDING_DIM, Memory
from kakolog.repository import (
    MemoryToSave,
    fetch_embeddings_by_ids,
    fetch_memories_by_ids,
    find_memory_by_qa,
    get_stats,
    insert_memory,
    update_memory,
)


def _make_embedding():
    vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    return vec.tolist()


def _make_memory(
    session_id="sess1", question="Q", answer="A", **kwargs
) -> MemoryToSave:
    return MemoryToSave(
        session_id=session_id,
        question=question,
        answer=answer,
        embedding=_make_embedding(),
        **kwargs,
    )


class TestInsert:
    def test_insert_returns_id(self, db_conn):
        mid = insert_memory(db_conn, _make_memory())
        assert mid > 0


class TestFindByQA:
    def test_find_existing(self, db_conn):
        insert_memory(db_conn, _make_memory())
        result = find_memory_by_qa(db_conn, "Q", "A")
        assert result is not None
        assert result.question == "Q"
        assert result.answer == "A"

    def test_find_nonexistent(self, db_conn):
        assert find_memory_by_qa(db_conn, "Q", "A") is None

    def test_find_with_project_path(self, db_conn):
        insert_memory(db_conn, _make_memory(project_path="/proj"))
        assert find_memory_by_qa(db_conn, "Q", "A", project_path="/proj") is not None
        assert find_memory_by_qa(db_conn, "Q", "A", project_path="/other") is None


class TestUpdateMemory:
    def test_update_created_at(self, db_conn):
        insert_memory(db_conn, _make_memory())
        existing = find_memory_by_qa(db_conn, "Q", "A")
        updated = Memory(
            id=existing.id,
            question=existing.question,
            answer=existing.answer,
            created_at="2026-01-01T00:00:00",
            project_path=existing.project_path,
        )
        update_memory(db_conn, updated)
        result = find_memory_by_qa(db_conn, "Q", "A")
        assert result.created_at == "2026-01-01T00:00:00"


class TestFetch:
    def test_fetch_memories_by_ids(self, db_conn):
        id1 = insert_memory(db_conn, _make_memory("s1", "Q1", "A1"))
        id2 = insert_memory(db_conn, _make_memory("s1", "Q2", "A2"))
        memories = fetch_memories_by_ids(db_conn, [id1, id2])
        assert len(memories) == 2
        assert all(isinstance(m, Memory) for m in memories)

    def test_fetch_empty_ids(self, db_conn):
        assert fetch_memories_by_ids(db_conn, []) == []

    def test_fetch_with_project_filter(self, db_conn):
        id1 = insert_memory(
            db_conn, _make_memory("s1", "Q1", "A1", project_path="/proj")
        )
        id2 = insert_memory(
            db_conn, _make_memory("s1", "Q2", "A2", project_path="/other")
        )
        memories = fetch_memories_by_ids(db_conn, [id1, id2], project_path="/proj")
        assert len(memories) == 1
        assert memories[0].project_path == "/proj"

    def test_fetch_embeddings(self, db_conn):
        mid = insert_memory(db_conn, _make_memory())
        embs = fetch_embeddings_by_ids(db_conn, [mid])
        assert mid in embs
        assert len(embs[mid]) == EMBEDDING_DIM

    def test_fetch_embeddings_empty(self, db_conn):
        assert fetch_embeddings_by_ids(db_conn, []) == {}


class TestStats:
    def test_empty_db(self, db_conn):
        s = get_stats(db_conn)
        assert s.memories == 0
        assert s.sessions == 0

    def test_after_insert(self, db_conn):
        insert_memory(db_conn, _make_memory("s1", "Q1", "A1"))
        insert_memory(db_conn, _make_memory("s2", "Q2", "A2"))
        s = get_stats(db_conn)
        assert s.memories == 2
        assert s.sessions == 2
