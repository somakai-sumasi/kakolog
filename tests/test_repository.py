import numpy as np

from kakolog.db import EMBEDDING_DIM, Memory
from kakolog.repository import (
    fetch_embeddings_by_ids,
    fetch_memories_by_ids,
    get_stats,
    insert_memory,
    touch_if_exists,
)


def _make_embedding():
    vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    return vec.tolist()


class TestInsertAndTouch:
    def test_insert_returns_id(self, db_conn):
        mid = insert_memory(db_conn, "sess1", "Q", "A", _make_embedding())
        assert mid > 0

    def test_touch_existing(self, db_conn):
        insert_memory(db_conn, "sess1", "Q", "A", _make_embedding())
        assert touch_if_exists(db_conn, "Q", "A") is True

    def test_touch_nonexistent(self, db_conn):
        assert touch_if_exists(db_conn, "Q", "A") is False

    def test_touch_with_project_path(self, db_conn):
        insert_memory(db_conn, "s1", "Q", "A", _make_embedding(), project_path="/proj")
        assert touch_if_exists(db_conn, "Q", "A", project_path="/proj") is True
        assert touch_if_exists(db_conn, "Q", "A", project_path="/other") is False


class TestFetch:
    def test_fetch_memories_by_ids(self, db_conn):
        id1 = insert_memory(db_conn, "s1", "Q1", "A1", _make_embedding())
        id2 = insert_memory(db_conn, "s1", "Q2", "A2", _make_embedding())
        memories = fetch_memories_by_ids(db_conn, [id1, id2])
        assert len(memories) == 2
        assert all(isinstance(m, Memory) for m in memories)

    def test_fetch_empty_ids(self, db_conn):
        assert fetch_memories_by_ids(db_conn, []) == []

    def test_fetch_with_project_filter(self, db_conn):
        id1 = insert_memory(db_conn, "s1", "Q1", "A1", _make_embedding(), "/proj")
        id2 = insert_memory(db_conn, "s1", "Q2", "A2", _make_embedding(), "/other")
        memories = fetch_memories_by_ids(db_conn, [id1, id2], project_path="/proj")
        assert len(memories) == 1
        assert memories[0].project_path == "/proj"

    def test_fetch_embeddings(self, db_conn):
        mid = insert_memory(db_conn, "s1", "Q", "A", _make_embedding())
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
        insert_memory(db_conn, "s1", "Q1", "A1", _make_embedding())
        insert_memory(db_conn, "s2", "Q2", "A2", _make_embedding())
        s = get_stats(db_conn)
        assert s.memories == 2
        assert s.sessions == 2
