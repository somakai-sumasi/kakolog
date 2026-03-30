from kakolog.db import _init_db


class TestInitDb:
    def test_creates_tables(self, db_conn):
        tables = {
            row[0]
            for row in db_conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'trigger')"
            ).fetchall()
        }
        assert "memories" in tables
        assert "fts_memories" in tables
        assert "vec_memories" in tables

    def test_creates_triggers(self, db_conn):
        triggers = {
            row[0]
            for row in db_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            ).fetchall()
        }
        assert "memories_ai" in triggers
        assert "memories_ad" in triggers
        assert "memories_au" in triggers

    def test_idempotent(self, db_conn):
        """_init_dbを2回呼んでもエラーにならない"""
        _init_db(db_conn)
