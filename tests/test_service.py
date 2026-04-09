from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np

from kakolog.db import EMBEDDING_DIM
from kakolog.extractor import SessionMeta
from kakolog.models import Memory


class TestSaveSession:
    @patch("kakolog.service.save.transaction")
    @patch("kakolog.service.save.embed_documents")
    @patch("kakolog.service.save._build_chunks")
    @patch("kakolog.service.save.is_excluded", return_value=False)
    @patch(
        "kakolog.service.save.read_session_meta",
        return_value=SessionMeta(cwd=None, entrypoint="cli", first_timestamp=None),
    )
    def test_save_new_chunks(
        self, mock_meta, mock_excluded, mock_chunk, mock_embed, mock_conn
    ):
        from kakolog.chunker import TurnChunk

        mock_chunk.return_value = [
            TurnChunk(user_turn="U1", agent_turn="A1", content="U: U1\nA: A1"),
            TurnChunk(user_turn="U2", agent_turn="A2", content="U: U2\nA: A2"),
        ]
        mock_embed.return_value = [
            np.zeros(EMBEDDING_DIM).tolist(),
            np.zeros(EMBEDDING_DIM).tolist(),
        ]
        fake_conn = MagicMock()
        mock_conn.return_value.__enter__ = MagicMock(return_value=fake_conn)
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)

        with patch("kakolog.service.save.find_memory_by_content", return_value=None):
            with patch("kakolog.service.save.insert_memory") as mock_insert:
                from kakolog.service.save import save_session

                count = save_session("sess1", "/path/to/transcript.jsonl")

        assert count == 2
        assert mock_insert.call_count == 2

    @patch(
        "kakolog.service.save.read_session_meta",
        return_value=SessionMeta(
            cwd=None,
            entrypoint="sdk-cli",
            first_timestamp=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
        ),
    )
    def test_sdk_cli_returns_zero(self, mock_meta):
        from kakolog.service.save import save_session

        count = save_session("sess1", "/path")
        assert count == 0

    def test_subagents_path_returns_zero(self):
        from kakolog.service.save import save_session

        count = save_session(
            "sess1", "/home/user/.claude/projects/-/subagents/abc.jsonl"
        )
        assert count == 0

    @patch("kakolog.service.save.is_excluded", return_value=True)
    def test_excluded_path_returns_zero(self, mock_excluded):
        from kakolog.service.save import save_session

        count = save_session("sess1", "/path", project_path="/excluded")
        assert count == 0

    @patch("kakolog.service.save.transaction")
    @patch("kakolog.service.save.embed_documents")
    @patch("kakolog.service.save._build_chunks")
    @patch("kakolog.service.save.is_excluded", return_value=False)
    @patch(
        "kakolog.service.save.read_session_meta",
        return_value=SessionMeta(
            cwd=None,
            entrypoint="cli",
            first_timestamp=datetime.fromisoformat("2026-01-15T10:00:00+00:00"),
        ),
    )
    def test_first_timestamp_passed_to_insert(
        self, mock_meta, mock_excluded, mock_chunk, mock_embed, mock_conn
    ):
        from kakolog.chunker import TurnChunk

        mock_chunk.return_value = [
            TurnChunk(user_turn="U", agent_turn="A", content="U: U\nA: A")
        ]
        mock_embed.return_value = [np.zeros(EMBEDDING_DIM).tolist()]
        fake_conn = MagicMock()
        mock_conn.return_value.__enter__ = MagicMock(return_value=fake_conn)
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)

        with patch("kakolog.service.save.find_memory_by_content", return_value=None):
            with patch("kakolog.service.save.insert_memory") as mock_insert:
                from kakolog.service.save import save_session

                save_session("sess1", "/path/transcript.jsonl")

        assert mock_insert.call_args[0][0].last_accessed_at == datetime.fromisoformat(
            "2026-01-15T10:00:00+00:00"
        )

    @patch("kakolog.service.save.transaction")
    @patch("kakolog.service.save.embed_documents")
    @patch("kakolog.service.save._build_chunks")
    @patch("kakolog.service.save.is_excluded", return_value=False)
    @patch(
        "kakolog.service.save.read_session_meta",
        return_value=SessionMeta(cwd=None, entrypoint="cli", first_timestamp=None),
    )
    def test_empty_chunks(
        self, mock_meta, mock_excluded, mock_chunk, mock_embed, mock_conn
    ):
        mock_chunk.return_value = []

        from kakolog.service.save import save_session

        count = save_session("sess1", "/path")
        assert count == 0
        mock_embed.assert_not_called()

    @patch("kakolog.service.save.transaction")
    @patch("kakolog.service.save.embed_documents")
    @patch("kakolog.service.save._build_chunks")
    @patch("kakolog.service.save.is_excluded", return_value=False)
    @patch(
        "kakolog.service.save.read_session_meta",
        return_value=SessionMeta(cwd=None, entrypoint="cli", first_timestamp=None),
    )
    def test_similar_skips_insert_and_touches(
        self, mock_meta, mock_excluded, mock_chunk, mock_embed, mock_conn
    ):
        """類似メモリが見つかった場合、insertせずtouch_memoriesが呼ばれる。"""
        from kakolog.chunker import TurnChunk

        mock_chunk.return_value = [
            TurnChunk(user_turn="U1", agent_turn="A1", content="U: U1\nA: A1"),
        ]
        mock_embed.return_value = [np.zeros(EMBEDDING_DIM).tolist()]
        fake_conn = MagicMock()
        mock_conn.return_value.__enter__ = MagicMock(return_value=fake_conn)
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)

        similar_memory = Memory(
            id=99,
            session_id="sess1",
            user_turn="U1",
            agent_turn="A1",
            content="U: U1\nA: A1",
            created_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
            access_count=0,
            project_path=None,
        )

        with (
            patch("kakolog.service.save.find_memory_by_content", return_value=None),
            patch("kakolog.service.save._find_similar", return_value=similar_memory),
            patch("kakolog.service.save.touch_memories") as mock_touch,
            patch("kakolog.service.save.insert_memory") as mock_insert,
        ):
            from kakolog.service.save import save_session

            count = save_session("sess1", "/path/transcript.jsonl")

        assert count == 0
        mock_insert.assert_not_called()
        mock_touch.assert_called_once_with([similar_memory])


class TestTouchMemories:
    def test_increments_access_count(self, db_conn):
        """access_countがインクリメントされlast_accessed_atが更新される。"""
        from kakolog.db import transaction
        from kakolog.repository import (
            MemoryToSave,
            find_memory_by_content,
            insert_memory,
        )
        from kakolog.service import touch_memories

        with transaction():
            insert_memory(
                MemoryToSave(
                    session_id="s1",
                    user_turn="U",
                    agent_turn="A",
                    content="U: U\nA: A",
                    embedding=np.zeros(EMBEDDING_DIM).tolist(),
                )
            )

        memory = find_memory_by_content("U: U\nA: A")
        assert memory.access_count == 0
        old_accessed = memory.last_accessed_at

        with transaction():
            touch_memories([memory])

        updated = find_memory_by_content("U: U\nA: A")
        assert updated.access_count == 1
        assert updated.last_accessed_at.replace(tzinfo=None) > old_accessed.replace(
            tzinfo=None
        )

    def test_touch_multiple(self, db_conn):
        """複数メモリを一括更新できる。"""
        from kakolog.db import transaction
        from kakolog.repository import (
            MemoryToSave,
            fetch_memories_by_ids,
            insert_memory,
        )
        from kakolog.service import touch_memories

        with transaction():
            id1 = insert_memory(
                MemoryToSave(
                    session_id="s1",
                    user_turn="U1",
                    agent_turn="A1",
                    content="U: U1\nA: A1",
                    embedding=np.zeros(EMBEDDING_DIM).tolist(),
                )
            )
            id2 = insert_memory(
                MemoryToSave(
                    session_id="s1",
                    user_turn="U2",
                    agent_turn="A2",
                    content="U: U2\nA: A2",
                    embedding=np.ones(EMBEDDING_DIM).tolist(),
                )
            )

        memories = fetch_memories_by_ids([id1, id2])
        with transaction():
            touch_memories(memories)

        updated = fetch_memories_by_ids([id1, id2])
        for m in updated:
            assert m.access_count == 1


class TestFindSimilar:
    def test_returns_similar_memory(self, db_conn):
        """類似度が閾値以上のメモリを返す。"""
        from kakolog.db import transaction
        from kakolog.repository import MemoryToSave, insert_memory
        from kakolog.service.save import _find_similar

        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        embedding = vec.tolist()

        with transaction():
            insert_memory(
                MemoryToSave(
                    session_id="s1",
                    user_turn="U",
                    agent_turn="A",
                    content="U: U\nA: A",
                    embedding=embedding,
                    project_path="/proj",
                )
            )

        result = _find_similar(embedding, "/proj")
        assert result is not None
        assert result.content == "U: U\nA: A"

    def test_returns_none_for_different_project(self, db_conn):
        """異なるproject_pathのメモリは返さない。"""
        from kakolog.db import transaction
        from kakolog.repository import MemoryToSave, insert_memory
        from kakolog.service.save import _find_similar

        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        embedding = vec.tolist()

        with transaction():
            insert_memory(
                MemoryToSave(
                    session_id="s1",
                    user_turn="U",
                    agent_turn="A",
                    content="U: U\nA: A",
                    embedding=embedding,
                    project_path="/proj-a",
                )
            )

        result = _find_similar(embedding, "/proj-b")
        assert result is None

    def test_returns_none_for_low_similarity(self, db_conn):
        """類似度が閾値未満ならNoneを返す。"""
        from kakolog.db import transaction
        from kakolog.repository import MemoryToSave, insert_memory
        from kakolog.service.save import _find_similar

        vec1 = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        vec1 /= np.linalg.norm(vec1)
        vec2 = -vec1  # 正反対のベクトル（類似度 -1.0）

        with transaction():
            insert_memory(
                MemoryToSave(
                    session_id="s1",
                    user_turn="U",
                    agent_turn="A",
                    content="U: U\nA: A",
                    embedding=vec1.tolist(),
                    project_path="/proj",
                )
            )

        result = _find_similar(vec2.tolist(), "/proj")
        assert result is None
