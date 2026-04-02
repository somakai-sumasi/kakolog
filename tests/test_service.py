from unittest.mock import MagicMock, patch

import numpy as np

from kakolog.db import EMBEDDING_DIM
from kakolog.transcript import SessionMeta


class TestSaveSession:
    @patch("kakolog.service.transaction")
    @patch("kakolog.service.embed_documents")
    @patch("kakolog.service.chunk_session")
    @patch("kakolog.service.is_excluded", return_value=False)
    @patch(
        "kakolog.service.read_session_meta",
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

        with patch("kakolog.service.find_memory_by_content", return_value=None):
            with patch("kakolog.service.insert_memory") as mock_insert:
                from kakolog.service import save_session

                count = save_session("sess1", "/path/to/transcript.jsonl")

        assert count == 2
        assert mock_insert.call_count == 2

    @patch(
        "kakolog.service.read_session_meta",
        return_value=SessionMeta(
            cwd=None, entrypoint="sdk-cli", first_timestamp="2026-01-01T00:00:00Z"
        ),
    )
    def test_sdk_cli_returns_zero(self, mock_meta):
        from kakolog.service import save_session

        count = save_session("sess1", "/path")
        assert count == 0

    def test_subagents_path_returns_zero(self):
        from kakolog.service import save_session

        count = save_session(
            "sess1", "/home/user/.claude/projects/-/subagents/abc.jsonl"
        )
        assert count == 0

    @patch("kakolog.service.is_excluded", return_value=True)
    def test_excluded_path_returns_zero(self, mock_excluded):
        from kakolog.service import save_session

        count = save_session("sess1", "/path", project_path="/excluded")
        assert count == 0

    @patch("kakolog.service.transaction")
    @patch("kakolog.service.embed_documents")
    @patch("kakolog.service.chunk_session")
    @patch("kakolog.service.is_excluded", return_value=False)
    @patch(
        "kakolog.service.read_session_meta",
        return_value=SessionMeta(
            cwd=None, entrypoint="cli", first_timestamp="2026-01-15T10:00:00Z"
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

        with patch("kakolog.service.find_memory_by_content", return_value=None):
            with patch("kakolog.service.insert_memory") as mock_insert:
                from kakolog.service import save_session

                save_session("sess1", "/path/transcript.jsonl")

        assert mock_insert.call_args[0][0].last_accessed_at == "2026-01-15T10:00:00Z"

    @patch("kakolog.service.transaction")
    @patch("kakolog.service.embed_documents")
    @patch("kakolog.service.chunk_session")
    @patch("kakolog.service.is_excluded", return_value=False)
    @patch(
        "kakolog.service.read_session_meta",
        return_value=SessionMeta(cwd=None, entrypoint="cli", first_timestamp=None),
    )
    def test_empty_chunks(
        self, mock_meta, mock_excluded, mock_chunk, mock_embed, mock_conn
    ):
        mock_chunk.return_value = []

        from kakolog.service import save_session

        count = save_session("sess1", "/path")
        assert count == 0
        mock_embed.assert_not_called()
