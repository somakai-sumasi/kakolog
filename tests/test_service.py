from unittest.mock import MagicMock, patch

import numpy as np

from kakolog.db import EMBEDDING_DIM


class TestSaveSession:
    @patch("kakolog.service.connection")
    @patch("kakolog.service.embed_documents")
    @patch("kakolog.service.chunk_session")
    @patch("kakolog.service.is_excluded", return_value=False)
    def test_save_new_chunks(self, mock_excluded, mock_chunk, mock_embed, mock_conn):
        from kakolog.chunker import QAChunk

        mock_chunk.return_value = [
            QAChunk(question="Q1", answer="A1"),
            QAChunk(question="Q2", answer="A2"),
        ]
        mock_embed.return_value = [
            np.zeros(EMBEDDING_DIM).tolist(),
            np.zeros(EMBEDDING_DIM).tolist(),
        ]
        fake_conn = MagicMock()
        mock_conn.return_value.__enter__ = MagicMock(return_value=fake_conn)
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)

        with patch("kakolog.service.touch_if_exists", return_value=False):
            with patch("kakolog.service.insert_memory") as mock_insert:
                from kakolog.service import save_session

                count = save_session("sess1", "/path/to/transcript.jsonl")

        assert count == 2
        assert mock_insert.call_count == 2

    @patch("kakolog.service.is_excluded", return_value=True)
    def test_excluded_path_returns_zero(self, mock_excluded):
        from kakolog.service import save_session

        count = save_session("sess1", "/path", project_path="/excluded")
        assert count == 0

    @patch("kakolog.service.connection")
    @patch("kakolog.service.embed_documents")
    @patch("kakolog.service.chunk_session")
    @patch("kakolog.service.is_excluded", return_value=False)
    def test_empty_chunks(self, mock_excluded, mock_chunk, mock_embed, mock_conn):
        mock_chunk.return_value = []

        from kakolog.service import save_session

        count = save_session("sess1", "/path")
        assert count == 0
        mock_embed.assert_not_called()
