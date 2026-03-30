import numpy as np

from kakolog.db import EMBEDDING_DIM


class TestEmbedder:
    def test_embed_query(self, mock_embedder):
        mock_embedder.encode.return_value = np.ones(EMBEDDING_DIM, dtype=np.float32)
        from kakolog.embedder import embed_query

        result = embed_query("テスト")
        assert len(result) == EMBEDDING_DIM
        mock_embedder.encode.assert_called_once()
        call_args = mock_embedder.encode.call_args
        assert "検索クエリ:" in call_args[0][0]

    def test_embed_document(self, mock_embedder):
        mock_embedder.encode.return_value = np.ones(EMBEDDING_DIM, dtype=np.float32)
        from kakolog.embedder import embed_document

        result = embed_document("テスト文書")
        assert len(result) == EMBEDDING_DIM
        call_args = mock_embedder.encode.call_args
        assert "検索文書:" in call_args[0][0]

    def test_embed_documents(self, mock_embedder):
        mock_embedder.encode.return_value = np.ones(
            (2, EMBEDDING_DIM), dtype=np.float32
        )
        from kakolog.embedder import embed_documents

        result = embed_documents(["text1", "text2"])
        assert len(result) == 2
