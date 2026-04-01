"""自然言語クエリ vs キーワードクエリの検索品質比較テスト。

実際のDBデータをもとに作成したテストデータで、
意味検索(ベクトル)とキーワード検索のハイブリッドが
どちらのクエリ形式でも正しい結果を返すかを検証する。

実モデルを使うため低速(~10秒)。
"""

import sqlite3

import numpy as np
import pytest
import sqlite_vec

from kakolog.db import EMBEDDING_DIM, _init_db
from kakolog.embedder import embed_document, embed_query
from kakolog.repository import MemoryToSave, insert_memory
from kakolog.search import search_keyword, search_vector, rrf_fuse


# テストデータ: 実際のDBから抽出した代表的な会話ペア
MEMORIES = [
    {
        "id_label": "hook_timeout",
        "user_turn": "SessionEnd hook が失敗する。transformersのimportが重くてタイムアウトになる",
        "agent_turn": "原因判明。transformers のimportが重くてClaudeのhookタイムアウト(~1秒)に引っかかってKILL"
        "されています。解決策: Pythonを直接起動せずcurlでバックグラウンド送信する形に変更します。",
    },
    {
        "id_label": "fts5_search",
        "user_turn": "SQLiteでFTS5を使った全文検索の実装方法を教えて",
        "agent_turn": "CREATE VIRTUAL TABLE fts_memories USING fts5(user_turn, agent_turn, content=memories)"
        "でFTS5仮想テーブルを作成します。trigram tokenizerを使うと部分一致検索が可能です。",
    },
    {
        "id_label": "git_push",
        "user_turn": "gitも更新しておいてリモートも",
        "agent_turn": "コミットしてリモートにプッシュしました。git push origin main を実行済みです。",
    },
    {
        "id_label": "mcp_registration",
        "user_turn": "MCPサーバーをlaunchdで常駐させてHTTPトランスポートで登録する方法は？",
        "agent_turn": "launchctl load でplistを読み込んでサーバーを起動してから、"
        "claude mcp add --transport http kakolog http://127.0.0.1:7377/mcp で登録します。"
        "stdioだとポート競合するので必ずHTTPトランスポートを使ってください。",
    },
    {
        "id_label": "vector_search",
        "user_turn": "ベクトル検索でsqlite-vecを使う実装を教えて",
        "agent_turn": "sqlite-vec拡張をenable_load_extensionで読み込み、vec_memoriesテーブルに"
        "serialize_float32でベクトルを保存します。SELECT ... WHERE embedding MATCH ? AND k = ?"
        "でkNN検索できます。",
    },
]


@pytest.fixture(scope="module")
def db_with_real_embeddings():
    """実モデルで埋め込みを生成してin-memoryDBに挿入"""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    _init_db(conn)

    label_to_id: dict[str, int] = {}
    for m in MEMORIES:
        text = f"{m['user_turn']}\n{m['agent_turn']}"
        embedding = embed_document(text)
        mem = MemoryToSave(
            session_id="test-session",
            user_turn=m["user_turn"],
            agent_turn=m["agent_turn"],
            embedding=embedding,
        )
        memory_id = insert_memory(conn, mem)
        label_to_id[m["id_label"]] = memory_id

    conn.commit()
    yield conn, label_to_id
    conn.close()


class TestKeywordSearch:
    def test_keyword_finds_hook_timeout(self, db_with_real_embeddings):
        conn, label_to_id = db_with_real_embeddings
        ids, _, _ = search_keyword(conn, "SessionEnd タイムアウト")
        assert label_to_id["hook_timeout"] in ids

    def test_keyword_finds_fts5(self, db_with_real_embeddings):
        conn, label_to_id = db_with_real_embeddings
        ids, _, _ = search_keyword(conn, "FTS5 全文検索")
        assert label_to_id["fts5_search"] in ids

    def test_keyword_finds_mcp(self, db_with_real_embeddings):
        conn, label_to_id = db_with_real_embeddings
        ids, _, _ = search_keyword(conn, "MCP launchd HTTP")
        assert label_to_id["mcp_registration"] in ids

    def test_keyword_empty_for_short_query(self, db_with_real_embeddings):
        conn, _ = db_with_real_embeddings
        # 2文字以下はtrigram非対応で空
        ids, _, _ = search_keyword(conn, "DB")
        assert ids == []


class TestVectorSearch:
    def test_natural_language_finds_hook_timeout(self, db_with_real_embeddings):
        conn, label_to_id = db_with_real_embeddings
        query_vec = embed_query("セッション終了時にフックがタイムアウトして失敗する問題")
        ids = search_vector(conn, query_vec)
        assert label_to_id["hook_timeout"] in ids

    def test_natural_language_finds_fts5(self, db_with_real_embeddings):
        conn, label_to_id = db_with_real_embeddings
        query_vec = embed_query("SQLiteで文字列を全文検索したい")
        ids = search_vector(conn, query_vec)
        assert label_to_id["fts5_search"] in ids

    def test_natural_language_finds_mcp(self, db_with_real_embeddings):
        conn, label_to_id = db_with_real_embeddings
        query_vec = embed_query("MCPサーバーをバックグラウンドで動かしてClaudeに接続する方法")
        ids = search_vector(conn, query_vec)
        assert label_to_id["mcp_registration"] in ids

    def test_natural_language_finds_vector_search(self, db_with_real_embeddings):
        conn, label_to_id = db_with_real_embeddings
        query_vec = embed_query("埋め込みベクトルをSQLiteに保存して類似検索する")
        ids = search_vector(conn, query_vec)
        assert label_to_id["vector_search"] in ids


class TestHybridSearch:
    """RRFで統合した場合、自然言語・キーワードどちらでも正しい結果が上位に来るか"""

    def _top_id(self, conn, query: str, label_to_id: dict) -> int:
        keyword_ids, hits, total = search_keyword(conn, query)
        query_vec = embed_query(query)
        vector_ids = search_vector(conn, query_vec)
        scores = rrf_fuse(keyword_ids, vector_ids, hits, total)
        return max(scores, key=lambda k: scores[k])

    def test_keyword_query_hook_timeout(self, db_with_real_embeddings):
        conn, label_to_id = db_with_real_embeddings
        top = self._top_id(conn, "SessionEnd hook タイムアウト", label_to_id)
        assert top == label_to_id["hook_timeout"]

    def test_natural_language_query_hook_timeout(self, db_with_real_embeddings):
        conn, label_to_id = db_with_real_embeddings
        top = self._top_id(conn, "セッション終了時にフックがKILLされてしまう原因と解決策", label_to_id)
        assert top == label_to_id["hook_timeout"]

    def test_keyword_query_fts5(self, db_with_real_embeddings):
        conn, label_to_id = db_with_real_embeddings
        top = self._top_id(conn, "SQLite FTS5 全文検索", label_to_id)
        assert top == label_to_id["fts5_search"]

    def test_natural_language_query_fts5(self, db_with_real_embeddings):
        conn, label_to_id = db_with_real_embeddings
        top = self._top_id(conn, "SQLiteでテキストを検索できる仕組みはどう実装したか", label_to_id)
        assert top == label_to_id["fts5_search"]

    def test_keyword_query_mcp(self, db_with_real_embeddings):
        conn, label_to_id = db_with_real_embeddings
        top = self._top_id(conn, "MCP HTTPトランスポート 登録", label_to_id)
        assert top == label_to_id["mcp_registration"]

    def test_natural_language_query_mcp(self, db_with_real_embeddings):
        conn, label_to_id = db_with_real_embeddings
        top = self._top_id(conn, "Claude CodeにMCPサーバーをHTTPで接続するにはどうすればいいか", label_to_id)
        assert top == label_to_id["mcp_registration"]
