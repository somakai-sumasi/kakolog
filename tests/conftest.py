import sqlite3
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import sqlite_vec

from kakolog.db import EMBEDDING_DIM, _current_conn, _init_db


@pytest.fixture()
def db_conn():
    """in-memory SQLite + sqlite-vec 拡張付き接続。
    _current_conn にセットするのでrepository関数がそのまま動く。"""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    _init_db(conn)
    token = _current_conn.set(conn)
    yield conn
    _current_conn.reset(token)
    conn.close()


@pytest.fixture()
def sample_embedding():
    """テスト用256次元ベクトル (正規化済み)"""
    vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


@pytest.fixture()
def mock_embedder():
    """embedder.get_model() をモックして実モデルのロードを回避"""
    fake_model = MagicMock()
    fake_model.encode.return_value = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    with patch("kakolog.embedder.get_model", return_value=fake_model):
        yield fake_model


@pytest.fixture()
def tmp_config(tmp_path):
    """config.py の CONFIG_PATH を一時ディレクトリに差し替え"""
    from kakolog.config import _get_exclude_paths_cached

    config_path = tmp_path / "config.json"
    with patch("kakolog.config.CONFIG_PATH", config_path):
        _get_exclude_paths_cached.cache_clear()
        yield config_path
    _get_exclude_paths_cached.cache_clear()


@pytest.fixture()
def sample_jsonl(tmp_path):
    """テスト用JONLトランスクリプト"""
    import json

    messages = [
        {
            "message": {
                "role": "user",
                "content": "Pythonでリストの重複を削除するには？",
            }
        },
        {
            "message": {
                "role": "assistant",
                "content": "set()を使うか、dict.fromkeys()を使います。例: list(set(my_list))",
            }
        },
        {
            "message": {
                "role": "user",
                "content": "SQLiteでFTS5を使う方法を教えて",
            }
        },
        {
            "message": {
                "role": "assistant",
                "content": "CREATE VIRTUAL TABLE ... USING fts5() でFTS5仮想テーブルを作成します。",
            }
        },
    ]
    path = tmp_path / "transcript.jsonl"
    with open(path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
    return path
