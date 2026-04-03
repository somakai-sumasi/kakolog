import sqlite3
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path

import sqlite_vec

DEFAULT_DB_PATH = Path.home() / ".kakolog" / "memory.db"
EMBEDDING_DIM = 256


def _parse_timestamp(val: bytes) -> datetime:
    """TIMESTAMP型カラムのカスタムコンバータ。"Z"付きISO8601にも対応。"""
    s = val.decode()
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


sqlite3.register_converter("TIMESTAMP", _parse_timestamp)

_current_conn: ContextVar[sqlite3.Connection] = ContextVar("_current_conn")


def _open_conn(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """DB接続を作成し、拡張ロード+スキーマ初期化を行う。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    _init_db(conn)
    return conn


def get_conn() -> sqlite3.Connection:
    """現在のDB接続を取得する。未接続なら自動で作成する。"""
    try:
        return _current_conn.get()
    except LookupError:
        conn = _open_conn()
        _current_conn.set(conn)
        return conn


_tx_depth: ContextVar[int] = ContextVar("_tx_depth", default=0)


class transaction:
    """書き込みトランザクション用のContext Manager。
    ネスト対応: 最外側のみcommit/rollbackを実行する。
    部分ロールバック非対応（SAVEPOINTは使わない）。
    内側で例外が発生した場合は最外側まで伝播してから全体rollback。"""

    def __init__(self):
        self.conn: sqlite3.Connection | None = None
        self._is_outermost = False

    def __enter__(self) -> None:
        depth = _tx_depth.get()
        self._is_outermost = depth == 0
        _tx_depth.set(depth + 1)
        self.conn = get_conn()

    def __exit__(self, exc_type, exc_val, exc_tb):
        _tx_depth.set(_tx_depth.get() - 1)
        if self._is_outermost and self.conn:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
        return False


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_turn TEXT NOT NULL,
            agent_turn TEXT NOT NULL,
            content TEXT NOT NULL,
            project_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            access_count INTEGER DEFAULT 0
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS fts_memories USING fts5(
            content,
            content=memories,
            content_rowid=id,
            tokenize='trigram'
        );

        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO fts_memories(rowid, content)
            VALUES (new.id, new.content);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO fts_memories(fts_memories, rowid, content)
            VALUES ('delete', old.id, old.content);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO fts_memories(fts_memories, rowid, content)
            VALUES ('delete', old.id, old.content);
            INSERT INTO fts_memories(rowid, content)
            VALUES (new.id, new.content);
        END;
    """)

    # vec0テーブルはCREATE IF NOT EXISTSが使えないのでtry
    try:
        conn.execute(f"""
            CREATE VIRTUAL TABLE vec_memories USING vec0(
                memory_id integer primary key,
                embedding float[{EMBEDDING_DIM}]
            )
        """)
    except sqlite3.OperationalError:
        pass  # already exists

    conn.commit()
