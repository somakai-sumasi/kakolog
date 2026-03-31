import sqlite3
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

DEFAULT_DB_PATH = Path.home() / ".kakolog" / "memory.db"
EMBEDDING_DIM = 256


@dataclass(frozen=True)
class Memory:
    """記憶のドメインモデル。DBから取得した1件の記憶を表す。"""

    id: int
    user_turn: str
    agent_turn: str
    last_accessed_at: str
    project_path: str | None


class connection:
    """DB接続のContext Manager。with文で使用する。init_dbも自動実行。"""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None

    def __enter__(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        _init_db(self.conn)
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
            self.conn.close()
        return False


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_turn TEXT NOT NULL,
            agent_turn TEXT NOT NULL,
            project_path TEXT,
            last_accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS fts_memories USING fts5(
            user_turn,
            agent_turn,
            content=memories,
            content_rowid=id,
            tokenize='trigram'
        );

        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO fts_memories(rowid, user_turn, agent_turn)
            VALUES (new.id, new.user_turn, new.agent_turn);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO fts_memories(fts_memories, rowid, user_turn, agent_turn)
            VALUES ('delete', old.id, old.user_turn, old.agent_turn);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO fts_memories(fts_memories, rowid, user_turn, agent_turn)
            VALUES ('delete', old.id, old.user_turn, old.agent_turn);
            INSERT INTO fts_memories(rowid, user_turn, agent_turn)
            VALUES (new.id, new.user_turn, new.agent_turn);
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
