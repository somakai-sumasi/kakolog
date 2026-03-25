import sqlite3
from pathlib import Path

import sqlite_vec
from sqlite_vec import serialize_float32

DEFAULT_DB_PATH = Path.home() / ".kakolog" / "memory.db"
EMBEDDING_DIM = 256


class connection:
    """DB接続のContext Manager。with文で使用する。"""

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
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
        return False


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS memories(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            project_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS fts_memories USING fts5(
            question,
            answer,
            content=memories,
            content_rowid=id,
            tokenize='trigram'
        );

        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO fts_memories(rowid, question, answer)
            VALUES (new.id, new.question, new.answer);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO fts_memories(fts_memories, rowid, question, answer)
            VALUES ('delete', old.id, old.question, old.answer);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO fts_memories(fts_memories, rowid, question, answer)
            VALUES ('delete', old.id, old.question, old.answer);
            INSERT INTO fts_memories(rowid, question, answer)
            VALUES (new.id, new.question, new.answer);
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


def touch_if_exists(conn: sqlite3.Connection, question: str, answer: str, project_path: str | None = None) -> bool:
    """同一Q&A+project_pathが存在すればcreated_atを更新してTrueを返す。"""
    row = conn.execute(
        "SELECT id FROM memories WHERE question = ? AND answer = ? AND project_path IS ? LIMIT 1",
        [question, answer, project_path],
    ).fetchone()
    if row:
        conn.execute(
            "UPDATE memories SET created_at = CURRENT_TIMESTAMP WHERE id = ?",
            [row[0]],
        )
        conn.commit()
        return True
    return False


def insert_memory(
    conn: sqlite3.Connection,
    session_id: str,
    question: str,
    answer: str,
    embedding: list[float],
    project_path: str | None = None,
) -> int:
    cursor = conn.execute(
        "INSERT INTO memories(session_id, question, answer, project_path) VALUES (?, ?, ?, ?)",
        [session_id, question, answer, project_path],
    )
    memory_id = cursor.lastrowid
    conn.execute(
        "INSERT INTO vec_memories(memory_id, embedding) VALUES (?, ?)",
        [memory_id, serialize_float32(embedding)],
    )
    conn.commit()
    return memory_id


def get_stats(conn: sqlite3.Connection) -> dict:
    memories = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    sessions = conn.execute("SELECT COUNT(DISTINCT session_id) FROM memories").fetchone()[0]
    return {"memories": memories, "sessions": sessions}
