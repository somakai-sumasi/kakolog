"""過去の全セッションを一括インポートする。"""

import json
import sys
import time
from pathlib import Path

from .chunker import chunk_session
from .config import is_excluded
from .db import connection, init_db
from .repository import insert_memory, touch_if_exists
from .embedder import embed_documents, get_model


def _extract_cwd(jsonl_path: Path) -> str | None:
    """JSONLの最初のエントリからcwdを取得する。"""
    try:
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if "cwd" in entry:
                    return entry["cwd"]
    except Exception:
        pass
    return None


def bulk_import(claude_projects_dir: Path | None = None):
    if claude_projects_dir is None:
        claude_projects_dir = Path.home() / ".claude" / "projects"

    jsonl_files = sorted(
        f for f in claude_projects_dir.rglob("*.jsonl")
        if "subagents" not in f.parts
    )
    print(f"Found {len(jsonl_files)} session files", file=sys.stderr)

    print("Loading model...", file=sys.stderr)
    get_model()

    with connection() as conn:
        init_db(conn)

        existing = {r[0] for r in conn.execute("SELECT DISTINCT session_id FROM memories").fetchall()}
        print(f"Already imported: {len(existing)} sessions", file=sys.stderr)

        total_memories = 0
        total_sessions = 0
        start = time.time()

        for i, jsonl_path in enumerate(jsonl_files):
            session_id = jsonl_path.stem
            if session_id in existing:
                continue

            project_path = _extract_cwd(jsonl_path)

            if is_excluded(project_path):
                continue

            try:
                chunks = chunk_session(jsonl_path)
            except Exception as e:
                print(f"  [{i+1}/{len(jsonl_files)}] Skip {session_id}: {e}", file=sys.stderr)
                continue

            if not chunks:
                continue

            texts = [f"{c.question}\n{c.answer}" for c in chunks]
            embeddings = embed_documents(texts)

            for chunk, emb in zip(chunks, embeddings):
                if not touch_if_exists(conn, chunk.question, chunk.answer, project_path):
                    insert_memory(conn, session_id, chunk.question, chunk.answer, emb, project_path)

            total_memories += len(chunks)
            total_sessions += 1

            elapsed = time.time() - start
            print(
                f"  [{i+1}/{len(jsonl_files)}] {session_id}: {len(chunks)} chunks "
                f"(total: {total_memories} memories, {total_sessions} sessions, {elapsed:.0f}s)",
                file=sys.stderr,
            )

    elapsed = time.time() - start
    print(f"\nDone: {total_sessions} sessions, {total_memories} memories in {elapsed:.0f}s", file=sys.stderr)


def main():
    bulk_import()


if __name__ == "__main__":
    main()
