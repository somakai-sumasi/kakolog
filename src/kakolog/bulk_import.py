"""過去の全セッションを一括インポートする。"""

import sys
import time
from pathlib import Path

from .db import get_conn
from .embedder import get_model
from .repository import get_existing_session_ids
from .service import save_session


def bulk_import(claude_projects_dir: Path | None = None):
    if claude_projects_dir is None:
        claude_projects_dir = Path.home() / ".claude" / "projects"

    jsonl_files = sorted(claude_projects_dir.rglob("*.jsonl"))
    print(f"Found {len(jsonl_files)} session files", file=sys.stderr)

    print("Loading model...", file=sys.stderr)
    get_model()

    get_conn()  # ensure DB is initialized
    existing = get_existing_session_ids()
    print(f"Already imported: {len(existing)} sessions", file=sys.stderr)

    total_memories = 0
    total_sessions = 0
    start = time.time()

    for i, jsonl_path in enumerate(jsonl_files):
        session_id = jsonl_path.stem
        if session_id in existing:
            continue

        try:
            count = save_session(session_id, str(jsonl_path))
        except Exception as e:
            print(
                f"  [{i + 1}/{len(jsonl_files)}] Skip {session_id}: {e}",
                file=sys.stderr,
            )
            continue

        if count == 0:
            continue

        total_memories += count
        total_sessions += 1

        elapsed = time.time() - start
        print(
            f"  [{i + 1}/{len(jsonl_files)}] {session_id}: {count} chunks "
            f"(total: {total_memories} memories, {total_sessions} sessions, {elapsed:.0f}s)",
            file=sys.stderr,
        )

    elapsed = time.time() - start
    print(
        f"\nDone: {total_sessions} sessions, {total_memories} memories in {elapsed:.0f}s",
        file=sys.stderr,
    )


def main():
    bulk_import()


if __name__ == "__main__":
    main()
