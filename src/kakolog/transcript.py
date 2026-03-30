"""JSONLトランスクリプトのI/Oユーティリティ。"""

import json
from dataclasses import dataclass
from pathlib import Path


def iter_jsonl(transcript_path: str | Path):
    """JSONLファイルを1行ずつパースするジェネレータ。"""
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def parse_jsonl(transcript_path: str | Path) -> list[dict]:
    return list(iter_jsonl(transcript_path))


@dataclass(frozen=True)
class SessionMeta:
    cwd: str | None
    entrypoint: str | None
    first_timestamp: str | None


def read_session_meta(transcript_path: str) -> SessionMeta:
    """JSONLからcwd・entrypoint・first_timestampを返す。"""
    cwd = None
    entrypoint = None
    first_timestamp = None
    try:
        for entry in iter_jsonl(transcript_path):
            if first_timestamp is None and entry.get("timestamp"):
                first_timestamp = entry["timestamp"]
            if cwd is None and entry.get("cwd"):
                cwd = entry["cwd"]
            if entrypoint is None and entry.get("entrypoint"):
                entrypoint = entry["entrypoint"]
            if (
                cwd is not None
                and entrypoint is not None
                and first_timestamp is not None
            ):
                break
    except Exception:
        pass
    return SessionMeta(cwd=cwd, entrypoint=entrypoint, first_timestamp=first_timestamp)
