"""JSONLトランスクリプトのI/Oユーティリティ。"""

import json
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
