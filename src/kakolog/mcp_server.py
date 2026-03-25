"""kakolog MCP Server - Claude Code長期記憶エンジン"""

import sys

from mcp.server.fastmcp import FastMCP

from .embedder import get_model
from .reranker import get_reranker
from .search import search as do_search
from .service import save_session
from .db import connection, init_db, get_stats

HOST = "127.0.0.1"
PORT = 7377

mcp = FastMCP("kakolog", host=HOST, port=PORT)


@mcp.tool()
def search(query: str, limit: int = 5) -> list[dict]:
    """過去のClaude Codeセッション会話を検索する。
    キーワード+意味検索のハイブリッド。具体的なキーワードほど精度が高い。"""
    results = do_search(query, limit=limit)
    return [
        {
            "question": r.question,
            "answer": r.answer,
            "score": r.score,
            "created_at": r.created_at,
            "project_path": r.project_path,
        }
        for r in results
    ]


@mcp.tool()
def save(session_id: str, transcript_path: str, project_path: str | None = None) -> dict:
    """セッション会話をメモリに保存する。通常はSessionEndフックから呼ばれる。"""
    count = save_session(session_id, transcript_path, project_path)
    return {"saved": count, "session_id": session_id}


@mcp.tool()
def stats() -> dict:
    """メモリの統計情報を返す。"""
    with connection() as conn:
        init_db(conn)
        return get_stats(conn)


def main():
    print("[kakolog] Loading models...", file=sys.stderr)
    get_model()
    get_reranker()
    print("[kakolog] Models loaded. Starting MCP server.", file=sys.stderr)

    with connection() as conn:
        init_db(conn)

    print(f"[kakolog] MCP server on {HOST}:{PORT}", file=sys.stderr)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
