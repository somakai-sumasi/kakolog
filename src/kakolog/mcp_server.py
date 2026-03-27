"""kakolog MCP Server - Claude Code長期記憶エンジン"""

import sys

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from .config import add_exclude_path, get_exclude_paths, remove_exclude_path
from .embedder import get_model
from .search import search as do_search
from .service import save_session
from .db import connection, init_db
from .repository import get_stats

HOST = "127.0.0.1"
PORT = 7377

mcp = FastMCP("kakolog", host=HOST, port=PORT)


@mcp.tool()
def search(query: str, limit: int = 5, use_rerank: bool = False, use_mmr: bool = True) -> list[dict]:
    """過去のClaude Codeセッション会話を検索する。
    キーワード+意味検索のハイブリッド。具体的なキーワードほど精度が高い。
    use_rerank=Trueでcross-encoderリランキングを有効化（精度向上、速度低下）。
    use_mmr=TrueでMMR多様性リランキングを有効化（類似結果の重複を削減）。"""
    results = do_search(query, limit=limit, use_rerank=use_rerank, use_mmr=use_mmr)
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


@mcp.custom_route("/hook/save", methods=["POST"])
async def hook_save(request: Request) -> JSONResponse:
    """SessionEndフック専用エンドポイント。curl一発で呼べる軽量HTTP API。"""
    data = await request.json()
    if not data.get("session_id") or not data.get("transcript_path"):
        return JSONResponse({"error": "session_id and transcript_path are required"}, status_code=400)
    count = save_session(data["session_id"], data["transcript_path"], data.get("cwd"))
    return JSONResponse({"saved": count, "session_id": data["session_id"]})


@mcp.tool()
def exclude_list() -> list[str]:
    """保存除外パスの一覧を返す。"""
    return get_exclude_paths()


@mcp.tool()
def exclude_add(path: str) -> list[str]:
    """指定パス（前方一致）を保存除外リストに追加する。"""
    return add_exclude_path(path)


@mcp.tool()
def exclude_remove(path: str) -> list[str]:
    """指定パスを保存除外リストから削除する。"""
    return remove_exclude_path(path)


def main():
    print("[kakolog] Loading models...", file=sys.stderr)
    get_model()
    print("[kakolog] Model loaded. Starting MCP server.", file=sys.stderr)

    with connection() as conn:
        init_db(conn)

    print(f"[kakolog] MCP server on {HOST}:{PORT}", file=sys.stderr)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
