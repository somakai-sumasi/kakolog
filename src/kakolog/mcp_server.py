"""kakolog MCP Server - Claude Code長期記憶エンジン"""

import sys

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from .config import add_exclude_path, get_exclude_paths, remove_exclude_path
from .repository import get_stats
from .search import search as _search
from .service import save_session

HOST = "127.0.0.1"
PORT = 7377

mcp = FastMCP("kakolog", host=HOST, port=PORT)


@mcp.tool()
def search(
    query: str, limit: int = 10, use_rerank: bool = False, use_mmr: bool = True
) -> list[dict]:
    """過去のClaude Codeセッション会話を検索する。
    具体的な語を含む自然言語クエリが効果的（例: 「FTS5でどう全文検索を実装したか」）。"""
    results = _search(
        query, limit=limit, use_rerank=use_rerank, use_mmr=use_mmr
    )
    return [r.to_dict() for r in results]


@mcp.tool()
def save(
    session_id: str, transcript_path: str, project_path: str | None = None
) -> dict:
    """セッション会話をメモリに保存する。通常はSessionEndフックから呼ばれる。"""
    count = save_session(session_id, transcript_path, project_path)
    return {"saved": count, "session_id": session_id}


@mcp.tool()
def stats() -> dict:
    """メモリの統計情報を返す。"""
    s = get_stats()
    return {"memories": s.memories, "sessions": s.sessions}


@mcp.custom_route("/hook/save", methods=["POST"])
async def hook_save(request: Request) -> JSONResponse:
    """SessionEndフック専用エンドポイント。curl一発で呼べる軽量HTTP API。"""
    data = await request.json()
    if not data.get("session_id") or not data.get("transcript_path"):
        return JSONResponse(
            {"error": "session_id and transcript_path are required"},
            status_code=400,
        )
    count = save_session(
        data["session_id"], data["transcript_path"], data.get("cwd")
    )
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


def _warmup():
    """初回リクエスト遅延を避けるためモデルとDBを先行初期化。"""
    from .db import get_conn
    from .embedder import get_model

    get_model()
    get_conn()


def main():
    print("[kakolog] Warming up...", file=sys.stderr)
    _warmup()
    print(f"[kakolog] MCP server on {HOST}:{PORT}", file=sys.stderr)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
