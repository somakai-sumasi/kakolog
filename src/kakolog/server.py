"""常駐HTTPサーバー。モデルをメモリに保持し、保存・検索を高速に処理する。"""

import json
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

from .db import connection, init_db, get_stats
from .embedder import get_model
from .reranker import get_reranker
from .search import search as do_search
from .service import save_session

HOST = "127.0.0.1"
PORT = 7377


class MemoryHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[kakolog] {args[0]}", file=sys.stderr)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        return json.loads(body) if body else {}

    def _respond(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def do_POST(self):
        if self.path == "/save":
            self._handle_save()
        elif self.path == "/search":
            self._handle_search()
        else:
            self._respond(404, {"error": "not found"})

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok"})
        elif self.path == "/stats":
            self._handle_stats()
        else:
            self._respond(404, {"error": "not found"})

    def _handle_save(self):
        try:
            data = self._read_json()
            # 即座に202を返し、バックグラウンドで処理
            self._respond(202, {"status": "accepted", "session_id": data.get("session_id")})
            threading.Thread(target=_save_worker, args=(data,), daemon=True).start()
        except Exception as e:
            self._respond(500, {"error": str(e)})

    def _handle_search(self):
        try:
            data = self._read_json()
            query = data.get("query", "")
            limit = data.get("limit", 5)
            project_path = data.get("project_path")

            results = do_search(query, limit=limit, project_path=project_path)
            self._respond(200, {
                "results": [
                    {
                        "id": r.id,
                        "question": r.question,
                        "answer": r.answer,
                        "score": r.score,
                        "created_at": r.created_at,
                        "project_path": r.project_path,
                    }
                    for r in results
                ]
            })
        except Exception as e:
            self._respond(500, {"error": str(e)})

    def _handle_stats(self):
        try:
            with connection() as conn:
                init_db(conn)
                stats = get_stats(conn)
            self._respond(200, stats)
        except Exception as e:
            self._respond(500, {"error": str(e)})


def _save_worker(data: dict):
    try:
        session_id = data["session_id"]
        transcript_path = data["transcript_path"]
        project_path = data.get("cwd")

        count = save_session(session_id, transcript_path, project_path)
        print(f"[kakolog] Saved {count} memories from session {session_id}", file=sys.stderr)
    except Exception as e:
        print(f"[kakolog] Save error: {e}", file=sys.stderr)


def main():
    print(f"[kakolog] Loading models...", file=sys.stderr)
    get_model()
    get_reranker()
    print(f"[kakolog] Models loaded. Starting server on {HOST}:{PORT}", file=sys.stderr)

    with connection() as conn:
        init_db(conn)

    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadingHTTPServer((HOST, PORT), MemoryHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[kakolog] Shutting down.", file=sys.stderr)
        server.shutdown()


if __name__ == "__main__":
    main()
