"""セッション終了時に常駐サーバーへ保存リクエストを送るクライアント。

SessionEndフックから呼ばれる:
  stdin から {"session_id", "transcript_path", "cwd", ...} を受け取る
  常駐サーバーにHTTP POSTして即座に返る
"""

import json
import sys
import urllib.request
import urllib.error

SERVER_URL = "http://127.0.0.1:7377"


def main():
    hook_input = json.load(sys.stdin)
    payload = json.dumps({
        "session_id": hook_input["session_id"],
        "transcript_path": hook_input["transcript_path"],
        "cwd": hook_input.get("cwd"),
    }).encode()

    req = urllib.request.Request(
        f"{SERVER_URL}/save",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            result = json.loads(resp.read())
            print(f"kakolog: accepted ({result.get('session_id', '')})", file=sys.stderr)
    except urllib.error.URLError:
        print("kakolog: server not running, skipping save", file=sys.stderr)
    except Exception as e:
        print(f"kakolog: error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
