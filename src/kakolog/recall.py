"""セッション開始時に関連記憶を検索して注入するエントリポイント。

SessionStartフックから呼ばれる:
  stdin から {"session_id", "cwd", ...} を受け取る
  stdout にコンテキスト注入用テキストを出力する
"""

import json
import sys

from .search import search


def recall(query: str, project_path: str | None = None, limit: int = 5) -> str:
    results = search(query, limit=limit, project_path=project_path)
    if not results:
        return ""

    lines = ["<kakolog>", "以下は過去のセッションから検索された関連記憶です:", ""]
    for r in results:
        lines.append(f"## Q: {r.question[:200]}")
        lines.append(f"A: {r.answer[:500]}")
        lines.append(f"(score: {r.score:.4f}, date: {r.created_at})")
        lines.append("")
    lines.append("</kakolog>")
    return "\n".join(lines)


def main():
    hook_input = json.load(sys.stdin)
    project_path = hook_input.get("cwd")

    # SessionStartでは具体的なクエリがないため、プロジェクトパスで直近の記憶を返す
    # 実際のクエリ検索はUserPromptSubmitフックで行う方が効果的
    results = search("", limit=5, project_path=project_path)
    if results:
        output = recall("", project_path=project_path)
        print(output)


if __name__ == "__main__":
    main()
