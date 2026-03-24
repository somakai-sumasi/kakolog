"""CLIエントリポイント。手動で検索やステータス確認に使う。"""

import argparse
import sys

from .db import connection, init_db, get_stats
from .search import search


def cmd_search(args):
    results = search(args.query, limit=args.limit, project_path=args.project)
    if not results:
        print("No memories found.")
        return
    for r in results:
        print(f"[{r.id}] score={r.score:.4f} ({r.created_at})")
        print(f"  Q: {r.question[:100]}")
        print(f"  A: {r.answer[:200]}")
        print()


def cmd_stats(args):
    with connection() as conn:
        init_db(conn)
        stats = get_stats(conn)
    print(f"Total memories: {stats['memories']}")
    print(f"Total sessions: {stats['sessions']}")


def main():
    parser = argparse.ArgumentParser(prog="kakolog", description="Claude Code long-term memory")
    sub = parser.add_subparsers(dest="command")

    p_search = sub.add_parser("search", help="Search memories")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("-n", "--limit", type=int, default=5)
    p_search.add_argument("-p", "--project", default=None)

    p_stats = sub.add_parser("stats", help="Show memory stats")

    args = parser.parse_args()
    if args.command == "search":
        cmd_search(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
