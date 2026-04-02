"""CLIエントリポイント。手動で検索やステータス確認に使う。"""

import argparse

from .repository import get_stats
from .search import search


def _cmd_search(args):
    results = search(
        args.query,
        limit=args.limit,
        project_path=args.project,
        use_rerank=args.rerank,
        use_mmr=args.mmr,
    )
    if not results:
        print("No memories found.")
        return
    for r in results:
        print(f"[{r.id}] score={r.score:.4f} ({r.last_accessed_at})")
        print(f"  U: {r.user_turn[:100]}")
        print(f"  A: {r.agent_turn[:200]}")
        print()


def _cmd_stats(_args):
    s = get_stats()
    print(f"Total memories: {s.memories}")
    print(f"Total sessions: {s.sessions}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kakolog", description="Claude Code long-term memory"
    )
    sub = parser.add_subparsers()

    p_search = sub.add_parser("search", help="Search memories")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("-n", "--limit", type=int, default=10)
    p_search.add_argument("-p", "--project", default=None)
    p_search.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking",
    )
    p_search.add_argument(
        "--no-mmr",
        dest="mmr",
        action="store_false",
        help="Disable MMR diversity reranking",
    )
    p_search.set_defaults(mmr=True, func=_cmd_search)

    p_stats = sub.add_parser("stats", help="Show memory stats")
    p_stats.set_defaults(func=_cmd_stats)

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
