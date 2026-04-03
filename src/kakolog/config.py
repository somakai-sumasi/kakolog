"""ユーザー設定の読み書き (~/.kakolog/config.json)"""

import functools
import json
from pathlib import Path

CONFIG_PATH = Path.home() / ".kakolog" / "config.json"

SIMILARITY_THRESHOLD = 0.96


def _load() -> dict:
    if not CONFIG_PATH.exists():
        return {"exclude_paths": []}
    try:
        return json.loads(CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {"exclude_paths": []}


def _save(config: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2))


@functools.lru_cache(maxsize=1)
def _get_exclude_paths_cached() -> tuple[str, ...]:
    return tuple(_load().get("exclude_paths", []))


def get_exclude_paths() -> list[str]:
    return list(_get_exclude_paths_cached())


def add_exclude_path(path: str) -> list[str]:
    config = _load()
    paths = config.setdefault("exclude_paths", [])
    if path not in paths:
        paths.append(path)
        _save(config)
        _get_exclude_paths_cached.cache_clear()
    return paths


def remove_exclude_path(path: str) -> list[str]:
    config = _load()
    paths = config.setdefault("exclude_paths", [])
    config["exclude_paths"] = [p for p in paths if p != path]
    _save(config)
    _get_exclude_paths_cached.cache_clear()
    return config["exclude_paths"]


def is_excluded(project_path: str | None) -> bool:
    if not project_path:
        return False
    for excluded in _get_exclude_paths_cached():
        if project_path == excluded or project_path.startswith(
            excluded.rstrip("/") + "/"
        ):
            return True
    return False


_EXCLUDED_ENTRYPOINTS = frozenset({"sdk-cli"})
_EXCLUDED_PATH_PARTS = frozenset({"subagents"})


def is_excluded_session(transcript_path: str, entrypoint: str | None) -> bool:
    """セッション単位の除外判定。パス部分一致 or entrypoint一致。"""
    if any(part in _EXCLUDED_PATH_PARTS for part in Path(transcript_path).parts):
        return True
    if entrypoint in _EXCLUDED_ENTRYPOINTS:
        return True
    return False
