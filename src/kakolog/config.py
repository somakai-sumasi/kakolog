"""ユーザー設定の読み書き (~/.kakolog/config.json)"""

import json
from pathlib import Path

CONFIG_PATH = Path.home() / ".kakolog" / "config.json"


def _load() -> dict:
    if not CONFIG_PATH.exists():
        return {"exclude_paths": []}
    try:
        return json.loads(CONFIG_PATH.read_text())
    except Exception:
        return {"exclude_paths": []}


def _save(config: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2))


def get_exclude_paths() -> list[str]:
    return _load().get("exclude_paths", [])


def add_exclude_path(path: str) -> list[str]:
    config = _load()
    paths = config.setdefault("exclude_paths", [])
    if path not in paths:
        paths.append(path)
        _save(config)
    return paths


def remove_exclude_path(path: str) -> list[str]:
    config = _load()
    paths = config.setdefault("exclude_paths", [])
    config["exclude_paths"] = [p for p in paths if p != path]
    _save(config)
    return config["exclude_paths"]


def is_excluded(project_path: str | None) -> bool:
    if not project_path:
        return False
    for excluded in get_exclude_paths():
        if project_path == excluded or project_path.startswith(excluded.rstrip("/") + "/"):
            return True
    return False
