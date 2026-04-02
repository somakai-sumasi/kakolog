"""ドメインモデル定義。"""

import sqlite3
from dataclasses import dataclass, fields
from datetime import datetime
from typing import TypeVar

T = TypeVar("T")


def _parse_timestamp(value: str | None) -> datetime | None:
    """DB格納形式の文字列をdatetimeに変換する。"""
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def from_row(row: sqlite3.Row, model_cls: type[T]) -> T:
    """sqlite3.Row を dataclass に変換する。
    datetime型フィールドは文字列から自動変換する。"""
    hints = {f.name: f.type for f in fields(model_cls)}
    values = {}
    for f in model_cls.__dataclass_fields__:
        v = row[f]
        if hints[f] is datetime and isinstance(v, str):
            v = _parse_timestamp(v)
        values[f] = v
    return model_cls(**values)


def columns_of(model_cls: type) -> str:
    """dataclass のフィールド名をSQLカラム列挙文字列として返す。"""
    return ", ".join(model_cls.__dataclass_fields__)


@dataclass(frozen=True)
class ConversationPair:
    """抽出された1組の会話ペア。"""

    user_turn: str
    agent_turn: str
    timestamp: str | None = None


@dataclass(frozen=True)
class Memory:
    """DBから取得した1件の記憶。"""

    id: int
    user_turn: str
    agent_turn: str
    content: str
    created_at: datetime
    last_accessed_at: datetime
    project_path: str | None


@dataclass(frozen=True)
class SearchResult:
    """検索結果1件。スコア付きの記憶。"""

    id: int
    user_turn: str
    agent_turn: str
    content: str
    score: float
    created_at: datetime
    last_accessed_at: datetime
    project_path: str | None

    @classmethod
    def from_memory(cls, m: Memory, score: float) -> "SearchResult":
        return cls(
            id=m.id,
            user_turn=m.user_turn,
            agent_turn=m.agent_turn,
            content=m.content,
            score=score,
            created_at=m.created_at,
            last_accessed_at=m.last_accessed_at,
            project_path=m.project_path,
        )

    def with_score(self, score: float) -> "SearchResult":
        return SearchResult(
            id=self.id,
            user_turn=self.user_turn,
            agent_turn=self.agent_turn,
            content=self.content,
            score=score,
            created_at=self.created_at,
            last_accessed_at=self.last_accessed_at,
            project_path=self.project_path,
        )

    def to_dict(self) -> dict:
        return {
            "user_turn": self.user_turn,
            "agent_turn": self.agent_turn,
            "content": self.content,
            "score": self.score,
            "created_at": self.created_at.isoformat(),
            "last_accessed_at": self.last_accessed_at.isoformat(),
            "project_path": self.project_path,
        }
