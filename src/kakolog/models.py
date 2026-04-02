"""ドメインモデル定義。"""

import sqlite3
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


def from_row(row: sqlite3.Row, model_cls: type[T]) -> T:
    """sqlite3.Row を dataclass に変換する。"""
    return model_cls(**{f: row[f] for f in model_cls.__dataclass_fields__})


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
    created_at: str
    last_accessed_at: str
    project_path: str | None


@dataclass(frozen=True)
class SearchResult:
    """検索結果1件。スコア付きの記憶。"""

    id: int
    user_turn: str
    agent_turn: str
    content: str
    score: float
    created_at: str
    last_accessed_at: str
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
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
            "project_path": self.project_path,
        }
