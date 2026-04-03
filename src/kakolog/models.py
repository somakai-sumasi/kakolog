"""ドメインモデル定義。"""

from dataclasses import dataclass, replace
from datetime import datetime


@dataclass(frozen=True)
class ConversationPair:
    """抽出された1組の会話ペア。"""

    user_turn: str
    agent_turn: str
    timestamp: datetime | None = None


@dataclass(frozen=True)
class Memory:
    """DBから取得した1件の記憶。"""

    id: int
    user_turn: str
    agent_turn: str
    content: str
    created_at: datetime
    last_accessed_at: datetime
    access_count: int
    project_path: str | None


@dataclass(frozen=True)
class SearchResult:
    """検索結果1件。Memoryにスコアを付与したラッパー。"""

    memory: Memory
    score: float

    @classmethod
    def from_memory(cls, m: Memory, score: float) -> "SearchResult":
        return cls(memory=m, score=score)

    def with_score(self, score: float) -> "SearchResult":
        return replace(self, score=score)

    @property
    def id(self) -> int:
        return self.memory.id

    def to_dict(self) -> dict:
        m = self.memory
        return {
            "user_turn": m.user_turn,
            "agent_turn": m.agent_turn,
            "content": m.content,
            "score": self.score,
            "created_at": m.created_at.isoformat(),
            "last_accessed_at": m.last_accessed_at.isoformat(),
            "access_count": m.access_count,
            "project_path": m.project_path,
        }


@dataclass(frozen=True)
class Stats:
    """メモリの統計情報。"""

    memories: int
    sessions: int
