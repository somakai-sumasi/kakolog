"""ドメインモデル定義。"""

from dataclasses import dataclass
from datetime import datetime


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
    access_count: int
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
    access_count: int
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
            access_count=m.access_count,
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
            access_count=self.access_count,
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
            "access_count": self.access_count,
            "project_path": self.project_path,
        }
