"""サービス層の共通処理。"""

from dataclasses import replace
from datetime import datetime, timezone

from ..models import Memory
from ..repository import update_memory


def touch_memories(memories: list[Memory]) -> None:
    """メモリのaccess_countをインクリメントしlast_accessed_atを現在時刻に更新する。
    トランザクション内で呼び出すこと。"""
    now = datetime.now(timezone.utc)
    for m in memories:
        update_memory(
            replace(
                m,
                last_accessed_at=now,
                access_count=m.access_count + 1,
            )
        )
