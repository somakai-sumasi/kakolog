import functools
import os
import shutil
import subprocess
from dataclasses import dataclass

import MeCab

from .cleaner import is_empty_answer, is_trivial
from .models import ConversationPair


def _find_mecabrc() -> str | None:
    """mecabrcのパスを自動検出する。"""
    if os.environ.get("MECABRC"):
        return os.environ["MECABRC"]
    if shutil.which("mecab-config"):
        try:
            sysconfdir = subprocess.check_output(
                ["mecab-config", "--sysconfdir"], text=True
            ).strip()
            path = os.path.join(sysconfdir, "mecabrc")
            if os.path.exists(path):
                return path
        except subprocess.CalledProcessError:
            pass
    for p in ["/opt/homebrew/etc/mecabrc", "/usr/local/etc/mecabrc", "/etc/mecabrc"]:
        if os.path.exists(p):
            return p
    return None


@dataclass(frozen=True)
class TurnChunk:
    user_turn: str
    agent_turn: str
    content: str
    timestamp: str | None = None


SHORT_USER_THRESHOLD = 30
MAX_MERGE_TURNS = 3
AGENT_TRUNCATE_LEN = 200
MIN_CHUNK_SIZE = 50
WCOST_THRESHOLD = 6000  # これ以上のコストの名詞は「珍しい語」と判定


def _format_content(user_turn: str, agent_turn: str) -> str:
    return f"U: {user_turn}\nA: {agent_turn}"


@functools.lru_cache(maxsize=1)
def _get_tagger() -> MeCab.Tagger:
    mecabrc = _find_mecabrc()
    if mecabrc:
        os.environ.setdefault("MECABRC", mecabrc)
    return MeCab.Tagger()


def has_important_words(text: str) -> bool:
    """珍しい名詞（wcost >= 閾値）が含まれていればTrueを返す。
    未知語・固有名詞に加え、辞書にあっても珍しい語を検出する。"""
    tagger = _get_tagger()
    node = tagger.parseToNode(text)
    while node:
        if node.surface:
            features = node.feature.split(",")
            pos = features[0]
            if pos == "名詞" and node.wcost >= WCOST_THRESHOLD:
                return True
        node = node.next
    return False


def _is_worth_saving(user_turn: str, agent_turn: str) -> bool:
    if not user_turn or not agent_turn:
        return False
    if is_empty_answer(agent_turn):
        return False
    if is_trivial(user_turn):
        return False
    if len(user_turn) + len(agent_turn) < MIN_CHUNK_SIZE:
        return has_important_words(user_turn + " " + agent_turn)
    return True


def merge_short_turns(
    pairs: list[ConversationPair],
) -> list[TurnChunk]:
    """短いuser_turnが連続する場合、U+A+U+A交互のまま統合する。

    user_turn: 最初のユーザー発言（表示用）
    agent_turn: 最後のagent応答（表示用）
    content: U+A+U+A交互の全文（embedding・FTS用）
    """
    chunks: list[TurnChunk] = []
    i = 0

    while i < len(pairs):
        p = pairs[i]
        if not _is_worth_saving(p.user_turn, p.agent_turn):
            i += 1
            continue

        if len(p.user_turn) <= SHORT_USER_THRESHOLD:
            group = [p]
            j = i + 1
            while (
                j < len(pairs)
                and len(pairs[j].user_turn) <= SHORT_USER_THRESHOLD
                and _is_worth_saving(pairs[j].user_turn, pairs[j].agent_turn)
                and len(group) < MAX_MERGE_TURNS
            ):
                group.append(pairs[j])
                j += 1

            if len(group) > 1:
                first_user = group[0].user_turn
                last_agent = group[-1].agent_turn
                parts = [
                    _format_content(
                        g.user_turn, g.agent_turn[:AGENT_TRUNCATE_LEN]
                    )
                    for g in group
                ]
                chunks.append(
                    TurnChunk(
                        user_turn=first_user,
                        agent_turn=last_agent,
                        content="\n\n".join(parts),
                        timestamp=p.timestamp,
                    )
                )
                i = j
                continue

        chunks.append(
            TurnChunk(
                user_turn=p.user_turn,
                agent_turn=p.agent_turn,
                content=_format_content(p.user_turn, p.agent_turn),
                timestamp=p.timestamp,
            )
        )
        i += 1

    return chunks
