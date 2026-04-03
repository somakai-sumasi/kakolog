"""Claude会話フォーマットからのQ&Aペア抽出とセッションメタ解析。"""

from dataclasses import dataclass
from datetime import datetime

from .models import ConversationPair
from .transcript import iter_jsonl


def extract_text(content: str | list) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


def _is_tool_result(content) -> bool:
    if isinstance(content, list):
        return all(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        )
    return False


def extract_conversations(messages: list[dict]) -> list[ConversationPair]:
    """user/assistantメッセージのペアを抽出する。

    同じuserの質問に対して複数のassistant応答がある場合
    （ツール実行を挟んで続く場合）、全テキストを結合する。
    """
    pairs = []
    current_user = None
    current_timestamp: datetime | None = None
    current_answer_parts: list[str] = []

    for entry in messages:
        msg = entry.get("message", {})
        role = msg.get("role")
        content = msg.get("content", "")

        if entry.get("isCompactSummary"):
            continue

        if role == "user" and _is_tool_result(content):
            continue

        text = extract_text(content).strip()

        if role == "user":
            if current_user and current_answer_parts:
                pairs.append(
                    ConversationPair(
                        user_turn=current_user,
                        agent_turn="\n\n".join(current_answer_parts),
                        timestamp=current_timestamp,
                    )
                )
            current_user = text if text else None
            raw_ts = entry.get("timestamp")
            current_timestamp = (
                datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                if raw_ts
                else None
            )
            current_answer_parts = []
        elif role == "assistant" and current_user:
            if text:
                current_answer_parts.append(text)

    if current_user and current_answer_parts:
        pairs.append(
            ConversationPair(
                user_turn=current_user,
                agent_turn="\n\n".join(current_answer_parts),
                timestamp=current_timestamp,
            )
        )

    return pairs


@dataclass(frozen=True)
class SessionMeta:
    cwd: str | None
    entrypoint: str | None
    first_timestamp: datetime | None


def read_session_meta(transcript_path: str) -> SessionMeta:
    """JSONLからcwd・entrypoint・first_timestampを抽出する。"""
    cwd = None
    entrypoint = None
    first_timestamp = None
    try:
        for entry in iter_jsonl(transcript_path):
            if first_timestamp is None and entry.get("timestamp"):
                first_timestamp = datetime.fromisoformat(
                    entry["timestamp"].replace("Z", "+00:00")
                )
            if cwd is None and entry.get("cwd"):
                cwd = entry["cwd"]
            if entrypoint is None and entry.get("entrypoint"):
                entrypoint = entry["entrypoint"]
            if (
                cwd is not None
                and entrypoint is not None
                and first_timestamp is not None
            ):
                break
    except (OSError, ValueError, KeyError):
        pass
    return SessionMeta(cwd=cwd, entrypoint=entrypoint, first_timestamp=first_timestamp)
