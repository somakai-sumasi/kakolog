"""Claude会話フォーマットからのQ&Aペア抽出。"""

from .models import ConversationPair


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
    current_timestamp: str | None = None
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
            current_timestamp = entry.get("timestamp")
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
