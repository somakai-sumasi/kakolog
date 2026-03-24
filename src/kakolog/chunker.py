import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import shutil
import subprocess

import MeCab


def _find_mecabrc() -> str | None:
    """mecabrcのパスを自動検出する。"""
    # 環境変数が設定済みならそれを使う
    if os.environ.get("MECABRC"):
        return os.environ["MECABRC"]
    # mecab-configから取得
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
    # よくあるパス
    for p in ["/opt/homebrew/etc/mecabrc", "/usr/local/etc/mecabrc", "/etc/mecabrc"]:
        if os.path.exists(p):
            return p
    return None


_mecabrc = _find_mecabrc()
if _mecabrc:
    os.environ.setdefault("MECABRC", _mecabrc)


@dataclass
class QAChunk:
    question: str
    answer: str


MIN_CHUNK_SIZE = 50
WCOST_THRESHOLD = 6000  # これ以上のコストの名詞は「珍しい語」と判定
_tagger: MeCab.Tagger | None = None


def _get_tagger() -> MeCab.Tagger:
    global _tagger
    if _tagger is None:
        _tagger = MeCab.Tagger()
    return _tagger


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


# XMLタグ形式のシステムノイズを除去するパターン
_NOISE_TAG_RE = re.compile(
    r"<(?:task-notification|system-reminder|local-command-caveat|command-name|"
    r"command-message|command-args|local-command-stdout|available-deferred-tools|"
    r"teammate-message)[^>]*>"
    r".*?"
    r"</(?:task-notification|system-reminder|local-command-caveat|command-name|"
    r"command-message|command-args|local-command-stdout|available-deferred-tools|"
    r"teammate-message)>",
    re.DOTALL,
)

# 自己完結タグ
_NOISE_SELF_CLOSING_RE = re.compile(
    r"<(?:task-notification|system-reminder)[^>]*/>"
)

# 意味のない短い入力
_TRIVIAL_INPUTS = frozenset([
    "y", "yes", "ok", "はい", "うん", "続けて", "続き", "お願い",
    "いいよ", "それで", "進めて", "やって", "頼む",
])


_NOISE_LINE_PREFIXES = [
    "Full transcript available at:",
    "Read the output file to retrieve",
    "Tool loaded.",
]

# 情報量ゼロの定型回答
_EMPTY_ANSWERS = frozenset([
    "No response requested.",
    "No response requested",
])


def clean_text(text: str) -> str:
    text = _NOISE_TAG_RE.sub("", text)
    text = _NOISE_SELF_CLOSING_RE.sub("", text)
    # ノイズ行を除去
    lines = text.split("\n")
    lines = [l for l in lines if not any(l.strip().startswith(p) for p in _NOISE_LINE_PREFIXES)]
    text = "\n".join(lines)
    # 連続空行を1つに
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_trivial(question: str) -> bool:
    return question.strip().lower() in _TRIVIAL_INPUTS


def parse_jsonl(transcript_path: str | Path) -> list[dict]:
    messages = []
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                messages.append(entry)
            except json.JSONDecodeError:
                continue
    return messages


def extract_text(content) -> str:
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
    """contentがtool_resultのみかどうか。"""
    if isinstance(content, list):
        return all(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in content
        )
    return False


def extract_conversations(messages: list[dict]) -> list[tuple[str, str]]:
    """user/assistantメッセージのペアを抽出する。

    同じuserの質問に対して複数のassistant応答がある場合
    （ツール実行を挟んで続く場合）、全テキストを結合する。
    """
    pairs = []
    current_user = None
    current_answer_parts: list[str] = []
    in_tool_loop = False  # ツール実行ループ中かどうか

    for entry in messages:
        msg = entry.get("message", {})
        role = msg.get("role")
        content = msg.get("content", "")

        if entry.get("isCompactSummary"):
            continue

        # tool_resultのuserメッセージはスキップ（ツールループの一部）
        if role == "user" and _is_tool_result(content):
            in_tool_loop = True
            continue

        text = extract_text(content).strip()

        if role == "user":
            # 新しいuserメッセージ → 前のペアを確定
            if current_user and current_answer_parts:
                pairs.append((current_user, "\n\n".join(current_answer_parts)))
            current_user = text if text else None
            current_answer_parts = []
            in_tool_loop = False
        elif role == "assistant" and current_user:
            if text:
                current_answer_parts.append(text)
            # tool_useがあればループフラグを立てる
            if isinstance(content, list):
                has_tool = any(
                    isinstance(b, dict) and b.get("type") == "tool_use"
                    for b in content
                )
                if has_tool:
                    in_tool_loop = True

    # 最後のペアを確定
    if current_user and current_answer_parts:
        pairs.append((current_user, "\n\n".join(current_answer_parts)))

    return pairs


def chunk_session(transcript_path: str | Path) -> list[QAChunk]:
    messages = parse_jsonl(transcript_path)
    pairs = extract_conversations(messages)

    chunks = []
    for q, a in pairs:
        q = clean_text(q)
        a = clean_text(a)

        # ノイズ除去後に空になったらスキップ
        if not q or not a:
            continue

        # 定型の空回答をスキップ
        if a in _EMPTY_ANSWERS:
            continue

        # 意味のない短い入力はスキップ
        if is_trivial(q):
            continue

        # 短いチャンクは重要語がある場合のみ採用
        if len(q) + len(a) < MIN_CHUNK_SIZE:
            if not has_important_words(q + " " + a):
                continue

        chunks.append(QAChunk(question=q, answer=a))

    return chunks
