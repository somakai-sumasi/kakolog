import functools
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import MeCab

from .cleaner import clean_text, is_empty_answer, is_trivial
from .extractor import extract_conversations
from .transcript import parse_jsonl


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
class QAChunk:
    question: str
    answer: str
    timestamp: str | None = None

    def to_text(self) -> str:
        return f"{self.question}\n{self.answer}"


MIN_CHUNK_SIZE = 50
WCOST_THRESHOLD = 6000  # これ以上のコストの名詞は「珍しい語」と判定


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


def _is_worth_saving(question: str, answer: str) -> bool:
    if not question or not answer:
        return False
    if is_empty_answer(answer):
        return False
    if is_trivial(question):
        return False
    if len(question) + len(answer) < MIN_CHUNK_SIZE:
        return has_important_words(question + " " + answer)
    return True


def chunk_session(transcript_path: str | Path) -> list[QAChunk]:
    messages = parse_jsonl(transcript_path)
    pairs = extract_conversations(messages)

    chunks = []
    for q, a, ts in pairs:
        q = clean_text(q)
        a = clean_text(a)
        if _is_worth_saving(q, a):
            chunks.append(QAChunk(question=q, answer=a, timestamp=ts))

    return chunks
