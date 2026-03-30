"""テキストクリーニングとトリビアルフィルタリング。"""

import re

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

_NOISE_SELF_CLOSING_RE = re.compile(r"<(?:task-notification|system-reminder)[^>]*/>")

_NOISE_LINE_PREFIXES = [
    "Full transcript available at:",
    "Read the output file to retrieve",
    "Tool loaded.",
]

_TRIVIAL_INPUTS = frozenset(
    [
        "y",
        "yes",
        "ok",
        "はい",
        "うん",
        "続けて",
        "続き",
        "お願い",
        "いいよ",
        "それで",
        "進めて",
        "やって",
        "頼む",
    ]
)

_EMPTY_ANSWERS = frozenset(
    [
        "No response requested.",
        "No response requested",
    ]
)


def clean_text(text: str) -> str:
    text = _NOISE_TAG_RE.sub("", text)
    text = _NOISE_SELF_CLOSING_RE.sub("", text)
    lines = text.split("\n")
    lines = [
        line
        for line in lines
        if not any(line.strip().startswith(p) for p in _NOISE_LINE_PREFIXES)
    ]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_trivial(question: str) -> bool:
    return question.strip().lower() in _TRIVIAL_INPUTS


def is_empty_answer(answer: str) -> bool:
    return answer in _EMPTY_ANSWERS
