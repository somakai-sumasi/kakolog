"""japanese-reranker-tiny-v2 によるリランキング"""

import functools
import platform
from dataclasses import dataclass
from typing import Generic, TypeVar

from sentence_transformers import CrossEncoder

MODEL_NAME = "hotchpotch/japanese-reranker-tiny-v2"

T = TypeVar("T")


def _onnx_file_name() -> str:
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return "onnx/model_qint8_arm64.onnx"
    return "onnx/model_qint8_avx2.onnx"


@dataclass(frozen=True)
class RerankCandidate(Generic[T]):
    text: str
    source: T
    rerank_score: float = 0.0


@functools.lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(
        MODEL_NAME,
        backend="onnx",
        model_kwargs={
            "providers": ["CPUExecutionProvider"],
            "file_name": _onnx_file_name(),
        },
    )


def rerank(
    query: str, candidates: list[RerankCandidate[T]], top_k: int = 10
) -> list[RerankCandidate[T]]:
    """RRF後の候補をリランキングする。新しいリストを返す。"""
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [(query, c.text) for c in candidates]
    scores = reranker.predict(pairs)

    scored = [
        RerankCandidate(text=c.text, source=c.source, rerank_score=float(s))
        for c, s in zip(candidates, scores)
    ]
    scored.sort(key=lambda c: c.rerank_score, reverse=True)
    return scored[:top_k]
