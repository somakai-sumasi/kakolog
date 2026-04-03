from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    import numpy

MODEL_NAME = "cl-nagoya/ruri-v3-30m"


@functools.lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME, device="cpu")


def embed_query(text: str) -> list[float]:
    model = get_model()
    vec = model.encode(f"検索クエリ: {text}", normalize_embeddings=True)
    return vec.tolist()


def embed_document(text: str) -> list[float]:
    model = get_model()
    vec = model.encode(f"検索文書: {text}", normalize_embeddings=True)
    return vec.tolist()


def cosine_similarity(
    a: list[float] | numpy.ndarray, b: list[float] | numpy.ndarray
) -> float:
    """2つのベクトルのコサイン類似度を計算する。
    bにndarrayを渡した場合はコピーせず直接使用する。"""
    import numpy as np

    va = np.array(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))


def embed_documents(texts: list[str]) -> list[list[float]]:
    model = get_model()
    vecs = model.encode(
        [f"検索文書: {t}" for t in texts], normalize_embeddings=True, batch_size=8
    )
    return vecs.tolist()
