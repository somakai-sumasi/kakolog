"""ハイブリッド検索: FTS5キーワード + vecベクトル → RRF統合 × 時間減衰 → MMR多様性"""

import re
from datetime import datetime, timezone

import numpy as np

from ..db import transaction
from ..embedder import cosine_similarity, embed_query
from ..models import SearchResult
from ..repository import (
    fetch_embeddings_by_ids,
    fetch_memories_by_ids,
    search_fts,
    search_vec,
)
from ..reranker import RerankCandidate, rerank
from . import touch_memories

RRF_K = 60
HALF_LIFE_DAYS = 30
RERANK_TOP = 10
MMR_LAMBDA = 0.7
SECONDS_PER_DAY = 86400


def time_decay(
    last_accessed_at: datetime, half_life_days: float = HALF_LIFE_DAYS
) -> float:
    now = datetime.now(timezone.utc)
    target = (
        last_accessed_at
        if last_accessed_at.tzinfo
        else last_accessed_at.replace(tzinfo=timezone.utc)
    )
    age_days = (now - target).total_seconds() / SECONDS_PER_DAY
    return 0.5 ** (age_days / half_life_days)


def rrf_fuse(
    keyword_ids: list[int],
    vector_ids: list[int],
    keyword_term_hits: dict[int, int] | None = None,
    total_terms: int = 1,
    k: int = RRF_K,
) -> dict[int, float]:
    scores: dict[int, float] = {}
    for rank, doc_id in enumerate(keyword_ids, start=1):
        scores.setdefault(doc_id, 0.0)
        base = 1.0 / (k + rank)
        if keyword_term_hits and total_terms > 1:
            hit_ratio = keyword_term_hits.get(doc_id, 1) / total_terms
            base *= 0.5 + 0.5 * hit_ratio
        scores[doc_id] += base
    for rank, doc_id in enumerate(vector_ids, start=1):
        scores.setdefault(doc_id, 0.0)
        scores[doc_id] += 1.0 / (k + rank)
    return scores


def _split_terms(query: str) -> list[str]:
    """クエリを単語分割。3文字未満はtrigramで検索できないので除外。"""
    terms = re.split(r"[\s、。,.!?　]+", query)
    return [t for t in terms if len(t) >= 3]


def _build_search_terms(query: str) -> list[str]:
    """クエリからFTS5検索用のタームリストを構築する。"""
    terms = _split_terms(query)
    if not terms and len(query) >= 3:
        terms = [query]
    return terms


def mmr_select(
    results: list[SearchResult],
    embeddings: dict[int, np.ndarray],
    limit: int,
    lambda_param: float = MMR_LAMBDA,
) -> list[SearchResult]:
    """MMRで関連性と多様性のバランスを取りながら結果を選択"""
    if len(results) <= 1:
        return results[:limit]

    candidates = list(results)
    selected: list[SearchResult] = []
    selected_vecs: list[np.ndarray] = []

    max_score = max(r.score for r in candidates)
    min_score = min(r.score for r in candidates)
    score_range = max_score - min_score if max_score > min_score else 1.0

    while candidates and len(selected) < limit:
        best_idx = -1
        best_mmr = -float("inf")

        for i, cand in enumerate(candidates):
            relevance = (cand.score - min_score) / score_range

            if not selected_vecs or cand.id not in embeddings:
                max_sim = 0.0
            else:
                cand_vec = embeddings[cand.id]
                max_sim = max(cosine_similarity(cand_vec, sv) for sv in selected_vecs)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i

        chosen = candidates.pop(best_idx)
        selected.append(chosen)
        if chosen.id in embeddings:
            selected_vecs.append(embeddings[chosen.id])

    return selected


def _deduplicate(results: list[SearchResult]) -> list[SearchResult]:
    """contentの重複を排除"""
    seen: set[str] = set()
    deduped = []
    for r in results:
        if r.memory.content not in seen:
            seen.add(r.memory.content)
            deduped.append(r)
    return deduped


def search(
    query: str,
    limit: int = 10,
    project_path: str | None = None,
    use_rerank: bool = False,
    use_mmr: bool = False,
) -> list[SearchResult]:
    query_embedding = embed_query(query)
    terms = _build_search_terms(query)

    keyword_ids, keyword_term_hits = search_fts(terms) if terms else ([], {})
    vector_ids = search_vec(query_embedding)

    rrf_scores = rrf_fuse(keyword_ids, vector_ids, keyword_term_hits, len(terms))
    if not rrf_scores:
        return []

    all_ids = list(rrf_scores.keys())
    memories = fetch_memories_by_ids(all_ids, project_path)

    results = [
        SearchResult.from_memory(
            m,
            score=rrf_scores.get(m.id, 0.0) * time_decay(m.last_accessed_at),
        )
        for m in memories
    ]

    results.sort(key=lambda r: r.score, reverse=True)
    deduped = _deduplicate(results)

    if use_rerank:
        candidates = [
            RerankCandidate(text=r.memory.content, source=r)
            for r in deduped[:RERANK_TOP]
        ]
        reranked = rerank(query, candidates, top_k=limit)
        deduped = [c.source.with_score(c.rerank_score) for c in reranked]

    if use_mmr:
        mmr_ids = [r.id for r in deduped[:RERANK_TOP]]
        embeddings = fetch_embeddings_by_ids(mmr_ids)
        final = mmr_select(deduped[:RERANK_TOP], embeddings, limit)
    else:
        final = deduped[:limit]

    hit_memories = [m for m in memories if m.id in {r.id for r in final}]
    with transaction():
        touch_memories(hit_memories)

    return final
