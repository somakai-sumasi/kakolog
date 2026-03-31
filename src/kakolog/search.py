"""ハイブリッド検索: FTS5キーワード + vecベクトル → RRF統合 × 時間減衰 → MMR多様性"""

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sqlite_vec import serialize_float32

from .db import connection
from .embedder import embed_query
from .repository import fetch_embeddings_by_ids, fetch_memories_by_ids
from .reranker import RerankCandidate, rerank

RRF_K = 60
HALF_LIFE_DAYS = 30
TOP_K = 50
RERANK_TOP = 10
MMR_LAMBDA = 0.7
SECONDS_PER_DAY = 86400


@dataclass(frozen=True)
class SearchResult:
    id: int
    user_turn: str
    agent_turn: str
    score: float
    last_accessed_at: str
    project_path: str | None

    def with_score(self, score: float) -> "SearchResult":
        return SearchResult(
            id=self.id,
            user_turn=self.user_turn,
            agent_turn=self.agent_turn,
            score=score,
            last_accessed_at=self.last_accessed_at,
            project_path=self.project_path,
        )


def time_decay(
    last_accessed_at_str: str, half_life_days: float = HALF_LIFE_DAYS
) -> float:
    last_accessed_at = datetime.fromisoformat(
        last_accessed_at_str.replace("Z", "+00:00")
    )
    now = (
        datetime.now(last_accessed_at.tzinfo)
        if last_accessed_at.tzinfo
        else datetime.now()
    )
    age_days = (now - last_accessed_at).total_seconds() / SECONDS_PER_DAY
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


def search_keyword(
    conn: sqlite3.Connection, query: str, limit: int = TOP_K
) -> tuple[list[int], dict[int, int], int]:
    """戻り値: (ソート済みID, {id: ヒットターム数}, 総ターム数)"""
    terms = _split_terms(query)
    if not terms and len(query) >= 3:
        terms = [query]
    if not terms:
        return [], {}, 0

    doc_hits: dict[int, int] = {}
    for term in terms:
        try:
            rows = conn.execute(
                "SELECT rowid FROM fts_memories WHERE fts_memories MATCH ? LIMIT ?",
                [term, limit],
            ).fetchall()
            for r in rows:
                doc_hits[r[0]] = doc_hits.get(r[0], 0) + 1
        except sqlite3.OperationalError:
            continue

    sorted_ids = sorted(doc_hits.keys(), key=lambda x: doc_hits[x], reverse=True)
    return sorted_ids[:limit], doc_hits, len(terms)


def search_vector(
    conn: sqlite3.Connection, query_embedding: list[float], limit: int = TOP_K
) -> list[int]:
    rows = conn.execute(
        "SELECT memory_id FROM vec_memories WHERE embedding MATCH ? AND k = ?",
        [serialize_float32(query_embedding), limit],
    ).fetchall()
    return [r[0] for r in rows]


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
                sims = [
                    float(
                        np.dot(cand_vec, sv)
                        / (np.linalg.norm(cand_vec) * np.linalg.norm(sv) + 1e-9)
                    )
                    for sv in selected_vecs
                ]
                max_sim = max(sims)

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
    """ターンペアの重複を排除"""
    seen: set[tuple[str, str]] = set()
    deduped = []
    for r in results:
        key = (r.user_turn, r.agent_turn)
        if key not in seen:
            seen.add(key)
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

    with connection() as conn:
        keyword_ids, keyword_term_hits, total_terms = search_keyword(conn, query)
        vector_ids = search_vector(conn, query_embedding)

        rrf_scores = rrf_fuse(keyword_ids, vector_ids, keyword_term_hits, total_terms)
        if not rrf_scores:
            return []

        all_ids = list(rrf_scores.keys())
        memories = fetch_memories_by_ids(conn, all_ids, project_path)

        results = [
            SearchResult(
                id=m.id,
                user_turn=m.user_turn,
                agent_turn=m.agent_turn,
                score=rrf_scores.get(m.id, 0.0) * time_decay(m.last_accessed_at),
                last_accessed_at=m.last_accessed_at,
                project_path=m.project_path,
            )
            for m in memories
        ]

        results.sort(key=lambda r: r.score, reverse=True)
        deduped = _deduplicate(results)

        if use_rerank:
            candidates = [
                RerankCandidate(text=f"{r.user_turn}\n{r.agent_turn}", source=r)
                for r in deduped[:RERANK_TOP]
            ]
            reranked = rerank(query, candidates, top_k=limit)
            deduped = [c.source.with_score(c.rerank_score) for c in reranked]

        if use_mmr:
            mmr_ids = [r.id for r in deduped[:RERANK_TOP]]
            embeddings = fetch_embeddings_by_ids(conn, mmr_ids)
            return mmr_select(deduped[:RERANK_TOP], embeddings, limit)

    return deduped[:limit]
