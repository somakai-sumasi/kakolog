"""ハイブリッド検索: FTS5キーワード + vecベクトル → RRF統合 × 時間減衰"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime

from sqlite_vec import serialize_float32

from .db import connection, init_db
from .embedder import embed_query
from .reranker import RerankCandidate, rerank

import re

RRF_K = 60
HALF_LIFE_DAYS = 30
TOP_K = 50


@dataclass
class SearchResult:
    id: int
    question: str
    answer: str
    score: float
    created_at: str
    project_path: str | None


def time_decay(created_at_str: str, half_life_days: float = HALF_LIFE_DAYS) -> float:
    created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
    now = datetime.now(created_at.tzinfo) if created_at.tzinfo else datetime.now()
    age_days = (now - created_at).total_seconds() / 86400
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
        # ターム一致率でブースト (全ターム一致=1.0, 1ターム一致=1/N)
        if keyword_term_hits and total_terms > 1:
            hit_ratio = keyword_term_hits.get(doc_id, 1) / total_terms
            base *= (0.5 + 0.5 * hit_ratio)  # 0.5〜1.0の範囲でスケール
        scores[doc_id] += base
    for rank, doc_id in enumerate(vector_ids, start=1):
        scores.setdefault(doc_id, 0.0)
        scores[doc_id] += 1.0 / (k + rank)
    return scores


def _split_terms(query: str) -> list[str]:
    """クエリを単語分割。3文字未満はtrigramで検索できないので除外。"""
    terms = re.split(r'[\s、。,.!?　]+', query)
    return [t for t in terms if len(t) >= 3]


def search_keyword(conn: sqlite3.Connection, query: str, limit: int = TOP_K) -> tuple[list[int], dict[int, int], int]:
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


def search_vector(conn: sqlite3.Connection, query_embedding: list[float], limit: int = TOP_K) -> list[int]:
    rows = conn.execute(
        "SELECT memory_id FROM vec_memories WHERE embedding MATCH ? AND k = ?",
        [serialize_float32(query_embedding), limit],
    ).fetchall()
    return [r[0] for r in rows]


RERANK_TOP = 10


def search(query: str, limit: int = 10, project_path: str | None = None, use_rerank: bool = False) -> list[SearchResult]:
    query_embedding = embed_query(query)

    with connection() as conn:
        init_db(conn)

        keyword_ids, keyword_term_hits, total_terms = search_keyword(conn, query)
        vector_ids = search_vector(conn, query_embedding)

        rrf_scores = rrf_fuse(keyword_ids, vector_ids, keyword_term_hits, total_terms)
        if not rrf_scores:
            return []

        all_ids = list(rrf_scores.keys())
        placeholders = ",".join("?" * len(all_ids))
        query_sql = f"SELECT id, question, answer, created_at, project_path FROM memories WHERE id IN ({placeholders})"
        params = all_ids
        if project_path:
            query_sql += " AND project_path = ?"
            params = all_ids + [project_path]

        rows = conn.execute(query_sql, params).fetchall()

    results = []
    for row in rows:
        mid = row[0]
        rrf_score = rrf_scores.get(mid, 0.0)
        decay = time_decay(row[3])
        results.append(SearchResult(
            id=mid, question=row[1], answer=row[2],
            score=rrf_score * decay, created_at=row[3], project_path=row[4],
        ))

    results.sort(key=lambda r: r.score, reverse=True)

    # 重複排除
    seen = set()
    deduped = []
    for r in results:
        key = (r.question, r.answer)
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    if not use_rerank:
        return deduped[:limit]

    # リランキング
    candidates = [
        RerankCandidate(text=f"{r.question}\n{r.answer}", source=r)
        for r in deduped[:RERANK_TOP]
    ]
    reranked = rerank(query, candidates, top_k=limit)

    return [
        SearchResult(
            id=c.source.id, question=c.source.question, answer=c.source.answer,
            score=c.rerank_score, created_at=c.source.created_at, project_path=c.source.project_path,
        )
        for c in reranked
    ]
