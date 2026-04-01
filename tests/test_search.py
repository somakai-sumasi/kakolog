from datetime import datetime, timedelta

import numpy as np

from kakolog.db import EMBEDDING_DIM
from kakolog.search import (
    SearchResult,
    mmr_select,
    rrf_fuse,
    time_decay,
)


class TestTimeDecay:
    def test_now_returns_one(self):
        now = datetime.now().isoformat()
        assert abs(time_decay(now) - 1.0) < 0.01

    def test_half_life(self):
        t = (datetime.now() - timedelta(days=30)).isoformat()
        assert abs(time_decay(t) - 0.5) < 0.05

    def test_old_date_decays(self):
        t = (datetime.now() - timedelta(days=90)).isoformat()
        assert time_decay(t) < 0.2


class TestRrfFuse:
    def test_single_source(self):
        scores = rrf_fuse([1, 2, 3], [])
        assert 1 in scores
        assert scores[1] > scores[2] > scores[3]

    def test_both_sources(self):
        scores = rrf_fuse([1, 2], [2, 3])
        assert scores[2] > scores[1]
        assert scores[2] > scores[3]

    def test_empty(self):
        assert rrf_fuse([], []) == {}

    def test_term_boost(self):
        # id=1は2ターム中2ヒット、id=2は2ターム中1ヒット
        hits = {1: 2, 2: 1}
        scores = rrf_fuse([1, 2], [], keyword_term_hits=hits, total_terms=2)
        # 全タームヒットのid=1が部分ヒットのid=2よりスコアが高い
        assert scores[1] > scores[2]


class TestMmrSelect:
    def _make_result(self, id: int, score: float) -> SearchResult:
        return SearchResult(
            id=id,
            user_turn=f"U{id}",
            agent_turn=f"A{id}",
            content=f"U: U{id}\nA: A{id}",
            score=score,
            created_at=datetime.now().isoformat(),
            last_accessed_at=datetime.now().isoformat(),
            project_path=None,
        )

    def _make_embeddings(self, ids: list[int]) -> dict[int, np.ndarray]:
        embs = {}
        for i in ids:
            vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            vec /= np.linalg.norm(vec)
            embs[i] = vec
        return embs

    def test_single_result(self):
        r = self._make_result(1, 1.0)
        selected = mmr_select([r], {}, limit=5)
        assert len(selected) == 1

    def test_respects_limit(self):
        results = [self._make_result(i, 1.0 - i * 0.1) for i in range(10)]
        embs = self._make_embeddings(list(range(10)))
        selected = mmr_select(results, embs, limit=3)
        assert len(selected) == 3

    def test_diverse_selection(self):
        results = [self._make_result(i, 1.0 - i * 0.01) for i in range(5)]
        # 同一ベクトル = 類似度が高い
        same_vec = np.ones(EMBEDDING_DIM, dtype=np.float32)
        same_vec /= np.linalg.norm(same_vec)
        embs = {i: same_vec.copy() for i in range(5)}
        # 1つだけ異なるベクトル
        diff_vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        diff_vec /= np.linalg.norm(diff_vec)
        embs[4] = diff_vec

        selected = mmr_select(results, embs, limit=3, lambda_param=0.3)
        selected_ids = {r.id for r in selected}
        # 多様性を重視(lambda=0.3)なので、異なるベクトルのid=4が選ばれやすい
        assert 4 in selected_ids
