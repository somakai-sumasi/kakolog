# Claude Code 長期記憶システム 調査レポート

参考記事: https://zenn.dev/noprogllama/articles/7c24b2c2410213

---

## 概要

Claude Codeにセッション横断の長期記憶を持たせるシステム。
セッション終了時に会話をQ&Aチャンクに分割→埋め込みベクトル化→SQLiteに保存し、
次回セッション開始時にハイブリッド検索で関連記憶を呼び出す。

### 技術スタック

| 要素 | 技術 |
|------|------|
| 言語 | Python |
| DB | SQLite (FTS5 + sqlite-vec) |
| 埋め込みモデル | Ruri v3-310m (日本語特化) |
| 検索方式 | ハイブリッド (キーワード + ベクトル、RRFで統合) |
| 時間減衰 | 30日半減期の指数減衰 |
| セットアップ | `uv sync` のみ |
| 依存 | sentence-transformers, sqlite-vec |

---

## 1. 埋め込みモデル: Ruri v3-310m

### 基本情報

- **HuggingFace**: `cl-nagoya/ruri-v3-310m`
- **開発**: 名古屋大学
- **ライセンス**: Apache 2.0
- **ベース**: ModernBERT-Ja

### スペック

| 項目 | 値 |
|------|-----|
| パラメータ数 | 315M |
| 出力次元 | 768 |
| 最大トークン長 | 8,192 |
| 語彙サイズ | 100K |
| 類似度関数 | コサイン類似度 |

### v3シリーズ全サイズ

| モデル | パラメータ | 次元 | JMTEB平均 |
|--------|-----------|------|----------|
| ruri-v3-30m | 37M | 256 | 74.51 |
| ruri-v3-70m | 70M | 384 | - |
| ruri-v3-130m | 132M | 512 | 76.55 |
| **ruri-v3-310m** | **315M** | **768** | **77.24** |

### 他モデルとの比較

| モデル | パラメータ | JMTEB平均 |
|--------|-----------|----------|
| **Ruri v3-310m** | **315M** | **77.24** |
| PLaMo-Embedding-1B | 1.05B | 76.10 |
| Ruri-Large-v2 | 337M | 74.55 |
| OpenAI text-embedding-3-large | 非公開 | 73.97 |
| multilingual-e5-large | ~560M | ~73 |

→ パラメータ数が半分以下でmE5-largeを大幅に上回る。30mモデルでさえOpenAI超え。

### 使い方

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("cl-nagoya/ruri-v3-310m")

# 検索時はプレフィックスが必要
query_emb = model.encode("検索クエリ: セッション管理の方法")
doc_emb = model.encode("検索文書: セッション終了時にSQLiteに保存する")
```

**プレフィックス規則**:
| プレフィックス | 用途 |
|--------------|------|
| `""` (空) | STS等の一般的な意味エンコード |
| `"トピック: "` | 分類・クラスタリング |
| `"検索クエリ: "` | 検索のクエリ側 |
| `"検索文書: "` | 検索の文書側 |

### CPU環境での考慮

- 315Mパラメータはmultilingual-e5-large(~560M)の約半分→CPU推論で有利
- 語彙100Kでトークン数が少なくなり推論高速化
- さらに軽量にしたい場合は ruri-v3-30m / 70m でもmE5-large以上の性能

---

## 2. SQLite: sqlite-vec + FTS5

### インストール

```bash
pip install sqlite-vec
```

**macOS注意**: 標準Pythonの SQLite は拡張読み込み不可。Homebrew版Python推奨。

### Pythonからの接続

```python
import sqlite3
import sqlite_vec
from sqlite_vec import serialize_float32

db = sqlite3.connect("memory.db")
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)
```

### テーブル設計例

```sql
-- メインテーブル
CREATE TABLE memories(
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    question TEXT,
    answer TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- FTS5テーブル（キーワード検索用・Trigramトークナイザ）
CREATE VIRTUAL TABLE fts_memories USING fts5(
    question,
    answer,
    content=memories,
    content_rowid=id,
    tokenize='trigram'
);

-- ベクトルテーブル（意味検索用）
CREATE VIRTUAL TABLE vec_memories USING vec0(
    memory_id integer primary key,
    embedding float[768]
);
```

### ベクトル操作

```python
# 挿入
embedding = model.encode("検索文書: ...")
db.execute(
    "INSERT INTO vec_memories(memory_id, embedding) VALUES (?, ?)",
    [1, serialize_float32(embedding.tolist())]
)

# KNN検索
query_emb = model.encode("検索クエリ: ...")
results = db.execute("""
    SELECT memory_id, distance
    FROM vec_memories
    WHERE embedding MATCH ?
      AND k = 10
    ORDER BY distance
    LIMIT 10
""", [serialize_float32(query_emb.tolist())]).fetchall()
```

### FTS5 Trigram の特性

**利点**:
- 形態素解析不要（MeCab等の辞書不要）
- 日本語のスペースなしテキストに対応
- 部分一致検索が可能

**制限**:
- **3文字未満の検索ができない**（trigramは常に3文字単位）
- インデックスサイズがunicode61の数倍〜10倍

### sqlite-vec のパフォーマンス

- **方式**: ブルートフォース（総当たり）のみ（v0.1.7時点）
- **適正規模**: 数千〜数十万ベクトル（記憶システムなら十分）
- **ANN未実装**: DiskANN/HNSW等はv1.0前に実装予定

---

## 3. ハイブリッド検索: RRF (Reciprocal Rank Fusion)

### RRFの数式

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

- `k`: スムージング定数（デフォルト推奨 **60**）
- 各検索システムの順位のみを使うため、スコアの正規化が不要

### Python実装

```python
def reciprocal_rank_fusion(search_results_dict, k=60):
    """
    Args:
        search_results_dict: {"keyword": [doc_id, ...], "vector": [doc_id, ...]}
        k: スムージング定数
    Returns:
        [(doc_id, rrf_score), ...] スコア降順
    """
    rrf_scores = {}
    for method, ranked_docs in search_results_dict.items():
        for rank, doc_id in enumerate(ranked_docs, start=1):
            rrf_scores.setdefault(doc_id, 0.0)
            rrf_scores[doc_id] += 1.0 / (k + rank)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
```

### 重み付きRRF（密/疎の比率調整）

```python
def weighted_rrf(search_results_dict, weights, k=60):
    rrf_scores = {}
    for method, ranked_docs in search_results_dict.items():
        w = weights.get(method, 1.0)
        for rank, doc_id in enumerate(ranked_docs, start=1):
            rrf_scores.setdefault(doc_id, 0.0)
            rrf_scores[doc_id] += w * (1.0 / (k + rank))
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
```

### SQL内でのRRF統合

```sql
WITH vec_matches AS (
    SELECT memory_id, row_number() OVER () AS rank_number
    FROM vec_memories
    WHERE embedding MATCH :query_embedding AND k = 20
    ORDER BY distance
    LIMIT 20
),
fts_matches AS (
    SELECT rowid, row_number() OVER (ORDER BY rank) AS rank_number
    FROM fts_memories
    WHERE fts_memories MATCH :search_query
    LIMIT 20
),
final AS (
    SELECT
        COALESCE(v.memory_id, f.rowid) AS id,
        COALESCE(1.0 / (60 + v.rank_number), 0.0)
        + COALESCE(1.0 / (60 + f.rank_number), 0.0)
        AS combined_score
    FROM vec_matches v
    FULL OUTER JOIN fts_matches f ON v.memory_id = f.rowid
)
SELECT id, combined_score FROM final ORDER BY combined_score DESC LIMIT 10;
```

---

## 4. 時間減衰 (Time Decay)

### 30日半減期の計算式

```python
import math
from datetime import datetime

def time_decay(created_at, half_life_days=30):
    age_days = (datetime.now() - created_at).total_seconds() / 86400
    return 0.5 ** (age_days / half_life_days)
    # 等価: math.exp(-math.log(2) / half_life_days * age_days)
```

### 減衰値の目安

| 経過日数 | 減衰係数 |
|---|---|
| 0日 | 1.000 (100%) |
| 15日 | 0.707 (70.7%) |
| 30日 | 0.500 (50%) |
| 60日 | 0.250 (25%) |
| 90日 | 0.125 (12.5%) |

### RRFとの組み合わせ

```python
final_score = rrf_score * time_decay(created_at)
```

RRF統合後に時間減衰を掛けるのが一般的。

---

## 5. Claude Codeセッションデータの取得

### 会話履歴の保存場所

```
~/.claude/projects/{project-path}/{sessionId}.jsonl
```

例: `/Users/ueki/.claude/projects/-Users-ueki-for_study/abc123.jsonl`

### JSONLファイルの構造

各行が独立したJSONオブジェクト:

```json
{
  "uuid": "a1b2c3d4-...",
  "parentUuid": "previous-uuid-...",
  "sessionId": "d8af951f-...",
  "timestamp": "2026-02-21T01:15:23.451Z",
  "type": "user | assistant | system | summary",
  "message": {
    "role": "user",
    "content": "Help me set up the project"
  }
}
```

- `parentUuid` によるリンクリスト構造で会話ツリーを再構築
- `isCompactSummary: true` のエントリはスキップすべき

### SessionEnd フックで自動保存

**settings.json に設定**:
```json
{
  "hooks": {
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python /path/to/save_memory.py"
          }
        ]
      }
    ]
  }
}
```

**フックが受け取るJSON (stdin)**:
```json
{
  "session_id": "abc123",
  "transcript_path": "/Users/ueki/.claude/projects/-Users-ueki-for_study/abc123.jsonl",
  "cwd": "/Users/ueki/for_study",
  "hook_event_name": "SessionEnd"
}
```

### 重要な制約

- **SessionEndのデフォルトタイムアウト: 1.5秒**
- 環境変数 `CLAUDE_CODE_SESSIONEND_HOOKS_TIMEOUT_MS=5000` で延長可能
- 重い処理（要約生成等）が必要な場合:
  - **方法A**: `Stop`フックで本処理、`SessionEnd`はフラグ更新のみ
  - **方法B**: SessionEndでバックグラウンドプロセスを起動し即exit
  - **方法C**: タイムアウトを十分大きく設定

### 会話パースの既存ツール

- [claude-conversation-extractor](https://github.com/ZeroSumQuant/claude-conversation-extractor) — PythonでJSONLパース
- [claude-JSONL-browser](https://github.com/withLinda/claude-JSONL-browser) — Web UIでMarkdown変換
- [claude-code-log](https://github.com/daaain/claude-code-log) — HTML出力、トークン追跡

---

## 6. 実装アーキテクチャ案

```
[Claude Code Session]
        |
        v (SessionEnd hook)
[save_memory.py]
   ├── 1. transcript_path からJSONLを読み込み
   ├── 2. user/assistant メッセージを抽出
   ├── 3. Q&A形式のチャンクに分割（LLM不使用）
   ├── 4. Ruri v3でベクトル化
   └── 5. SQLiteに保存 (memories + fts_memories + vec_memories)

[Claude Code Session Start]
        |
        v (SessionStart hook / CLAUDE.md)
[search_memory.py]
   ├── 1. ユーザーの入力からクエリ生成
   ├── 2. FTS5 Trigram でキーワード検索
   ├── 3. Ruri v3 でベクトル検索
   ├── 4. RRF で統合 × 時間減衰
   └── 5. top-k の記憶をコンテキストに注入
```

### 必要パッケージ

```toml
[project]
dependencies = [
    "sentence-transformers>=3.0",
    "sqlite-vec>=0.1.7",
]
```

---

## 参考リンク

### モデル・ライブラリ
- [cl-nagoya/ruri-v3-310m (HuggingFace)](https://huggingface.co/cl-nagoya/ruri-v3-310m)
- [sqlite-vec GitHub](https://github.com/asg017/sqlite-vec)
- [sqlite-vec Python docs](https://alexgarcia.xyz/sqlite-vec/python.html)
- [sqlite-vec ハイブリッド検索ブログ](https://alexgarcia.xyz/blog/2024/sqlite-vec-hybrid-search/index.html)

### RRF
- [Cormack et al. 原論文 (PDF)](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)
- [RRF Python実装](https://safjan.com/implementing-rank-fusion-in-python/)

### Claude Code
- [Hooks reference](https://code.claude.com/docs/en/hooks)
- [claude-mem (GitHub)](https://github.com/thedotmack/claude-mem)
- [claude-conversation-saver (GitHub)](https://github.com/sirkitree/claude-conversation-saver)
- [会話履歴の構造解説](https://kentgigger.com/posts/claude-code-conversation-history)
