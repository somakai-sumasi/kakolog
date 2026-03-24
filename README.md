# kakolog

Claude Codeに長期記憶を持たせるエンジン。セッション終了時に会話を自動保存し、ハイブリッド検索+リランキングで過去の記憶を呼び出す。

参考: [Claude Codeに長期記憶を持たせたら壁打ちの質が変わった](https://zenn.dev/noprogllama/articles/7c24b2c2410213)

## アーキテクチャ

### 全体構成

```
                    SessionEnd hook
Claude Code ──────────────────────────> save-on-session-end.sh
                                              │
                                         curl (即座に返る)
                                              │
                                              v
                                    ┌─────────────────────┐
                                    │  kakolog-server   │
                                    │     :7377            │
                                    │                     │
                                    │  ┌───────────────┐  │
                                    │  │ Ruri v3-30m   │  │  埋め込みモデル (常駐)
                                    │  └───────────────┘  │
                                    │  ┌───────────────┐  │
                                    │  │ Reranker      │  │  japanese-reranker-tiny-v2 (常駐)
                                    │  │ (ONNX int8)   │  │
                                    │  └───────────────┘  │
                                    │  ┌───────────────┐  │
                                    │  │ SQLite        │  │  memories + FTS5 + sqlite-vec
                                    │  └───────────────┘  │
                                    └─────────────────────┘
```

常駐HTTPサーバー方式を採用。モデルをメモリに保持し、SessionEndフックからは軽量なHTTPリクエストだけ送る。

- **保存 (`POST /save`)**: 202 Accepted を即返却し、バックグラウンドでチャンク分割→ベクトル化→DB保存
- **検索 (`POST /search`)**: FTS5 + ベクトル検索 → RRF統合 → リランキング → 結果を返す

### なぜ常駐サーバー?

当初はSessionEndフックでPythonを直接起動していたが、PyTorch + モデルロードで毎回35秒かかった。常駐化でモデルロードは起動時1回のみとなり、保存応答は0.01秒に改善。

## 検索パイプライン

3段階のパイプラインで検索精度を段階的に高める。

```
クエリ
  │
  ├── FTS5 (Trigramキーワード検索)
  │     └── クエリを単語分割して個別検索
  │         ヒットターム数でランキング
  │
  ├── sqlite-vec (ベクトル検索)
  │     └── Ruri v3-30m で埋め込み
  │         コサイン類似度で上位50件
  │
  v
RRF (Reciprocal Rank Fusion)
  │   └── 両検索の順位を統合 (k=60)
  │       ターム一致率でFTS5側をブースト
  │       時間減衰 (30日半減期) を乗算
  │
  v
Reranking (Cross-Encoder)
  │   └── japanese-reranker-tiny-v2 (ONNX int8)
  │       RRF上位10件をクエリと文書のペアで精査
  │
  v
結果 (スコア付き上位N件)
```

### なぜ3段階?

| 段階 | 方式 | 速度 | 精度 |
|------|------|------|------|
| FTS5 | キーワード一致 | ~1ms | 低 (広く拾う) |
| ベクトル検索 | 意味的類似度 | ~5ms | 中 (Bi-Encoder) |
| リランキング | クエリ-文書ペア評価 | ~400ms | 高 (Cross-Encoder) |

FTS5とベクトル検索はそれぞれ異なる強みを持つ。FTS5は固有名詞やエラーコードの完全一致に強く、ベクトル検索は意味的な類似性を捉える。RRFで統合した後、Cross-Encoderリランカーがクエリと文書を直接比較して最終スコアを決定する。

### FTS5 Trigramの選択理由

日本語のキーワード検索にはMeCab等の形態素解析が必要になるが、FTS5のTrigramトークナイザ（3文字ずつの断片に分割）を使えば外部辞書なしで部分文字列検索が可能。プロジェクト固有のクラス名やAPI名の検索に強い。

## チャンク分割の工夫

Claude Codeの会話履歴(JSONL)からQ&Aペアを抽出する際、複数のフィルタリングを適用して検索品質を高めている。

### 1. assistant回答の結合

Claude Codeは1つの質問に対して複数回応答する（ツール実行を挟む）。

```
user:      "503エラーを無視するようにしたい"
assistant: "まず現在の実装を確認します。"     ← 前置き
           [tool_use: Grep, Read ...]
assistant: "ProductServiceでtry-catchを..."  ← 本回答
```

最初の「確認します」だけでなく、ツール実行後の本回答まで結合してチャンクにする。

### 2. MeCab wcostによる重要語判定

短いチャンク (Q+A合計50文字未満) は情報量が少ないことが多いが、プロジェクト固有の用語を含む場合は残す価値がある。

MeCabの単語コスト(wcost)で「珍しさ」を判定:

```
wcost ~4,000  : コミット、設定、テスト     → 一般語、スキップ
wcost ~9,000  : リファクタリング、デプロイ   → 技術専門語、採用
wcost ~13,000 : プロジェクト固有名詞       → 未知語/固有名詞、採用
```

閾値 `WCOST_THRESHOLD = 6000` 以上の名詞が1つでもあれば採用。

### 3. ノイズ除去

- XMLタグ (`<task-notification>`, `<system-reminder>` 等)
- 定型応答 (`No response requested`)
- トリビアル入力 (`y`, `ok`, `続けて` 等)
- 重複チャンクの排除 (同一Q&Aは1件のみ保存)

これらのフィルタにより、元の会話ペア3,611件から1,586件（56%削減）に圧縮。

## 技術スタック

| 要素 | 選定 | 備考 |
|------|------|------|
| 埋め込みモデル | [Ruri v3-30m](https://huggingface.co/cl-nagoya/ruri-v3-310m) (256次元) | 日本語特化。37Mパラメータで OpenAI text-embedding-3-large を上回る性能 |
| リランカー | [japanese-reranker-tiny-v2](https://huggingface.co/hotchpotch/japanese-reranker-tiny-v2) (ONNX int8) | Ruri v3ベースの日本語Cross-Encoder |
| DB | SQLite (FTS5 trigram + sqlite-vec) | 単一ファイル、外部サービス不要 |
| 形態素解析 | MeCab (IPAdic) | チャンクフィルタリング時の重要語判定 |
| Python | 3.11.11 | SQLite拡張対応ビルドが必要 |
| パッケージ管理 | uv | |

## セットアップ

### 前提条件

- macOS (Apple Silicon)
- Homebrew
- pyenv

### インストール

```bash
# MeCab
brew install mecab mecab-ipadic

# Python 3.11.11 (SQLite拡張対応ビルド)
PYTHON_CONFIGURE_OPTS="--enable-loadable-sqlite-extensions" \
LDFLAGS="-L/opt/homebrew/opt/sqlite/lib" \
CPPFLAGS="-I/opt/homebrew/opt/sqlite/include" \
pyenv install 3.11.11

# プロジェクト
git clone https://github.com/somakai-sumasi/kakolog.git
cd kakolog
pyenv local 3.11.11
uv sync
```

### Claude Code Hooks設定

`~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/kakolog/hooks/save-on-session-end.sh"
          }
        ]
      }
    ]
  }
}
```

## 使い方

```bash
# サーバー起動 (初回はモデルダウンロードが走る)
uv run kakolog-server

# 手動検索
uv run kakolog search "認証エラーの対応方法"

# 統計
uv run kakolog stats

# 過去セッションの一括インポート
uv run kakolog-import
```

## パフォーマンス

1,586 memories / 323 sessions での実測値:

| 観点 | 結果 |
|------|------|
| 検索レスポンス (リランキング込み) | ~400ms |
| 保存応答 (非同期受付) | ~10ms |
| バックグラウンド保存 (25チャンク) | ~30秒 |
| 全インポート (479セッション) | ~10分 |
| 常駐メモリ | ~627MB |
| DBサイズ | ~10MB |

## ファイル構成

```
src/kakolog/
├── server.py       # 常駐HTTPサーバー (ThreadingHTTPServer, port 7377)
├── service.py      # 保存ビジネスロジック (重複チェック+保存オーケストレーション)
├── save.py         # SessionEndフック用クライアント
├── search.py       # ハイブリッド検索 (FTS5 + vec → RRF → Reranking)
├── reranker.py     # japanese-reranker-tiny-v2 (ONNX int8, Cross-Encoder)
├── db.py           # SQLite接続 (Context Manager) ・CRUD
├── chunker.py      # JSONL→Q&Aチャンク分割 (ノイズフィルタ+MeCab重要語判定)
├── embedder.py     # Ruri v3-30m (CPU, 256次元)
├── recall.py       # 記憶呼び出し (未連携)
├── cli.py          # 手動検索・stats用CLI
└── bulk_import.py  # 過去セッション一括インポート
hooks/
└── save-on-session-end.sh  # SessionEndフック (curlでサーバーにPOST)
```

## 設計判断

### Ruri v3-30m (310mではなく30m)

310mモデルは精度が高い (JMTEB 77.24 vs 74.51) が、常駐で1.2GB消費+エンコードが遅い。30mは37Mパラメータで256次元ながら、OpenAI text-embedding-3-large (JMTEB 73.97) を上回る。コストパフォーマンスで30mを選択。

### CPU強制 (MPSではなく)

Apple SiliconのMPS (Metal Performance Shaders) を使うと、バッチエンコード時にメモリ不足でPCがクラッシュした（62GBバッファ確保エラー）。`device="cpu"`を強制して安定動作を優先。

### LLM不使用のチャンク分割

LLMで要約・分割すればチャンク品質は上がるが、セッション終了のたびにAPIコストが発生する。MeCabのwcost判定やルールベースのノイズフィルタで、LLMなしでも実用的な品質を実現。

### 長いチャンクは分割しない

10,000文字超のチャンクもあるが、ルールベースで意味単位の分割は難しく、文脈が切れるリスクがある。FTS5がフルテキストをインデックスするため、キーワード検索で補完される。

## License

MIT
