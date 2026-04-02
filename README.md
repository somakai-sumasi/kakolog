# kakolog

Claude Codeに長期記憶を持たせるエンジン。セッション終了時に会話を自動保存し、ハイブリッド検索で過去の記憶を呼び出す。

参考: [Claude Codeに長期記憶を持たせたら壁打ちの質が変わった](https://zenn.dev/noprogllama/articles/7c24b2c2410213)

## アーキテクチャ

### 全体構成

```
[Claude Code] ──MCP (streamable-http)──> [kakolog-mcp :7377]
                                            ├── search       (検索)
                                            ├── save         (保存)
                                            ├── stats        (統計)
                                            ├── exclude_list/add/remove  (保存除外パス管理)
                                            └── POST /hook/save  (SessionEndフック専用)

[SessionEnd hook] ──curl──> [kakolog-mcp :7377/hook/save]
```

MCPサーバー（streamable-http）方式を採用。launchdで常駐起動し、モデルをメモリに保持。Claude Codeからはネイティブツールとして直接呼び出せる。

- **検索 (`search`)**: FTS5 + ベクトル検索 → RRF統合 → MMR多様性選択 → 結果を返す（`use_rerank=True`でcross-encoderリランキングも可能）
- **保存 (`save`)**: チャンク分割→ベクトル化→DB保存
- **除外パス管理 (`exclude_list/add/remove`)**: `~/.kakolog/config.json` で保存しないプロジェクトパスを管理
- **SessionEndフック**: バックグラウンドcurlで `/hook/save` に転送。`/new`（`reason:clear`）・通常終了・exit時に発火し会話を保存

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
MMR (Maximal Marginal Relevance)
  │   └── 関連性と多様性のバランスで選択 (λ=0.7)
  │       意味的に類似したチャンクの重複を削減
  v
結果 (スコア付き上位N件)

[オプション: use_rerank=True]
  │
  v
Reranking (Cross-Encoder)
  │   └── japanese-reranker-tiny-v2 (ONNX int8)
  │       RRF上位10件をクエリと文書のペアで精査
  v
MMR → 結果
```

### 検索モード

| モード | 速度 | 精度 | 用途 |
|--------|------|------|------|
| デフォルト (FTS5 + vec → RRF → MMR) | ~15ms | 中 | 通常の検索 |
| リランキング有効 (`use_rerank=True`) | ~400ms | 高 | 精度が必要な時 |

FTS5とベクトル検索はそれぞれ異なる強みを持つ。FTS5は固有名詞やエラーコードの完全一致に強く、ベクトル検索は意味的な類似性を捉える。MMRはデフォルト有効で、類似した結果の重複を削減して多様な情報を返す。リランキングはオプションで、`use_rerank=True`を指定するとCross-Encoderがクエリと文書を直接比較して最終スコアを決定する。

### FTS5 Trigramの選択理由

日本語のキーワード検索にはMeCab等の形態素解析が必要になるが、FTS5のTrigramトークナイザ（3文字ずつの断片に分割）を使えば外部辞書なしで部分文字列検索が可能。プロジェクト固有のクラス名やAPI名の検索に強い。

## チャンク分割の工夫

Claude Codeの会話履歴(JSONL)からターンペアを抽出する際、複数のフィルタリングと統合を適用して検索品質を高めている。

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
- 重複チャンクの排除 (同一content+project_pathは1件のみ保存)

### 4. 短ターン統合

user_turnが30文字以下のターンが連続する場合、最大3ターンをU+A交互のまま1チャンクに統合する。

```
統合前:
  チャンク1: U:「修正して」       A:「IN句を分割バッチ処理に修正しました」
  チャンク2: U:「テスト」         A:「全テストパスしました」
  チャンク3: U:「コミットして」   A:「コミットしました」

統合後（1チャンク）:
  user_turn: 「修正して」（表示用: 最初のuser）
  agent_turn: 「コミットしました」（表示用: 最後のagent）
  content: 「U: 修正して\nA: IN句を分割バッチ処理に...\n\nU: テスト\nA: 全テストパス...\n\nU: コミットして\nA: コミットしました」
```

会話の因果関係がembeddingに反映され、「修正して」単体では検索できなかった文脈が検索可能になる。

## 技術スタック

| 要素 | 選定 | 備考 |
|------|------|------|
| 埋め込みモデル | [Ruri v3-30m](https://huggingface.co/cl-nagoya/ruri-v3-310m) (256次元) | 日本語特化。37Mパラメータで OpenAI text-embedding-3-large を上回る性能 |
| リランカー | [japanese-reranker-tiny-v2](https://huggingface.co/hotchpotch/japanese-reranker-tiny-v2) (ONNX int8) | Ruri v3ベースの日本語Cross-Encoder |
| DB | SQLite (FTS5 trigram + sqlite-vec) | 単一ファイル、外部サービス不要 |
| 形態素解析 | MeCab (IPAdic) | チャンクフィルタリング時の重要語判定 |
| Linter/Formatter | [Ruff](https://docs.astral.sh/ruff/) | `ruff check` + `ruff format` |
| テスト | pytest + pytest-cov | ユニットテスト62件 |
| CI | GitHub Actions | PR・push時にlint + testを自動実行 |
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

### MCP登録

```bash
# グローバルに登録（全プロジェクトで利用可能）
claude mcp add -s user --transport http kakolog http://localhost:7377/mcp
```

### CLI

```bash
uv run kakolog search "認証エラーの対応方法"   # 手動検索
uv run kakolog stats                          # 統計
uv run kakolog-import                         # 過去セッション一括インポート
```

### 開発

```bash
uv run ruff check src/ tests/      # lint
uv run ruff format src/ tests/     # フォーマット
uv run pytest -v                   # テスト実行 (62件)
```

GitHub ActionsのCIでPR・push時にlint + testが自動実行される。

### launchd自動起動 (macOS)

`~/Library/LaunchAgents/com.kakolog.plist` を作成してサーバーを常駐させる。詳細は [hooks-setup.md](hooks-setup.md) を参照。

## パフォーマンス

1,586 memories / 323 sessions での実測値:

| 観点 | 結果 |
|------|------|
| 検索レスポンス | ~15ms (リランキング有効時: ~400ms) |
| 保存 (25チャンク) | ~30秒 |
| 全インポート (479セッション) | ~10分 |
| 常駐メモリ | ~627MB |
| DBサイズ | ~10MB |

## ファイル構成

```
src/kakolog/
├── mcp_server.py   # MCPサーバー (FastMCP, streamable-http, port 7377)
├── service.py      # 保存ビジネスロジック (重複スキップ+保存オーケストレーション)
├── search.py       # ハイブリッド検索 (FTS5 + vec → RRF → MMR → Reranking)
├── reranker.py     # japanese-reranker-tiny-v2 (ONNX int8, Cross-Encoder)
├── repository.py   # メモリのデータ操作 (挿入・検索・統計)
├── db.py           # SQLite接続 (Context Manager) ・スキーマ管理
├── chunker.py      # JSONL→TurnChunk分割 (ノイズフィルタ+MeCab重要語判定+短ターン統合)
├── embedder.py     # Ruri v3-30m (CPU, 256次元)
├── cli.py          # 手動検索・stats用CLI
├── bulk_import.py  # 過去セッション一括インポート
└── config.py       # ユーザー設定 (~/.kakolog/config.json, 除外パス管理)
tests/
├── conftest.py         # 共通フィクスチャ (in-memory DB, embedderモック等)
├── test_chunker.py     # チャンク分割テスト
├── test_config.py      # 設定I/O、パス除外判定テスト
├── test_db.py          # スキーマ作成、冪等性テスト
├── test_embedder.py    # 埋め込みモデルモックテスト
├── test_repository.py  # CRUD操作、統計テスト
├── test_search.py      # RRF、MMR、時間減衰テスト
└── test_service.py     # セッション保存オーケストレーションテスト
.github/workflows/
└── ci.yml              # GitHub Actions (lint + test)
hooks/
├── save-on-session-end.sh  # SessionEndフック (バックグラウンドcurlでMCPサーバーに転送)
└── start-server.sh         # launchd用サーバー起動スクリプト
```

## 設計判断

### MCP streamable-http + launchd常駐

MCPのstdioトランスポートはセッション中は常駐するが、プロセスが死んだ場合にセッション内で自動復帰しない。streamable-httpにしてlaunchdで常駐させることで、モデルロード（~30秒）は起動時1回のみとなり、プロセス死亡時もlaunchdが自動再起動する。

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
