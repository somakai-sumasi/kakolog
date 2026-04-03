# kakolog

Claude Code長期記憶エンジンの開発リポジトリ。

セッション終了時に会話をチャンク分割→ベクトル化→SQLiteに保存し、ハイブリッド検索で過去の記憶を呼び出す。このファイルはkakolog自体の開発・改善時の参照用。

## アーキテクチャ

MCPサーバー（streamable-http）方式。launchdで常駐起動し、Claude CodeはHTTPで接続する。

```
[Claude Code] --MCP(http:7377)--> [kakolog-mcp (launchd常駐)]
                                      ├── search              (検索: FTS5 + vec → RRF → MMR → Reranking)
                                      ├── save                (保存: チャンク分割 → 埋め込み → DB保存)
                                      ├── stats               (統計)
                                      ├── exclude_list/add/remove  (保存除外パス管理)
                                      └── POST /hook/save     (SessionEndフック専用HTTP endpoint)

[SessionEnd hook] --curl--> [kakolog-mcp :7377/hook/save]
```

## 技術スタック

| 要素 | 選定 | 備考 |
|------|------|------|
| インターフェース | MCP (streamable-http) | `mcp[cli]` パッケージ、FastMCP、port 7377 |
| 埋め込みモデル | Ruri v3-30m (256次元) | CPU強制。30mでOpenAI text-embedding-3-largeを上回る |
| リランカー | japanese-reranker-tiny-v2 (ONNX int8) | RRF上位10件をcross-encoderで精査 |
| DB | SQLite (FTS5 trigram + sqlite-vec) | `~/.kakolog/memory.db` |
| 形態素解析 | MeCab (IPAdic) | チャンクフィルタリング時の重要語判定 |
| Linter/Formatter | Ruff | `ruff check` + `ruff format` |
| テスト | pytest + pytest-cov | 62件のユニットテスト |
| CI | GitHub Actions | lint + test を自動実行 |
| Python | 3.11.11 | SQLite拡張対応ビルドが必要 |

## ファイル構成

```
src/kakolog/
├── mcp_server.py   # MCPサーバー (FastMCP, streamable-http, port 7377)
├── service.py      # 保存ビジネスロジック (パイプライン調整+重複スキップ+保存)
├── search.py       # ハイブリッド検索アルゴリズム (RRF → MMR → Reranking)
├── models.py       # ドメインモデル (Memory, SearchResult, ConversationPair)
├── repository.py   # データ操作 (CRUD + FTS5検索 + ベクトル検索)
├── db.py           # SQLite接続 (contextvars自動管理) ・スキーマ・トランザクション
├── db_util.py      # DB↔Model変換ユーティリティ (from_row, columns_of)
├── reranker.py     # japanese-reranker-tiny-v2 (ONNX int8, Cross-Encoder)
├── chunker.py      # チャンク分割 (MeCab重要語判定+短ターン統合)
├── extractor.py    # Claude会話フォーマットからQ&Aペア抽出
├── cleaner.py      # テキストクリーニング・ノイズ除去
├── embedder.py     # Ruri v3-30m (CPU, 256次元)
├── transcript.py   # JSONLトランスクリプトI/O
├── cli.py          # 手動検索・stats用CLI
├── bulk_import.py  # 過去セッション一括インポート
└── config.py       # ユーザー設定 + 除外ポリシー
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

## セットアップ

README.md を参照。Python 3.11.11 (SQLite拡張対応ビルド) + MeCab が必要。

## コマンド

```bash
uv run kakolog search "クエリ"      # 手動検索
uv run kakolog stats               # 統計
uv run kakolog-import              # 過去セッション一括インポート (~10分)
rm ~/.kakolog/memory.db && uv run kakolog-import  # DB再構築

# 開発
uv run ruff check src/ tests/      # lint
uv run ruff format src/ tests/     # フォーマット
uv run pytest -v                   # テスト実行
uv run pytest -v --tb=short        # テスト実行 (短縮出力)
```

## MCP登録

launchdでサーバーを常駐させた上で、HTTPトランスポートで登録する:

```bash
# launchd登録 (初回のみ)
cp ~/Library/LaunchAgents/com.somakai-sumasi.kakolog.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.somakai-sumasi.kakolog.plist

# MCP登録
claude mcp add -s user --transport http kakolog http://127.0.0.1:7377/mcp
```

stdioで登録するとセッション起動時にポート競合して接続失敗する。

## Hooks設定

`~/.claude/settings.json`:
```json
"SessionEnd": [{
  "hooks": [{
    "type": "command",
    "command": "/path/to/kakolog/hooks/save-on-session-end.sh"
  }]
}]
```

`SessionEnd`は`/new`（`reason:clear`）・通常終了・exit時に発火し会話を保存する。curlはバックグラウンド実行（`&`）なのでClaudeの応答をブロックしない。

## チャンク分割・検索パイプライン

詳細はREADME.mdを参照。概要:

- **チャンク分割**: agentターン結合 → ノイズ除去 → MeCab wcost重要語判定(閾値6000) → 短ターン統合(user≤30字が連続→最大3ターンをU+A交互で1チャンクに) → 重複排除
- **検索**: FTS5(`content`) + sqlite-vec → RRF(k=60) × 時間減衰(30日半減期) → MMR(λ=0.7) → [オプション] Reranking

## データモデル

- ドメインモデル(`Memory`, `SearchResult`, `ConversationPair`)は `models.py` に集約。全て `frozen=True` の dataclass
- `Memory.created_at` / `last_accessed_at` は `datetime` 型。DBの `TIMESTAMP` カラムは `detect_types=PARSE_DECLTYPES` + カスタムコンバータで自動変換
- `db_util.from_row(row, ModelClass)` で汎用的にsqlite3.Row→dataclass変換。`columns_of(ModelClass)` でSELECT句を自動生成
- DB接続は `contextvars` で暗黙管理。`get_conn()` で自動取得、書き込み時のみ `with transaction():` を使用
- DBカラムは `user_turn`(表示用) / `agent_turn`(表示用) / `content`(embedding・FTS用全文) / `created_at` / `last_accessed_at`
- `content` のフォーマットは `U: {user}\nA: {agent}` で統一（`_format_content()` で生成）。統合チャンクは `\n\n` 区切りで複数ペアを連結

## Gotchas

- pyenv 3.11.11必須。3.11.0は`enable_load_extension`非対応でsqlite-vecが動かない
- PyTorchはMPSを自動使用しようとする → `device="cpu"`で強制(embedder.py)
- ONNXモデルはプラットフォーム自動判定(ARM→qint8_arm64, x86→qint8_avx2)
- MeCabのmecabrcはchunker.pyが自動検出(`mecab-config` → フォールバックパス)
- SessionEndフックはバックグラウンドcurl（`&`）で `/hook/save` に転送。Python直接起動だとtransformersのimportで1秒以上かかりhookタイムアウト(SIGINT)に引っかかるため

## MCPツールのdescription指針

- **1〜2文が標準**（公式SDK例: `"""Add two numbers"""` `"""Get weather for a city."""`）
- パラメータ説明はdescriptionではなくスキーマ（型ヒント・引数）に任せる
- LLMの行動に影響する場合（クエリの書き方など）は2文目に補足してよい
- searchツールの場合: 「具体的な語を含む自然言語クエリ」がFTS5+ベクトル両方に効いて最も精度が高い（キーワード列挙はFTSにしか効かず、スコアが約半分になる）

## TODO

- [ ] チャンクのメタデータ抽出 (ファイルパス、コマンド等)
