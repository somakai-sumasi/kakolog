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
| Python | 3.11.11 | SQLite拡張対応ビルドが必要 |

## ファイル構成

```
src/kakolog/
├── mcp_server.py   # MCPサーバー (FastMCP, streamable-http, port 7377)
├── service.py      # 保存ビジネスロジック (重複チェック+保存オーケストレーション)
├── search.py       # ハイブリッド検索 (FTS5 + vec → RRF → MMR → Reranking)
├── reranker.py     # japanese-reranker-tiny-v2 (ONNX int8, Cross-Encoder)
├── repository.py   # メモリのデータ操作 (CRUD)
├── db.py           # SQLite接続 (Context Manager) ・スキーマ管理
├── chunker.py      # JSONL→Q&Aチャンク分割 (ノイズフィルタ+MeCab重要語判定)
├── embedder.py     # Ruri v3-30m (CPU, 256次元)
├── cli.py          # 手動検索・stats用CLI
├── bulk_import.py  # 過去セッション一括インポート
└── config.py       # ユーザー設定 (~/.kakolog/config.json, 除外パス管理)
hooks/
├── save-on-session-end.sh  # SessionEndフック (curlでMCPサーバーに転送)
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

## チャンク分割の方針

- assistant回答結合: ツール実行を挟む複数応答を結合
- ノイズ除去: XMLタグ、定型応答、トリビアル入力をフィルタ
- MeCab wcost判定: 50文字未満のチャンクは`WCOST_THRESHOLD=6000`以上の名詞があれば採用
- 重複排除: 同一Q&A+project_pathは1件のみ保存。再出現時はcreated_atを更新（時間減衰リセット）

## 検索パイプライン

FTS5(キーワード) + sqlite-vec(ベクトル) → RRF統合(k=60, ターム一致率ブースト) × 時間減衰(30日半減期) → MMR多様性選択(λ=0.7, デフォルト有効)。`use_rerank=True`でjapanese-reranker-tiny-v2によるリランキング(上位10件)を追加可能

## Gotchas

- pyenv 3.11.11必須。3.11.0は`enable_load_extension`非対応でsqlite-vecが動かない
- PyTorchはMPSを自動使用しようとする → `device="cpu"`で強制(embedder.py)
- ONNXモデルはプラットフォーム自動判定(ARM→qint8_arm64, x86→qint8_avx2)
- MeCabのmecabrcはchunker.pyが自動検出(`mecab-config` → フォールバックパス)
- SessionEndフックはcurlで `/hook/save` に転送。Python直接起動だとtransformersのimportで1秒以上かかりhookタイムアウト(SIGINT)に引っかかるため

## TODO

- [ ] チャンクのメタデータ抽出 (ファイルパス、コマンド等)
