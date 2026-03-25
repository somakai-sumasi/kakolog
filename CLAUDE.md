# kakolog

Claude Code長期記憶エンジンの開発リポジトリ。

セッション終了時に会話をチャンク分割→ベクトル化→SQLiteに保存し、ハイブリッド検索+リランキングで過去の記憶を呼び出す。このファイルはkakolog自体の開発・改善時の参照用。

## アーキテクチャ

MCPサーバー（stdio）方式。Claude Codeがネイティブツールとして直接呼び出す。

```
[Claude Code] --MCP(stdio)--> [kakolog-mcp]
                                  ├── search  (検索: FTS5 + vec → RRF → Reranking)
                                  ├── save    (保存: チャンク分割 → 埋め込み → DB保存)
                                  └── stats   (統計)

[SessionEnd hook] --Python直接実行--> [service.save_session]
```

## 技術スタック

| 要素 | 選定 | 備考 |
|------|------|------|
| インターフェース | MCP (stdio) | `mcp[cli]` パッケージ、FastMCP |
| 埋め込みモデル | Ruri v3-30m (256次元) | CPU強制。30mでOpenAI text-embedding-3-largeを上回る |
| リランカー | japanese-reranker-tiny-v2 (ONNX int8) | RRF上位10件をcross-encoderで精査 |
| DB | SQLite (FTS5 trigram + sqlite-vec) | `~/.kakolog/memory.db` |
| 形態素解析 | MeCab (IPAdic) | チャンクフィルタリング時の重要語判定 |
| Python | 3.11.11 | SQLite拡張対応ビルドが必要 |

## ファイル構成

```
src/kakolog/
├── mcp_server.py   # MCPサーバー (FastMCP, stdio, search/save/statsツール)
├── service.py      # 保存ビジネスロジック (重複チェック+保存オーケストレーション)
├── search.py       # ハイブリッド検索 (FTS5 + vec → RRF → Reranking)
├── reranker.py     # japanese-reranker-tiny-v2 (ONNX int8, Cross-Encoder)
├── db.py           # SQLite接続 (Context Manager) ・CRUD
├── chunker.py      # JSONL→Q&Aチャンク分割 (ノイズフィルタ+MeCab重要語判定)
├── embedder.py     # Ruri v3-30m (CPU, 256次元)
├── cli.py          # 手動検索・stats用CLI
└── bulk_import.py  # 過去セッション一括インポート
hooks/
└── save-on-session-end.sh  # SessionEndフック (Python直接実行で保存)
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

```bash
claude mcp add -s user kakolog -- uv run --directory /path/to/kakolog kakolog-mcp
```

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

FTS5(キーワード) + sqlite-vec(ベクトル) → RRF統合(k=60, ターム一致率ブースト) × 時間減衰(30日半減期) → japanese-reranker-tiny-v2でリランキング(上位10件)

## Gotchas

- pyenv 3.11.11必須。3.11.0は`enable_load_extension`非対応でsqlite-vecが動かない
- PyTorchはMPSを自動使用しようとする → `device="cpu"`で強制(embedder.py)
- ONNXモデルはプラットフォーム自動判定(ARM→qint8_arm64, x86→qint8_avx2)
- MeCabのmecabrcはchunker.pyが自動検出(`mecab-config` → フォールバックパス)
- SessionEndフックはMCPではなくPython直接実行（stdioのMCPはフックから呼べないため）

## TODO

- [ ] チャンクのメタデータ抽出 (ファイルパス、コマンド等)
