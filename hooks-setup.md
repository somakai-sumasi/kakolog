# Hooks設定ガイド

## 1. インストール

```bash
cd /path/to/kakolog
uv sync
```

## 2. Claude Code Hooks設定

`~/.claude/settings.json` に以下を追加:

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

## 3. サーバー自動起動 (macOS)

`~/Library/LaunchAgents/com.kakolog.plist` を作成:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.kakolog</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/kakolog/hooks/start-server.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/kakolog-server.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/kakolog-server.log</string>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.kakolog.plist
```

## 4. 動作確認

```bash
# ヘルスチェック
curl -s http://127.0.0.1:7377/health

# ステータス確認
uv run kakolog stats

# 手動検索
uv run kakolog search "検索したいキーワード"
```
