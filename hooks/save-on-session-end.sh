#!/bin/bash
# SessionEnd hook: MCPサーバーのHTTPエンドポイントに転送する
INPUT=$(cat)
curl -s -X POST http://127.0.0.1:7377/hook/save \
  -H "Content-Type: application/json" \
  -d "$INPUT" > /dev/null &
