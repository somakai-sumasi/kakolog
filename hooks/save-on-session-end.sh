#!/bin/bash
# SessionEnd hook: stdinからJSONを読み、サーバーにPOSTする
INPUT=$(cat)
curl -s -X POST http://127.0.0.1:7377/save \
  -H "Content-Type: application/json" \
  -d "$INPUT" \
  --connect-timeout 1 \
  --max-time 2 \
  > /dev/null 2>&1 || true
