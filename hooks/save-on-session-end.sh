#!/bin/bash
# SessionEnd hook: stdinからJSONを読み、kakologに保存する
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INPUT=$(cat)
echo "$INPUT" | "$SCRIPT_DIR/.venv/bin/python" -c "
import json, sys
sys.path.insert(0, '$SCRIPT_DIR/src')
from kakolog.service import save_session
data = json.load(sys.stdin)
count = save_session(data['session_id'], data['transcript_path'], data.get('cwd'))
print(f'kakolog: saved {count} memories', file=sys.stderr)
" 2>/dev/null || true
