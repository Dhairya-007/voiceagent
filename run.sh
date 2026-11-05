#!/usr/bin/env bash
# Start the demo on http://localhost:8080
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || uv venv --python 3.11 .venv
uv pip install -q --python .venv -e ".[dev]"
exec .venv/bin/python -m uvicorn voiceagent.app:app --host 127.0.0.1 --port 8080 "$@"
