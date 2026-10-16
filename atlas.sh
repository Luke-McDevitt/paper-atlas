#!/usr/bin/env sh
# Paper Atlas kit: run tools/atlas.py with the kit's own Python (.venv), else python3 on PATH.
KIT="$(cd "$(dirname "$0")" && pwd)"
if [ -x "$KIT/.venv/bin/python" ]; then PY="$KIT/.venv/bin/python"; else PY=python3; fi
exec "$PY" "$KIT/tools/atlas.py" "$@"
