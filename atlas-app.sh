#!/usr/bin/env sh
# Paper Atlas APP: opens your atlas in the browser, with buttons for adding papers, updating, discovering, the zip and
# settings (tools/atlas_app.py). Runs with the kit's own Python (.venv); the first run on a new computer runs setup.sh.
# It stops by itself 15 minutes after its browser tab is closed (or press Ctrl+C here).
KIT="$(cd "$(dirname "$0")" && pwd)"
if [ ! -x "$KIT/.venv/bin/python" ]; then
  echo "The Paper Atlas kit is not set up on this computer yet: running setup.sh once (a few minutes) ..."
  sh "$KIT/setup.sh" || exit 1
fi
exec "$KIT/.venv/bin/python" "$KIT/tools/atlas_app.py" "$@"
