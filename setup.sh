#!/usr/bin/env sh
# Paper Atlas kit setup: a private Python in .venv with the core packages; add "gpu" and / or "ocr" for the extras.
# gpu = tools/atlas_setup.py: PyTorch's CUDA build for this machine's NVIDIA driver (the CPU build without one; PyPI's
# on a Mac), then requirements-gpu.txt held to that torch, then a check that a CUDA operation runs.
set -e
KIT="$(cd "$(dirname "$0")" && pwd)"
[ -x "$KIT/.venv/bin/python" ] || python3 -m venv "$KIT/.venv"
PY="$KIT/.venv/bin/python"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$KIT/requirements.txt"
for extra in "$@"; do
  case "$extra" in
    gpu) "$PY" "$KIT/tools/atlas_setup.py" gpu "$KIT/requirements-gpu.txt" ;;
    ocr) "$PY" -m pip install -r "$KIT/requirements-ocr.txt" ;;
  esac
done
echo "Done. Check it with:  ./atlas.sh selftest"
