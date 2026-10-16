@echo off
rem Paper Atlas kit setup: a private Python in .venv with the core packages; add "gpu" and / or "ocr" for the extras.
rem gpu = tools\atlas_setup.py: PyTorch's CUDA build for this machine's NVIDIA driver (the CPU build without one), then
rem requirements-gpu.txt held to that torch, then a check that a CUDA operation runs. No labels / goto in this file:
rem it ships with LF line endings, where cmd can miss a label; a failed part sets FAILED instead.
setlocal
set "KIT=%~dp0"
if not exist "%KIT%.venv\Scripts\python.exe" python -m venv "%KIT%.venv"
if not exist "%KIT%.venv\Scripts\python.exe" echo Python 3.11 or newer is needed on PATH (python.org) & exit /b 1
set "PY=%KIT%.venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r "%KIT%requirements.txt" || exit /b 1
set "FAILED="
for %%E in (%*) do (
  if /I "%%E"=="gpu" "%PY%" "%KIT%tools\atlas_setup.py" gpu "%KIT%requirements-gpu.txt" || set "FAILED=%%E"
  if /I "%%E"=="ocr" "%PY%" -m pip install -r "%KIT%requirements-ocr.txt" || set "FAILED=%%E"
)
if defined FAILED (
  echo setup: the "%FAILED%" part FAILED -- see the messages above
  exit /b 1
)
echo Done. Check it with:  atlas selftest
