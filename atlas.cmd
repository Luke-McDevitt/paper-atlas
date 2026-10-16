@echo off
rem Paper Atlas kit: run tools\atlas.py with the kit's own Python (.venv), else the python on PATH.
setlocal
set "KIT=%~dp0"
set "PY=python"
if exist "%KIT%.venv\Scripts\python.exe" set "PY=%KIT%.venv\Scripts\python.exe"
"%PY%" "%KIT%tools\atlas.py" %*
