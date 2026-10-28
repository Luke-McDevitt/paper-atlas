@echo off
rem Paper Atlas APP: double-click to open your atlas in the browser, with buttons for adding papers, updating, discovering,
rem the zip and settings (tools\atlas_app.py). It runs with the kit's own Python (.venv); the first double-click on a new
rem computer runs setup.cmd once. No labels / goto / parentheses in echo text inside a block: the file ships with LF endings.
setlocal
set "KIT=%~dp0"
if not exist "%KIT%.venv\Scripts\pythonw.exe" (
  echo The Paper Atlas kit is not set up on this computer yet: running setup.cmd once - a few minutes ...
  call "%KIT%setup.cmd"
)
if not exist "%KIT%.venv\Scripts\pythonw.exe" (
  echo Setup did not finish - read the messages above, then double-click atlas-app.cmd again.
  pause
  exit /b 1
)
start "" "%KIT%.venv\Scripts\pythonw.exe" "%KIT%tools\atlas_app.py" %*
