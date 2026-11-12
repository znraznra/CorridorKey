@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run Setup_P600.bat first.
    exit /b 1
)
if "%~1"=="" (
    ".venv\Scripts\python.exe" -X utf8 corridorkey_cli.py --device cuda --help
) else (
    ".venv\Scripts\python.exe" -X utf8 corridorkey_cli.py --device cuda %*
)
