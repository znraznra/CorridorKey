@echo off
cd /d "%~dp0"
py -3.12 -X utf8 scripts\setup_gpu.py --profile p600
if errorlevel 1 exit /b 1
echo Run Run_P600.bat for command-line help.
