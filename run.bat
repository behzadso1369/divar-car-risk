@echo off
chcp 65001 >nul
cd /d %~dp0
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
    .venv\Scripts\python -m pip install -r requirements.txt
)
start "" http://127.0.0.1:8000
.venv\Scripts\python cli.py serve --port 8000
