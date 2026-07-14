@echo off
rem Atualizacao periodica do EdgeFinder: re-baixa CSVs da temporada corrente
rem (football-data.co.uk) e recarrega o banco a partir do cache. Pensado para
rem o Agendador de Tarefas do Windows (semanal). O aquecimento de FBref e
rem Understat de novas rodadas e mais lento e roda a parte (warm_*.py).
cd /d "%~dp0.."
.venv\Scripts\python.exe scripts\warm_fast.py
.venv\Scripts\python.exe scripts\load_db.py
