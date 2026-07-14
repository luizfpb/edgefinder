@echo off
rem Fluxo diario completo (coleta, analise, paper bets, CLV) + publicacao dos
rem artefatos do dashboard. Agendavel no Agendador de Tarefas do Windows.
cd /d "%~dp0.."
.venv\Scripts\edgefinder.exe daily
git add data\reports
git commit -m "chore: refresh dashboard artifacts"
git push
