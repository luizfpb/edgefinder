@echo off
rem Atualiza os artefatos do dashboard publicado: roda a analise do dia e
rem commita/pusha data\reports. Rode depois de collect-odds (ou agende).
cd /d "%~dp0.."
.venv\Scripts\edgefinder.exe collect-odds
.venv\Scripts\edgefinder.exe analise
git add data\reports
git commit -m "chore: refresh dashboard artifacts"
git push
