# scripts/cicd_pipeline.ps1
# Simulation d'un pipeline CI/CD local

$ErrorActionPreference = "Stop"
$PYTHON = "$PSScriptRoot/../../.venv/Scripts/python.exe"

Write-Host "--- Starting local CI/CD sequence ---" -ForegroundColor Cyan

# 1. Linting
Write-Host "[1/3] Linting code..." -ForegroundColor Yellow
& $PYTHON -m flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics

# 2. Smoke Test Training (1 epoch)
Write-Host "[2/3] Running training smoke test..." -ForegroundColor Yellow
& $PYTHON -u -m src.train_cv --epochs 1 --imgsz 64 --batch 4 --exp-name local_ci_test

# 3. Model Monitoring
Write-Host "[3/3] Running model monitoring comparison..." -ForegroundColor Yellow
& $PYTHON -u -m src.monitor_runs --experiment local_ci_test --threshold 0.1

Write-Host "--- CI/CD Sequence Completed Successfully ---" -ForegroundColor Green
