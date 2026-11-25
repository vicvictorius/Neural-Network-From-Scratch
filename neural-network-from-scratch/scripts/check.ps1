# "CI local" para Windows (PowerShell). Rode antes de cada commit.
# Uso: .\scripts\check.ps1            (tudo)
#      .\scripts\check.ps1 numpy      (só os testes da Fase 1)
param([string]$Escopo = "tudo")
$ErrorActionPreference = "Stop"
Write-Host "== Lint =="
ruff check .
if ($LASTEXITCODE -ne 0) { exit 1 }
Write-Host "== Testes =="
if ($Escopo -eq "numpy") {
    pytest tests/test_data.py tests/test_numpy_layers.py tests/test_numpy_training.py
} else {
    pytest --cov=src --cov-report=term-missing
}
if ($LASTEXITCODE -ne 0) { exit 1 }
Write-Host "Tudo certo."
