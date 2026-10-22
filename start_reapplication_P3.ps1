$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

Write-Host "Five-core P3 reapplication recording" -ForegroundColor Cyan
Write-Host "Keep the transmitter ON and follow every Prepare / HOLD / RELAX prompt." -ForegroundColor Yellow
Write-Host "Do not close this window until the validation result appears." -ForegroundColor Yellow
Write-Host ""

& .\.venv\Scripts\python.exe .\calibrate.py `
    --protocol dataset_v2 `
    --source serial `
    --port COM10 `
    --session-role B `
    --conditions-json .\data\reapplication_five_core_P3.example.json

if ($LASTEXITCODE -eq 0) {
    Write-Host "" 
    Write-Host "P3 RECORDING COMPLETED AND VALIDATED. Return to the ChatGPT conversation." -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "P3 RECORDING FAILED. Leave this window open and report the error." -ForegroundColor Red
}
