$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

Write-Host "P3 wrist-down versus rest calibration" -ForegroundColor Cyan
Write-Host "Keep the existing electrodes and reference arm posture unchanged." -ForegroundColor Yellow
Write-Host "There is a 4-second Prepare countdown before each 3-second HOLD." -ForegroundColor Yellow
Write-Host "For REST, fully relax the hand and wrist. For WRIST_DOWN, make the same deliberate pose each time." -ForegroundColor Yellow
Write-Host ""

& .\.venv\Scripts\python.exe .\calibrate.py `
    --protocol dataset_v2 `
    --source serial `
    --port COM10 `
    --session-role B `
    --conditions-json .\data\reapplication_P3_wrist_down_focus_20261010.json `
    --gestures rest wrist_down `
    --repetitions 4 `
    --baseline-seconds 5 `
    --preparation-seconds 4 `
    --gesture-seconds 3 `
    --rest-seconds 2 `
    --independent-rest-seconds 5 `
    --posture-change-seconds 5

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "WRIST-DOWN CALIBRATION COMPLETED AND VALIDATED. Return to the ChatGPT conversation." -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "WRIST-DOWN CALIBRATION FAILED. Leave this window open and report the error." -ForegroundColor Red
}
