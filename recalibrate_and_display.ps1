param(
    [string]$Port = "COM10",
    [int]$Trials = 3,
    [switch]$SkipReadyPrompt
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$sessionId = "electrode_reapply_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$sessionDirectory = Join-Path $projectRoot "recordings\S01\$sessionId"
$modelPath = Join-Path $projectRoot "models\${sessionId}_lda.joblib"

Set-Location -LiteralPath $projectRoot

Write-Host ""
Write-Host "FRESH EMG CALIBRATION" -ForegroundColor Cyan
Write-Host "Keep the original seated posture with the forearm perpendicular to the ground."
Write-Host "Session: $sessionId"
Write-Host "The hamster classifier will open automatically after recording and training."
Write-Host ""

if (-not $SkipReadyPrompt) {
    Read-Host "Turn the transmitter ON, close other EMG/serial programs, assume the trained seated posture, then press Enter"
}

& $pythonPath ".\calibrate.py" `
    --source serial `
    --port $Port `
    --subject S01 `
    --session $sessionId `
    --posture seated_forearm_perpendicular `
    --gestures fist peace index_thumb thumbs_up `
    --trials $Trials `
    --baseline-seconds 5 `
    --preparation-seconds 3 `
    --gesture-seconds 2 `
    --rest-seconds 1

if ($LASTEXITCODE -ne 0) {
    throw "Calibration exited with code $LASTEXITCODE; training was not started."
}

Write-Host ""
Write-Host "Training the fresh LDA model..." -ForegroundColor Cyan
& $pythonPath ".\train_model.py" `
    $sessionDirectory `
    --output $modelPath `
    --window-size 25 `
    --hop-size 5 `
    --low-hz 10 `
    --high-hz 55 `
    --guard-ms 300 `
    --spatial-reference common_average `
    --baseline-normalize

if ($LASTEXITCODE -ne 0) {
    throw "Training exited with code $LASTEXITCODE; the hamster display was not started."
}

Write-Host ""
Write-Host "Opening the hamster classifier with $modelPath" -ForegroundColor Green
& $pythonPath ".\live_display.py" --model $modelPath --port $Port

Write-Host ""
Write-Host "Hamster classifier closed. Model retained at: $modelPath"
