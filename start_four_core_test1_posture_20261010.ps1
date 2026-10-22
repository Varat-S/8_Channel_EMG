$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

Write-Host "FOUR-CORE TEST 1 OF 3: SAME-APPLICATION POSTURE CHALLENGE" -ForegroundColor Cyan
Write-Host "Keep the P3 electrodes and channel order unchanged." -ForegroundColor Yellow
Write-Host "This is held-out evaluation data; it will not retrain the model before scoring." -ForegroundColor Yellow
Write-Host "The three blocks are arm down, horizontal/palm down, and side/palm up." -ForegroundColor Yellow
Write-Host "REST means relax the hand and wrist while continuing to hold the requested arm position." -ForegroundColor Yellow
Write-Host "WRIST_UP means extend the wrist relative to the forearm, with fingers relaxed." -ForegroundColor Yellow
Write-Host ""

& .\record_dataset_v2.ps1 `
    -ConditionsPath .\data\four_core_P3_posture_challenge_20261010.json `
    -Port COM10 `
    -SessionRole B `
    -Gestures @('rest','fist','splayed','wrist_up') `
    -Postures @('elbow_extended','elbow_flexed','arm_abducted') `
    -Repetitions 3 `
    -Seed 31 `
    -PostureChunk `
    -MaxTrials 36
