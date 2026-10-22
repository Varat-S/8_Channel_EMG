param([Parameter(Mandatory=$true)][string]$CollectionPath, [string]$Port="COM10", [switch]$ValidateSetupOnly)
$ErrorActionPreference = 'Stop'
$CollectionPath = (Resolve-Path -LiteralPath $CollectionPath).Path
$collectionRoot = Split-Path -Parent $CollectionPath
$collection = Get-Content -LiteralPath $CollectionPath -Raw | ConvertFrom-Json

function Invoke-Chunk($chunk, [switch]$ValidateOnly) {
    $arguments = @{
        ConditionsPath = (Join-Path $collectionRoot $chunk.conditions_file)
        Port = $Port; Gestures = [string[]]$collection.gestures
        Postures = @([string]$chunk.posture); Repetitions = [int]$chunk.repetitions
        Seed = [int]$chunk.seed; PostureChunk = $true; MaxTrials = 99
        ValidateSetupOnly = [bool]$ValidateOnly
    }
    & (Join-Path $PSScriptRoot 'record_dataset_v2.ps1') @arguments
}

if ($ValidateSetupOnly) {
    foreach ($chunk in $collection.chunks) { Invoke-Chunk $chunk -ValidateOnly }
    return
}
while ($true) {
    Write-Host ''
    Write-Host 'SHORT SEVEN-GESTURE RECORDINGS' -ForegroundColor Cyan
    Write-Host 'Keep the electrodes in their original positions. Each chunk saves separately.'
    Write-Host "Existing reference: $($collection.completed_reference_trial_count) completed trials."
    foreach ($chunk in $collection.chunks) {
        Write-Host "$($chunk.number): $($chunk.name) | $($chunk.trials) trials | $($chunk.purpose)"
        Write-Host "   $($chunk.position_instructions)"
    }
    Write-Host 'Take breaks between chunks. Chunk 4 is reserved for testing; keep it out of training.'
    $selection = Read-Host 'Choose a chunk number to set up, or Q to close'
    if ($selection -eq 'q') { break }
    $selected = @($collection.chunks | Where-Object { [string]$_.number -eq $selection })
    if ($selected.Count -ne 1) { continue }
    Invoke-Chunk $selected[0]
}
