param(
    [Parameter(Mandatory = $true)][string]$ConditionsPath,
    [string]$Port = "COM10",
    [ValidateSet('A','B','C')][string]$SessionRole = 'A',
    [string[]]$Gestures = @('rest','fist','splayed','wrist_up','wrist_down','pinch','point'),
    [string[]]$Postures = @('reference','elbow_flexed','elbow_extended','arm_abducted'),
    [ValidateRange(2,99)][int]$Repetitions = 3,
    [ValidateRange(1,99)][int]$MaxTrials = 99,
    [int]$Seed = 7,
    [switch]$PostureChunk,
    [switch]$Ready,
    [switch]$ValidateSetupOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
Set-Location -LiteralPath $projectRoot
$ConditionsPath = (Resolve-Path -LiteralPath $ConditionsPath).Path
$collectionRoot = Split-Path -Parent $ConditionsPath
$conditions = Get-Content -LiteralPath $ConditionsPath -Raw | ConvertFrom-Json
$sessionId = $conditions.session_id
$sessionDirectory = Join-Path $projectRoot ("recordings\" + $conditions.subject_id + "\" + $sessionId)
$statusPath = Join-Path $collectionRoot ($sessionId + ".status.json")
$logPath = Join-Path $collectionRoot ($sessionId + ".console.log")
$planPath = Join-Path $collectionRoot ($sessionId + ".plan.json")

function Set-CollectionStatus([string]$Stage, [string]$Message = "") {
    $statusJson = [ordered]@{stage=$Stage; message=$Message; launcher_pid=$PID; session_id=$sessionId;
        port=$Port; recording_directory=$sessionDirectory; updated_at=(Get-Date -Format o)} |
        ConvertTo-Json
    [System.IO.File]::WriteAllText($statusPath, $statusJson, (New-Object System.Text.UTF8Encoding($false)))
}

function Get-ReceiverPorts {
    $portScanCode = @'
import json
from serial.tools import list_ports
print(json.dumps([{"device": p.device, "description": p.description,
                   "usb": p.vid is not None or "USB" in p.hwid}
                  for p in list_ports.comports()]))
'@
    $scan = $portScanCode | & $pythonPath -
    if ($LASTEXITCODE -ne 0) { throw "Unable to list serial ports." }
    $decodedPorts = $scan | ConvertFrom-Json
    foreach ($item in $decodedPorts) { Write-Output $item }
}

$gestureHints = @{
    rest = "Relax your wrist and fingers."
    fist = "Close your hand into a comfortable fist."
    splayed = "Straighten and spread your fingers."
    wrist_up = "Extend your wrist: bring the back of your hand toward the back of your forearm; fingers relaxed."
    wrist_down = "Bend your palm toward the inner forearm; fingers relaxed."
    radial_deviation = "Bend the wrist toward the THUMB side; keep forearm orientation steady."
    ulnar_deviation = "Bend the wrist toward the LITTLE-FINGER side; keep forearm orientation steady."
    pinch = "Touch thumb and index fingertips; keep other fingers consistent."
    point = "Straighten index finger and curl the others; keep thumb position consistent."
}

try {
    if (Test-Path -LiteralPath $sessionDirectory) { throw "Recording directory already exists; prepare a new session ID." }
    $recordArgs = @('--protocol','dataset_v2','--source','serial','--session-role',$SessionRole,
        '--conditions-json',$ConditionsPath,'--gestures') + $Gestures + @('--postures') + $Postures +
        @('--repetitions',"$Repetitions",'--seed',"$Seed")
    if ($PostureChunk) { $recordArgs += '--posture-chunk' }
    $plan = & $pythonPath .\calibrate.py @recordArgs --plan
    if ($LASTEXITCODE -ne 0) { throw "Recording configuration failed validation." }
    [System.IO.File]::WriteAllText($planPath, ($plan -join "`n"), (New-Object System.Text.UTF8Encoding($false)))
    $parsedPlan = ($plan -join "`n") | ConvertFrom-Json
    $trialCount = $parsedPlan.planned_trial_order.Count
    if ($trialCount -gt $MaxTrials) { throw "This recording has $trialCount trials; maximum is $MaxTrials. Split into smaller chunks." }
    $estimatedSeconds = $trialCount * ($parsedPlan.preparation_seconds + $parsedPlan.gesture_seconds + $parsedPlan.rest_seconds) +
        $parsedPlan.baseline_seconds + 2 * $parsedPlan.independent_rest_seconds +
        $parsedPlan.posture_change_seconds * (1 + @($Postures | Where-Object { $_ -ne 'reference' }).Count + [int]($Postures[-1] -ne 'reference'))
    if ($ValidateSetupOnly) {
        $ports = @(Get-ReceiverPorts)
        Set-CollectionStatus "setup_validated"
        Write-Host "Validated $trialCount trials, $($Gestures.Count) gestures, $($Postures.Count) trial postures; hardware was not opened."
        foreach ($item in $ports) { Write-Host "$($item.device): $($item.description)" }
        return
    }

    $Host.UI.RawUI.WindowTitle = "Dataset v2 - $sessionId - waiting until ready"
    Clear-Host
    Write-Host "DATASET V2 - SESSION $SessionRole" -ForegroundColor Cyan
    Write-Host "Session: $sessionId | Placement: $($conditions.electrode_placement_id)"
    Write-Host "$trialCount trials | roughly $([math]::Ceiling($estimatedSeconds / 60)) minutes at nominal sampling rate | keep electrodes in place"
    Write-Host "Purpose: $($conditions.collection_purpose) | $Repetitions repetitions per gesture and trial posture"
    Write-Host ""
    foreach ($gesture in $Gestures) {
        Write-Host ("{0,-19} {1}" -f $gesture, $gestureHints[$gesture])
    }
    Write-Host ""
    Write-Host "TRIAL POSTURES: $($Postures -join ' -> ')"
    Write-Host "Begin and end relaxed in your original reference position."
    foreach ($posture in $Postures) {
        $description = $conditions.block_conditions.$posture.notes
        if ($description) { Write-Host "$posture`: $description" }
    }
    Write-Host "Keep the arm still during each hold. Prepare for 3 s, HOLD for 3 s, RELAX for 2 s."
    Write-Host "Stay relaxed during the initial baseline and both 60-second rest recordings."
    Write-Host "Connect the USB receiver, switch on the transmitter, and close other EMG/serial apps."
    Write-Host "Ctrl+C stops recording; interrupted sessions remain incomplete."
    Set-CollectionStatus "waiting_for_user"
    if (-not $Ready) {
        $null = Read-Host "When the electrodes are on and you are ready, press Enter to check the receiver"
    }

    Set-CollectionStatus "checking_receiver"
    while ($true) {
        $ports = @(Get-ReceiverPorts)
        $found = @($ports | Where-Object {$_.device -eq $Port})
        if ($found.Count -gt 0) { break }
        $usbPorts = @($ports | Where-Object {$_.usb})
        if ($usbPorts.Count -eq 1) {
            $Port = $usbPorts[0].device
            Write-Host "Receiver candidate: $Port ($($usbPorts[0].description))" -ForegroundColor Cyan
            break
        }
        foreach ($item in $ports) { Write-Host "$($item.device): $($item.description)" }
        Write-Host "Receiver port $Port is not detected yet." -ForegroundColor Yellow
        $selection = Read-Host "Connect the receiver and press Enter to retry, or type its COM port"
        if ($selection) {
            if ($selection -notmatch '^COM[0-9]+$') { Write-Host "Use a name such as COM10."; continue }
            $Port = $selection.ToUpperInvariant()
        }
    }

    # Reload so condition descriptions supplied in chat during setup can be used.
    $conditions = Get-Content -LiteralPath $ConditionsPath -Raw | ConvertFrom-Json
    if ($conditions.notes -match 'REFERENCE_DESCRIPTION_PENDING') {
        do { $description = Read-Host "Which arm, and what is your usual reference position? Describe it briefly" } until ($description)
        $conditions.notes = $conditions.notes.Replace('REFERENCE_DESCRIPTION_PENDING', $description)
    }
    Write-Host "Forearm orientation: neutral = handshake orientation (palm inward)."
    Write-Host "Pronated/supinated mean rotating toward palm-down/palm-up when the forearm is horizontal."
    $referenceNotes = ($conditions.notes -split 'Reference arm/position: ')[-1]
    Write-Host "Reference: $referenceNotes"
    do {
        $orientation = Read-Host "Reference forearm orientation [neutral]"
        if (-not $orientation) { $orientation = "neutral" }
        $orientation = $orientation.ToLowerInvariant()
    } until ($orientation -in @('neutral','pronated','supinated'))
    $conditions.forearm_orientation = $orientation
    [System.IO.File]::WriteAllText($ConditionsPath, ($conditions | ConvertTo-Json -Depth 8), (New-Object System.Text.UTF8Encoding($false)))

    Write-Host "Relax in reference posture for a 3-second receiver check." -ForegroundColor Cyan
    Set-CollectionStatus "checking_stream"
    while ($true) {
        & $pythonPath .\frame_probe.py --port $Port --seconds 3
        if ($LASTEXITCODE -eq 0) { break }
        Write-Host "Receiver check failed. Close any app holding the port and check receiver/transmitter power." -ForegroundColor Yellow
        $null = Read-Host "Press Enter to retry, or Ctrl+C to stop"
    }
    Set-CollectionStatus "ready_to_record"
    $null = Read-Host "Receiver check passed. Press Enter to start this $trialCount-trial recording"
    Set-CollectionStatus "recording"
    $Host.UI.RawUI.WindowTitle = "Dataset v2 - Session $SessionRole - recording"
    $trialNumber = 0
    & $pythonPath .\calibrate.py @recordArgs --port $Port |
        ForEach-Object {
            $line = [string]$_
            Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
            if ($line -match '^HOLD ([a-z_]+)') {
                $gesture = $Matches[1]
                $trialNumber++
                Write-Host ""
                Write-Host "[$trialNumber / $trialCount] $line" -ForegroundColor Green
                Write-Host $gestureHints[$gesture] -ForegroundColor Green
            } elseif ($line -match '^CHANGE POSTURE|^POSITION|^Independent|^Short normalization') {
                Write-Host ""
                Write-Host $line -ForegroundColor Cyan
            } else {
                Write-Host $line
            }
        }
    if ($LASTEXITCODE -ne 0) { throw "Acquisition exited with code $LASTEXITCODE. Keep the files for review; this session is incomplete." }
    Set-CollectionStatus "complete"
    Write-Host ""
    Write-Host "RECORDING SAVED AND STRUCTURALLY VALIDATED" -ForegroundColor Green
    Write-Host $sessionDirectory
    Write-Host "Keep the electrodes in place until you have reviewed the recording."
    $null = Read-Host "Press Enter to close"
} catch {
    Set-CollectionStatus "error" $_.Exception.Message
    Write-Host $_.Exception.Message -ForegroundColor Red
    if ($ValidateSetupOnly) { throw }
    $null = Read-Host "Press Enter to close"
}
