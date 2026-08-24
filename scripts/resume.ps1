<#
    Auto-resume watchdog for the pipeline.

    This machine bugchecks under sustained GPU load (see DECISIONS.md D-013), which kills
    the run without warning. Every stage checkpoints per cell, so a resume costs only the
    work since the last completed cell - but only if something actually restarts it. This
    script is that something.

    It is safe to run at any time and as often as you like:
      * exits immediately if the pipeline is already running (no double-launch);
      * exits immediately if results/PIPELINE_COMPLETE exists (nothing left to do);
      * otherwise relaunches, which resumes from the manifests.

    Before Gate 2 has been reviewed, the pipeline halts at Gate 2 on its own. Once you have
    read the Gate 2 report and want to continue, create the approval file:

        New-Item results/organisms/GATE2_APPROVED -ItemType File

    and this script will start passing --continue-past-gate.

    Register it (no admin needed):   powershell -File scripts/register_resume_task.ps1
    Run it by hand:                  powershell -File scripts/resume.ps1
#>

$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$logDir = Join-Path $repo 'logs'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$watchdogLog = Join-Path $logDir 'watchdog.log'

function Write-Log([string]$msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg"
    Add-Content -Path $watchdogLog -Value $line -Encoding utf8
    Write-Output $line
}

# --- already finished? -------------------------------------------------------
if (Test-Path (Join-Path $repo 'results/PIPELINE_COMPLETE')) {
    Write-Log 'PIPELINE_COMPLETE present; nothing to resume.'
    exit 0
}

# --- already running? --------------------------------------------------------
# Match on the interpreter inside this repo's venv so unrelated python processes
# elsewhere on the machine are never mistaken for this pipeline.
$venvPython = Join-Path $repo '.venv\Scripts\python.exe'
$running = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.ExecutablePath -eq $venvPython }

if ($running) {
    Write-Log "already running (PID $($running.ProcessId -join ', ')); nothing to do."
    exit 0
}

# --- waiting at Gate 2? ------------------------------------------------------
# Gate 2 is the one human checkpoint. Once its report exists and has not been
# approved, relaunching only recomputes the same verdict and halts again, so the
# watchdog stands down until a human signs off.
$gateApproved = Test-Path (Join-Path $repo 'results/organisms/GATE2_APPROVED')
$gateReported = Test-Path (Join-Path $repo 'results/organisms/gate2_report.txt')
if ($gateReported -and -not $gateApproved) {
    Write-Log 'halted at Gate 2 awaiting human review; not relaunching.'
    exit 0
}
$argList = @('-u', 'scripts/run_all.py')
if ($gateApproved) { $argList += '--continue-past-gate' }

$outLog = Join-Path $logDir "run-$stamp.log"
$errLog = Join-Path $logDir "run-$stamp.err"

Write-Log "resuming pipeline (gate2_approved=$gateApproved) -> $outLog"

$p = Start-Process -FilePath $venvPython -ArgumentList $argList `
    -WorkingDirectory $repo -RedirectStandardOutput $outLog -RedirectStandardError $errLog `
    -WindowStyle Hidden -PassThru

Write-Log "launched PID $($p.Id)"
exit 0
