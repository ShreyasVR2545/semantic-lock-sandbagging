<#
    Register the auto-resume watchdog as a Windows scheduled task.

    Two triggers:
      * at logon  - covers the reboot after a bugcheck;
      * every 15 minutes - covers the process dying without a reboot, and covers the case
        where the machine sat at the lock screen for a while before anyone logged in.

    Both are harmless when nothing needs doing: resume.ps1 exits immediately if the
    pipeline is already running or already complete.

    No administrator rights are needed - this registers a task for the current user only.

    Register:    powershell -ExecutionPolicy Bypass -File scripts/register_resume_task.ps1
    Remove:      powershell -ExecutionPolicy Bypass -File scripts/register_resume_task.ps1 -Remove
#>

param([switch]$Remove)

$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$taskName = 'SemanticLockSandbagging-Resume'
$script = Join-Path $repo 'scripts\resume.ps1'

if ($Remove) {
    if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        "removed scheduled task '$taskName'"
    } else {
        "no scheduled task '$taskName' to remove"
    }
    exit 0
}

if (-not (Test-Path $script)) { throw "missing $script" }

$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$script`"" `
    -WorkingDirectory $repo

$logonTrigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

# A repeating trigger, starting a minute from now and repeating for a year. This is the
# part that recovers a run which died without taking the machine down with it.
$repeatTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 15) `
    -RepetitionDuration (New-TimeSpan -Days 365)

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries:$false `
    -DontStopIfGoingOnBatteries:$false `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Days 0)   # 0 = no time limit

$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive

Register-ScheduledTask -TaskName $taskName -Action $action `
    -Trigger @($logonTrigger, $repeatTrigger) -Settings $settings -Principal $principal -Force | Out-Null

"registered scheduled task '$taskName'"
"  triggers : at logon, and every 15 minutes"
"  action   : $script"
"  logs     : $repo\logs\watchdog.log"
""
"To stop it later:  powershell -ExecutionPolicy Bypass -File scripts/register_resume_task.ps1 -Remove"
