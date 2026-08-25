<#
    Read-only crash diagnostic. Run ELEVATED - the interesting parts (minidump inventory,
    Intel XTU's applied tuning profile) are not readable as a normal user.

        powershell -ExecutionPolicy Bypass -File scripts\diagnose_bsod.ps1

    This script only reads. It changes no setting, deletes nothing, and installs nothing.
    See BSOD_DIAGNOSIS.md for what the findings meant on 2026-08-25.
#>

$ErrorActionPreference = 'Continue'

function Section($t) { "`n$('=' * 74)`n$t`n$('=' * 74)" }

$elevated = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()
            ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
Section "ELEVATION"
if ($elevated) { "Running as Administrator - full diagnostic available." }
else { "NOT elevated. Minidumps and XTU tuning values will be unreadable. Re-run from an elevated PowerShell for the parts that matter." }

Section "CRASH HISTORY (90 days)"
Get-WinEvent -FilterHashtable @{LogName='System'; Id=41,1001,6008; StartTime=(Get-Date).AddDays(-90)} -EA SilentlyContinue |
    Select-Object TimeCreated, Id, @{n='Detail';e={ ($_.Message -replace "`r?`n",' ').Trim() }} |
    Sort-Object TimeCreated -Descending | Format-List

Section "MINIDUMPS"
$dumps = Get-ChildItem 'C:\Windows\Minidump\*.dmp' -EA SilentlyContinue
if ($dumps) {
    $dumps | Sort-Object LastWriteTime -Desc |
        Select-Object Name, LastWriteTime, @{n='KB';e={[int]($_.Length/1KB)}} | Format-Table -AutoSize
    ""
    "To identify the faulting driver, open the newest .dmp in one of:"
    "  * BlueScreenView (NirSoft)  - fastest, names the driver directly"
    "  * WinDbg  ->  !analyze -v"
} else {
    "NO MINIDUMPS PRESENT."
    "The event log records dumps being written, so they are being deleted after the fact."
    "Storage Sense is the usual culprit - see the STORAGE SENSE section below."
}
"MEMORY.DMP (kernel dump): $(if (Test-Path C:\Windows\MEMORY.DMP) { (Get-Item C:\Windows\MEMORY.DMP).LastWriteTime } else { 'absent' })"

Section "DUMP CONFIGURATION"
$cc = Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\CrashControl' -EA SilentlyContinue
$kinds = @{0='none';1='complete';2='kernel';3='small (minidump)';7='automatic'}
"CrashDumpEnabled : $($cc.CrashDumpEnabled)  ($($kinds[[int]$cc.CrashDumpEnabled]))"
"MinidumpDir      : $($cc.MinidumpDir)"
"AutoReboot       : $($cc.AutoReboot)"
"A *kernel* dump (value 2) retains far more evidence than a minidump. Change under"
"System > About > Advanced system settings > Startup and Recovery."

Section "STORAGE SENSE (deletes crash dumps)"
$sp = 'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\StorageSense\Parameters\StoragePolicy'
$ss = Get-ItemProperty $sp -EA SilentlyContinue
if ($ss) {
    "Enabled ('01')                : $($ss.'01')"
    "Cleanup temp files ('04')     : $($ss.'04')"
    "If enabled, turn it off (or exclude system dumps) before trying to reproduce a crash,"
    "otherwise the evidence is removed automatically."
} else { "StorageSense policy key not present." }

Section "CPU TUNING SOFTWARE (prime suspect for 0x101 CLOCK_WATCHDOG_TIMEOUT)"
Get-Service -Name '*XTU*','*Extreme Tuning*','*ThrottleStop*','*Afterburner*' -EA SilentlyContinue |
    Select-Object Name, DisplayName, Status, StartType | Format-Table -AutoSize
$names = 'Afterburner','XTU','ThrottleStop','Armoury','Precision','RivaTuner','Ryzen Master','Vantage','Command Center'
$inst = Get-ChildItem 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall',
                      'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall' -EA SilentlyContinue |
        ForEach-Object { $_.GetValue('DisplayName') } |
        Where-Object { $d = $_; $d -and ($names | Where-Object { $d -like "*$_*" }) }
if ($inst) { "Installed:"; $inst | ForEach-Object { "  $_" } } else { "None of the common tuning tools are installed." }

Section "INTEL XTU APPLIED TUNING (needs elevation)"
$xtuPaths = @(
    'HKLM:\SOFTWARE\Intel\Intel(R) Extreme Tuning Utility'
    'HKLM:\SOFTWARE\WOW6432Node\Intel\Intel(R) Extreme Tuning Utility'
    'HKLM:\SOFTWARE\Intel\XTU'
)
$any = $false
foreach ($p in $xtuPaths) {
    if (Test-Path $p) {
        $any = $true
        "--- $p"
        Get-ChildItem $p -Recurse -EA SilentlyContinue | ForEach-Object {
            $k = $_
            $k.Property | ForEach-Object { "    $($k.PSChildName)\$_ = $($k.GetValue($_))" }
        }
    }
}
if (-not $any) { "No readable XTU keys (either not installed, or elevation is required)." }
"`nWhat to look for: any non-zero core/cache VOLTAGE OFFSET, or a raised turbo ratio or"
"power limit. If one is applied, resetting XTU to defaults is the first thing to try."

Section "GPU AND DISPLAY DRIVER"
& nvidia-smi --query-gpu=name,driver_version,vbios_version,temperature.gpu,power.draw --format=csv 2>&1
"`nDisplay-driver error/recovery events (nvlddmkm), last 30 days:"
$nv = Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='nvlddmkm'; StartTime=(Get-Date).AddDays(-30)} -EA SilentlyContinue
if ($nv) { $nv | Select-Object TimeCreated, Id | Sort-Object TimeCreated -Desc | Select-Object -First 20 | Format-Table -AutoSize }
else { "  none" }
"Event ID 153 = GPU error recovery. Repeated 153s under load point at driver instability."

Section "TDR CONFIGURATION"
$gd = Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers' -EA SilentlyContinue
"TdrLevel    : $($gd.TdrLevel)     (unset = default 3, recover)"
"TdrDelay    : $($gd.TdrDelay)     (unset = default 2 seconds)"
"TdrDdiDelay : $($gd.TdrDdiDelay)  (unset = default 5 seconds)"
"Raised delays are common for ML workloads, but they let a hung GPU sit longer before"
"reset, which makes escalation to a CPU watchdog bugcheck more likely."

Section "THERMAL / POWER"
$b = Get-CimInstance Win32_Battery -EA SilentlyContinue
if ($b) { "BatteryStatus: $($b.BatteryStatus)  (2 = running on AC)" }
powercfg /getactivescheme
"`nSustained-load sanity check: run a GPU job and watch"
"  nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,clocks_event_reasons.active --format=csv -l 5"

Section "SUGGESTED ORDER OF ACTION"
@"
1. Reset Intel XTU to defaults (or uninstall). 0x101 CLOCK_WATCHDOG_TIMEOUT is the
   signature bugcheck of an unstable CPU undervolt/overclock.
2. Disable Storage Sense and switch to a kernel dump, so the NEXT crash is diagnosable.
3. Update the NVIDIA driver.
4. Re-test with a long GPU job. If it survives, step 1 was the cause.
5. Only if crashes persist: revert TdrDelay/TdrDdiDelay to Windows defaults.
"@
