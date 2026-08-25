# Host instability during this run — diagnosis

The machine bugchecked three times while running this project. Nothing in the pipeline can
cause that (user-space code cannot bugcheck a kernel), but a sustained GPU+CPU load is an
effective way to expose a latent driver or voltage fault, so it is worth writing down what
the evidence says.

**This is diagnosis, not a fix.** Every recommendation below needs administrator rights and
is the machine owner's call.

## What happened

| Time (24 Aug 2026) | Bugcheck | Meaning |
|---|---|---|
| 03:16:00 | `0x00000101` | **CLOCK_WATCHDOG_TIMEOUT** — a CPU core stopped responding to an interprocessor interrupt within the watchdog interval |
| 03:17:31 | `0x00020001` | crashed ~1 s after booting |
| 05:01:28 | `0x00020001` | again, mid-run |

Bugcheck args for the 0x101: `(0x28, 0x1, 0x29b92701, 0xfc810000)`.

**Ruled out:** sleep and screen timeouts. The machine is on AC and every idle timeout in
the active scheme (Balanced) is `0`. No WHEA errors were logged in 24 hours, which argues
against outright failing silicon.

## Primary suspect: Intel XTU

```
XtuService   Status: Running   StartType: Automatic
Installed:   Install_Intel_IPF_XTU, XTU_Installer
```

**`0x101 CLOCK_WATCHDOG_TIMEOUT` is the canonical bugcheck for an unstable CPU undervolt
or overclock.** The mechanism fits exactly: a core running at insufficient voltage for its
requested frequency fails to retire work and misses the interprocessor interrupt, so the
watchdog fires. It shows up specifically under sustained all-core load — which is what
feeding a GPU trainer for hours produces, and why this workload triggered it when lighter
long-running jobs did not.

XTU's stored tuning profile could not be read without elevation, so **I could not confirm a
non-default voltage offset is actually applied** — only that the tuning service is
installed and running. That is the single highest-value thing to check.

## Secondary suspect: NVIDIA driver

```
24 Aug 19:04:45   nvlddmkm   Event ID 153
23 Aug 23:14:03   nvlddmkm   Event ID 153
```

Two GPU error-recovery events from the display driver, **both during heavy GPU load in this
project's runs**. Driver `592.07` on an RTX 5070 Laptop (Blackwell, `sm_120`) — a very new
mobile part, where sustained-load driver faults are not unusual.

These could be cause or effect: a display driver that hangs while holding a spinlock at
high IRQL will itself prevent another core from servicing an IPI, which produces a `0x101`.
So the NVIDIA and CPU stories are not mutually exclusive — one can produce the other.

## Why there is no dump to analyse

The event log records dumps written to `C:\Windows\Minidump\082426-*.dmp`. **The directory
is empty.**

```
CrashDumpEnabled : 3  (small memory dump / minidump)
MinidumpDir      : C:\Windows\Minidump      -> File Not Found
MEMORY.DMP       : absent
StorageSense     : ENABLED
```

Storage Sense is deleting the crash dumps. That is why the faulting driver cannot be named
— the one piece of evidence that would settle this is being cleaned up automatically.

## Non-default TDR configuration

```
TdrDelay    = 10   (Windows default: 2 seconds)
TdrDdiDelay = 10   (Windows default: 5 seconds)
TdrLevel    = <unset, i.e. default: recover>
```

Raising these is common for ML work — it stops Windows resetting the GPU during long
kernels. But it is double-edged: a genuinely hung GPU now sits for 10 seconds before the
driver is reset, which is a long time for a driver holding a kernel lock, and makes
escalation into a CPU-level watchdog bugcheck more likely rather than a recoverable TDR.

Worth knowing about; not obviously worth changing while doing GPU training.

## Recommendations, in order of expected value

1. **Reset Intel XTU to defaults, or uninstall it.** Highest probability fix given the
   `0x101` signature. If a voltage offset is applied, this is almost certainly the cause.
2. **Stop Storage Sense deleting crash dumps**, and switch to a *kernel* memory dump rather
   than a minidump (`System > About > Advanced system settings > Startup and Recovery`).
   Without this, the next crash will be as undiagnosable as these three.
3. **Update the NVIDIA driver** from 592.07. Two `nvlddmkm` error-recovery events under load
   on a new Blackwell mobile part is worth clearing before anything else is concluded.
4. **Re-test under load** — a long GPU job with XTU at defaults is the experiment that
   distinguishes hypothesis 1 from hypothesis 3.
5. Optional: revert `TdrDelay`/`TdrDdiDelay` to Windows defaults, accepting the risk of TDRs
   during long kernels, if crashes persist after 1–3.

## Running the elevated diagnostic

`scripts/diagnose_bsod.ps1` collects everything above **plus** the parts that need
administrator rights: the minidump inventory, XTU's applied tuning values, and the full
bugcheck parameter set. Run it from an elevated PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\diagnose_bsod.ps1
```

It only reads; it changes nothing.
