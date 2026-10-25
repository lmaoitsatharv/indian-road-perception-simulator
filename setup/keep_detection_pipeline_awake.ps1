# Temporary system wake request for the owned preparation/training/evaluation pipeline.
param([string]$SessionName = 'detection_pilot_prepare_20261025_01')
$ErrorActionPreference = 'Stop'
if ($SessionName -notmatch '^detection_pilot_(prepare|recovery)_[A-Za-z0-9_]+$') { throw 'Invalid pipeline session name' }
$projectDirectory = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$sessionDirectory = Join-Path (Join-Path $projectDirectory 'runs') $SessionName
$owner = Get-Content -LiteralPath (Join-Path $sessionDirectory 'owner.json') -Raw | ConvertFrom-Json
$workerId = [int]$owner.pid
if ($workerId -le 0 -or [string]$owner.linux_process_start_ticks -notmatch '^\d+$') { throw 'Invalid worker identity' }
$recordPath = Join-Path $sessionDirectory 'keep_awake.json'
$startedUtc = [DateTime]::UtcNow.ToString('o')
$deadline = [DateTime]::UtcNow.AddHours(72)
$releaseReason = 'deadline'
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class DetectionPipelinePowerRequest {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern UInt32 SetThreadExecutionState(UInt32 flags);
}
'@
try {
    if ([DetectionPipelinePowerRequest]::SetThreadExecutionState([uint32]2147483649) -eq 0) { throw 'Could not request system wakefulness' }
    @{status='active';started_utc=$startedUtc;windows_pid=$PID;worker_pid=$workerId;display_forced_on=$false;power_plan_changed=$false} |
        ConvertTo-Json | Set-Content -LiteralPath $recordPath
    while ([DateTime]::UtcNow -lt $deadline) {
        if (Test-Path -LiteralPath (Join-Path $sessionDirectory 'exit.json')) { $releaseReason='pipeline_finished'; break }
        $workerStatPath = '/proc/' + $workerId + '/stat'
        $workerStat = & wsl.exe -d Ubuntu-24.04 --exec /bin/cat $workerStatPath 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $workerStat) { $releaseReason='worker_not_running'; break }
        $remainingFields = ([string]$workerStat -split '\)\s',2)[1] -split '\s+'
        if ($remainingFields[19] -ne [string]$owner.linux_process_start_ticks) { $releaseReason='worker_identity_changed'; break }
        Start-Sleep -Seconds 30
    }
}
finally {
    [DetectionPipelinePowerRequest]::SetThreadExecutionState([uint32]2147483648) | Out-Null
    @{status='released';started_utc=$startedUtc;released_utc=[DateTime]::UtcNow.ToString('o');reason=$releaseReason;power_plan_changed=$false} |
        ConvertTo-Json | Set-Content -LiteralPath $recordPath
}
