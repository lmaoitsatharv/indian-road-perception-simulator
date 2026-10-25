# Temporary system wake request scoped to one live detector-training worker.
param([Parameter(Mandatory=$true)][string]$RunName)
$ErrorActionPreference = 'Stop'
if ($RunName -notmatch '^detection_train_[A-Za-z0-9_]+$') { throw 'Invalid detector run name' }
$projectDirectory = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$runsDirectory = [IO.Path]::GetFullPath((Join-Path $projectDirectory 'runs'))
$ownerDirectory = (Resolve-Path -LiteralPath (Join-Path $runsDirectory ($RunName + '_launcher'))).Path
if ((Split-Path -Parent $ownerDirectory) -ne $runsDirectory) { throw 'Launcher must be inside this project' }
$owner = Get-Content -LiteralPath (Join-Path $ownerDirectory 'owner.json') -Raw | ConvertFrom-Json
$workerId = [int]$owner.pid
if ($workerId -le 0 -or [string]$owner.linux_process_start_ticks -notmatch '^\d+$') { throw 'Invalid worker identity' }
$recordPath = Join-Path $ownerDirectory 'keep_awake.json'
$startedUtc = [DateTime]::UtcNow.ToString('o')
$deadline = [DateTime]::UtcNow.AddHours(72)
$releaseReason = 'deadline'
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class DetectionTrainingPowerRequest {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern UInt32 SetThreadExecutionState(UInt32 flags);
}
'@
try {
    if ([DetectionTrainingPowerRequest]::SetThreadExecutionState([uint32]2147483649) -eq 0) {
        throw 'Could not request system wakefulness'
    }
    @{status='active';started_utc=$startedUtc;windows_pid=$PID;worker_pid=$workerId;display_forced_on=$false;power_plan_changed=$false} |
        ConvertTo-Json | Set-Content -LiteralPath $recordPath
    while ([DateTime]::UtcNow -lt $deadline) {
        if (Test-Path -LiteralPath (Join-Path $ownerDirectory 'exit.json')) { $releaseReason='worker_finished'; break }
        $workerStatPath = '/proc/' + $workerId + '/stat'
        $workerStat = & wsl.exe -d Ubuntu-24.04 --exec /bin/cat $workerStatPath 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $workerStat) { $releaseReason='worker_not_running'; break }
        $remainingFields = ([string]$workerStat -split '\)\s',2)[1] -split '\s+'
        if ($remainingFields[19] -ne [string]$owner.linux_process_start_ticks) { $releaseReason='worker_identity_changed'; break }
        Start-Sleep -Seconds 30
    }
}
finally {
    [DetectionTrainingPowerRequest]::SetThreadExecutionState([uint32]2147483648) | Out-Null
    @{status='released';started_utc=$startedUtc;released_utc=[DateTime]::UtcNow.ToString('o');reason=$releaseReason;power_plan_changed=$false} |
        ConvertTo-Json | Set-Content -LiteralPath $recordPath
}
