# Keep this long training run awake, then release the Windows power request.
param([switch]$WaitForReview)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$statusPath = Join-Path $projectDirectory 'runs\frozen_01\run_status.json'
$logName = if ($WaitForReview) { 'frozen_01_review_keep_awake.log' } else { 'frozen_01_keep_awake.log' }
$logPath = Join-Path $projectDirectory ('setup\' + $logName)
$completionPath = Join-Path $projectDirectory 'setup\frozen_01_work_complete.json'
$deadline = [DateTime]::UtcNow.AddHours(4)
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class TrainingPowerRequest {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern UInt32 SetThreadExecutionState(UInt32 flags);
}
'@
try {
    $result = [TrainingPowerRequest]::SetThreadExecutionState([uint32]2147483649)
    if ($result -eq 0) { throw 'Could not request system wakefulness' }
    "Temporary system wake request active at $(Get-Date -Format o)" | Set-Content -LiteralPath $logPath
    while (Test-Path -LiteralPath $statusPath) {
        if ([DateTime]::UtcNow -ge $deadline) { break }
        $runStatus = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
        if ($runStatus.status -ne 'running') {
            if (-not ($WaitForReview -and $runStatus.status -eq 'completed' -and
                      -not (Test-Path -LiteralPath $completionPath))) { break }
        }
        Start-Sleep -Seconds 5
    }
}
finally {
    [TrainingPowerRequest]::SetThreadExecutionState([uint32]2147483648) | Out-Null
    "Temporary system wake request released at $(Get-Date -Format o)" | Add-Content -LiteralPath $logPath
}
