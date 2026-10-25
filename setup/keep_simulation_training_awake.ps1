# Scoped temporary wake request; releases when this simulation run ends.
param([Parameter(Mandatory=$true)][string]$RunDirectory)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$runsDirectory = [IO.Path]::GetFullPath((Join-Path $projectDirectory 'runs'))
$resolvedRun = (Resolve-Path -LiteralPath $RunDirectory).Path
if ((Split-Path -Parent $resolvedRun) -ne $runsDirectory -or
    -not (Split-Path -Leaf $resolvedRun).StartsWith('simulation_')) {
    throw 'Only a local simulation training run can hold this power request'
}
$statusPath = Join-Path $resolvedRun 'status.json'
$logPath = Join-Path $resolvedRun 'keep_awake.log'
$deadline = [DateTime]::UtcNow.AddHours(4)
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class SimulationTrainingPowerRequest {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern UInt32 SetThreadExecutionState(UInt32 flags);
}
'@
try {
    $result = [SimulationTrainingPowerRequest]::SetThreadExecutionState([uint32]2147483649)
    if ($result -eq 0) { throw 'Could not request system wakefulness' }
    "Temporary system wake request active at $(Get-Date -Format o)" | Set-Content -LiteralPath $logPath
    while ([DateTime]::UtcNow -lt $deadline -and (Test-Path -LiteralPath $statusPath)) {
        $state = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
        if ($state.status -notin @('starting', 'running')) { break }
        Start-Sleep -Seconds 10
    }
}
finally {
    [SimulationTrainingPowerRequest]::SetThreadExecutionState([uint32]2147483648) | Out-Null
    "Temporary system wake request released at $(Get-Date -Format o)" | Add-Content -LiteralPath $logPath
}
