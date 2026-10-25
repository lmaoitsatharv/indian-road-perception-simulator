# Temporary wake request for this session only; never changes persistent power settings.
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$statusPath = Join-Path $projectDirectory 'setup\overnight_20261003\session_status.json'
$logPath = Join-Path $projectDirectory 'setup\overnight_20261003\keep_awake.log'
$deadline = [DateTimeOffset]::Parse('2026-10-03T01:30:00+00:00')
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class OvernightPowerRequest {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern UInt32 SetThreadExecutionState(UInt32 flags);
}
'@
try {
    if ([OvernightPowerRequest]::SetThreadExecutionState([uint32]2147483649) -eq 0) {
        throw 'Could not keep system awake'
    }
    "Wake request started $(Get-Date -Format o)" | Add-Content -LiteralPath $logPath
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        try {
            $state = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
            if ($state.status -in @('completed','stopped','failed')) { break }
        } catch { }
        Start-Sleep -Seconds 5
    }
} finally {
    [OvernightPowerRequest]::SetThreadExecutionState([uint32]2147483648) | Out-Null
    "Wake request released $(Get-Date -Format o)" | Add-Content -LiteralPath $logPath
}
