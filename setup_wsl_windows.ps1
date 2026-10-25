$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$setupDir = Join-Path $taskRoot 'setup'
New-Item -ItemType Directory -Path $setupDir -Force | Out-Null
$logPath = Join-Path $setupDir 'windows_wsl_install.log'
$statePath = Join-Path $setupDir 'windows_wsl_install.json'
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) { throw 'Run this script as Administrator.' }
Start-Transcript -Path $logPath -Append | Out-Null
$state = [ordered]@{ started = (Get-Date).ToString('o'); status = 'running'; automatic_restart = $false }
try {
    & "$env:SystemRoot\System32\wsl.exe" --install --distribution Ubuntu-24.04 --no-launch
    $state.wsl_install_exit_code = $LASTEXITCODE
    if ($LASTEXITCODE -ne 0) {
        # Enable prerequisites without restarting the user's computer.
        foreach ($feature in @('VirtualMachinePlatform', 'Microsoft-Windows-Subsystem-Linux')) {
            $result = Enable-WindowsOptionalFeature -Online -FeatureName $feature -All -NoRestart
            $state[$feature] = @{ state = $result.State.ToString(); restart_needed = $result.RestartNeeded }
        }
        $state.status = 'windows_features_enabled_restart_and_retry_install'
    } else {
        $state.status = 'install_command_completed_check_restart_requirement'
    }
    $state.virtualization_firmware_enabled = (Get-CimInstance Win32_Processor | Select-Object -First 1).VirtualizationFirmwareEnabled
    $state.hypervisor_present = (Get-CimInstance Win32_ComputerSystem).HypervisorPresent
} catch {
    $state.status = 'failed'
    $state.error = $_.Exception.Message
} finally {
    $state.finished = (Get-Date).ToString('o')
    $state | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $statePath -Encoding UTF8
    Stop-Transcript | Out-Null
}
