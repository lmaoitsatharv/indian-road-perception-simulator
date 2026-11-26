param(
    [ValidateRange(10,1500)][int]$SimulationSeconds = 600,
    [int]$Seed = 8101,
    [ValidateSet('clear','overcast','wet','dusk')][string]$Weather = 'clear',
    [ValidateSet('Low','Epic')][string]$Quality = 'Low',
    [switch]$NoReset,
    [switch]$Check,
    [string]$Destination = '',
    [string]$Destinations = '',
    [switch]$RLSpeed
)
$ErrorActionPreference = 'Stop'
$Quality = if ($Quality -ieq 'Epic') { 'Epic' } else { 'Low' }
$ownerPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$owner = Join-Path $PSScriptRoot 'setup/run_camera_driving.py'
$ownerArgs = @($owner,'--sim-seconds',"$SimulationSeconds",'--seed',"$Seed",'--weather',$Weather.ToLowerInvariant(),'--quality',$Quality)
if ($NoReset) { $ownerArgs += '--no-reset' }
if ($Check) { $ownerArgs += '--check' }
if ($Destination) { $ownerArgs += "--destination=$Destination" }   # "=": a negative x is not an option
if ($Destinations) { $ownerArgs += "--destinations=$Destinations" }
if ($RLSpeed) { $ownerArgs += '--rl-speed' }   # learned speed governor (docs/RL_SPEED_DESIGN_20261112.md)
& $ownerPython @ownerArgs
if ($LASTEXITCODE -ne 0) { Write-Warning 'Inspect the newest runs/camera_driving_* folder and its town result.' }
