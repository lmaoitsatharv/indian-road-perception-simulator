param(
    [int]$Seed = 8101,
    [ValidateRange(10,1500)][int]$SimulationSeconds = 900,
    [ValidateSet('clear','overcast','wet','dusk')][string]$Weather = 'clear',
    [ValidateSet('Low','Epic')][string]$Quality = 'Low',
    [switch]$Check
)
$ErrorActionPreference = 'Stop'
$Quality = if ($Quality -ieq 'Epic') { 'Epic' } else { 'Low' }
$townPython = Join-Path $PSScriptRoot '.venv-carla/Scripts/python.exe'
$townRunner = Join-Path $PSScriptRoot 'setup/carla/run_session.py'
$townArguments = @($townRunner,'--india-world','combined','--seed',"$Seed",'--sim-seconds',"$SimulationSeconds",'--seconds','1800','--weather',$Weather.ToLowerInvariant(),'--quality',$Quality)
if ($Check) { $townArguments += '--smoke' }
& $townPython @townArguments
if ($LASTEXITCODE -ne 0) { Write-Warning 'Inspect the newest runs/carla_indian_worlds_* combined-town result.' }
