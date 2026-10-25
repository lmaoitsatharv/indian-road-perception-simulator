param(
    [ValidateRange(20,150)][int]$Seconds = 110,
    [int]$Seed = 4101,
    [ValidateSet('mixed','clear_overtake','oncoming','pedestrian','return_gap','junction')][string]$Scenario = 'mixed',
    [ValidateSet('clear','overcast','wet','dusk')][string]$Weather = 'clear',
    [switch]$Check
)
$ErrorActionPreference = 'Stop'
$livePython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$liveOwner = Join-Path $PSScriptRoot 'setup/run_live_simulation.py'
$liveArgs = @($liveOwner,'--sim-seconds',"$Seconds",'--seed',"$Seed",'--case',$Scenario.ToLowerInvariant(),'--weather',$Weather.ToLowerInvariant())
if ($Check) { $liveArgs += '--check' }
& $livePython @liveArgs
if ($LASTEXITCODE -ne 0) { throw 'Live preview needs inspection; its logs and results were preserved under runs/simulation_live_*.' }
