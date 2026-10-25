param(
    [ValidateSet('combined','market_street','residential_colony','divided_main_road','rural_road','irregular_junction','all')][string]$Map = 'combined',
    [int]$Seed = 7101,
    [ValidateRange(5,1500)][int]$SimulationSeconds = 60,
    [ValidateSet('clear','overcast','wet','dusk')][string]$Weather = 'clear',
    [ValidateSet('Low','Epic')][string]$Quality = 'Low',
    [switch]$Check
)
$ErrorActionPreference = 'Stop'
if ($Map -ieq 'combined') {
    $townSeconds = if ($PSBoundParameters.ContainsKey('SimulationSeconds')) { $SimulationSeconds } else { 900 }
    & (Join-Path $PSScriptRoot 'Start_Combined_Indian_Traffic.ps1') -Seed $Seed -SimulationSeconds $townSeconds -Weather $Weather -Quality $Quality -Check:$Check
    exit $LASTEXITCODE
}
$worldPython = Join-Path $PSScriptRoot '.venv-carla/Scripts/python.exe'
$worldRunner = Join-Path $PSScriptRoot 'setup/carla/run_session.py'
$worldArguments = @($worldRunner,'--india-world',$Map.ToLowerInvariant(),'--seed',"$Seed",'--sim-seconds',"$SimulationSeconds",'--seconds','1500','--weather',$Weather.ToLowerInvariant(),'--quality',$Quality)
if ($Check) { $worldArguments += '--smoke' }
& $worldPython @worldArguments
if ($LASTEXITCODE -ne 0) { Write-Warning 'Inspect the newest runs/carla_indian_worlds_* result and log.' }
