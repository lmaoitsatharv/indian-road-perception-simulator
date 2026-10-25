param(
    [int]$Seed = 7,
    [ValidateRange(30,1800)][int]$Seconds = 600,
    [ValidateSet('mixed','clear_overtake','oncoming','pedestrian','return_gap','junction')][string]$Scenario = 'mixed',
    [ValidateSet('follow','driver','junction')][string]$Camera = 'follow',
    [ValidateSet(1,1.5,2,2.5,3)][double]$Speed = 1,
    [ValidateSet('clear','overcast','wet','dusk')][string]$Weather = 'clear',
    [ValidateSet('Low','Epic')][string]$Quality = 'Low',
    [ValidateSet('asphalt','legacy')][string]$RoadSurface = 'asphalt',
    [ValidateSet(1,5)][int]$RoadCameraHz = 5,
    [switch]$Check,
    [switch]$ObserveRoad
)
$ErrorActionPreference = 'Stop'
$Scenario = $Scenario.ToLowerInvariant()
$Camera = $Camera.ToLowerInvariant()
$Weather = $Weather.ToLowerInvariant()
$RoadSurface = $RoadSurface.ToLowerInvariant()
$Quality = if ($Quality -ieq 'Epic') { 'Epic' } else { 'Low' }
$indiaPython = Join-Path $PSScriptRoot '.venv-carla/Scripts/python.exe'
$indiaRunner = Join-Path $PSScriptRoot 'setup/carla/run_session.py'
if (-not (Test-Path -LiteralPath $indiaPython)) { throw 'The CARLA environment is missing. Read INDIAN_SIMULATION.md.' }
$indiaSpeed = $Speed.ToString([Globalization.CultureInfo]::InvariantCulture)
$indiaArguments = @($indiaRunner, '--india', '--case', $Scenario, '--seed', "$Seed", '--seconds', "$Seconds", '--sim-seconds', '110', '--view', $Camera, '--speed', $indiaSpeed, '--weather', $Weather, '--quality', $Quality, '--surface-style', $RoadSurface)
if ($Check) { $indiaArguments += '--smoke' }
if ($ObserveRoad) { $indiaArguments += @('--observe-road','--road-camera-hz',"$RoadCameraHz") }
& $indiaPython @indiaArguments
if ($LASTEXITCODE -ne 0) { Write-Warning 'This run needs inspection. Its results and incidents are preserved in the newest runs/carla_india_* folder.' }
