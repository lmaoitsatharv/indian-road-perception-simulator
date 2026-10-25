param(
    [int]$Seed = 7,
    [ValidateRange(1,1800)][int]$Seconds = 600,
    [string]$Map = 'Town10HD_Opt',
    [ValidateSet('Epic','Low')][string]$Quality = 'Epic',
    [switch]$Smoke
)
$ErrorActionPreference = 'Stop'
$carlaPython = Join-Path $PSScriptRoot '.venv-carla/Scripts/python.exe'
$carlaRunner = Join-Path $PSScriptRoot 'setup/carla/run_session.py'
if (-not (Test-Path -LiteralPath $carlaPython)) { throw 'The isolated CARLA Python environment is missing.' }
$carlaArguments = @($carlaRunner, '--seed', "$Seed", '--seconds', "$Seconds", '--map', $Map, '--quality', $Quality)
if ($Smoke) { $carlaArguments += '--smoke' }
& $carlaPython @carlaArguments
if ($LASTEXITCODE -ne 0) { throw 'CARLA did not finish successfully. Inspect setup/carla/server_owner.json and the newest runs/carla_* result.' }
