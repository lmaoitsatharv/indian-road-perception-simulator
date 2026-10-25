param(
    [Parameter(Mandatory=$true)][string[]]$Episode,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [ValidateRange(30,3600)][int]$MaxWallSeconds = 1200,
    [string]$WslDistribution = '',
    [string]$WslPython = '',
    [ValidateSet('gpu','cpu')][string]$Device = 'gpu'
)
$ErrorActionPreference = 'Stop'
if (-not $WslDistribution) { $WslDistribution = if ($env:PROJECT_WSL_DISTRIBUTION) { $env:PROJECT_WSL_DISTRIBUTION } else { 'Ubuntu-24.04' } }
if (-not $WslPython) { $WslPython = $env:PROJECT_WSL_PYTHON }
if (-not $WslPython) {
    $wslHome = (& wsl.exe -d $WslDistribution --exec /usr/bin/printenv HOME | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $wslHome.StartsWith('/') -or $wslHome -match '[\r\n]') { throw 'Could not resolve the selected WSL home; set -WslPython explicitly.' }
    $WslPython = $wslHome.TrimEnd('/') + '/.venvs/idd-tf220/bin/python'
}
function Convert-RoadObserverPath([string]$Path) {
    $absolutePath = [IO.Path]::GetFullPath($Path)
    if ($absolutePath -notmatch '^([A-Za-z]):[\\/]') { throw 'Use a local Windows drive path.' }
    '/mnt/' + $Matches[1].ToLowerInvariant() + '/' + $absolutePath.Substring(3).Replace('\','/')
}
$targetPath = if ([IO.Path]::IsPathRooted($OutputDir)) { $OutputDir } else { Join-Path $PSScriptRoot $OutputDir }
if (Test-Path -LiteralPath $targetPath) { throw 'Choose a new output directory; existing comparisons are preserved.' }
if (-not $WslPython.StartsWith('/') -or $WslPython -match '[\r\n]') { throw 'WslPython must be an absolute Linux path.' }
$arguments = @('-d',$WslDistribution,'--cd',(Convert-RoadObserverPath $PSScriptRoot),'--exec',
    $WslPython,
    (Convert-RoadObserverPath (Join-Path $PSScriptRoot 'evaluate_carla_road.py')),
    '--output',(Convert-RoadObserverPath $targetPath),'--max-wall-seconds',"$MaxWallSeconds",'--device',$Device)
foreach ($sourcePath in $Episode) {
    $resolved = (Resolve-Path -LiteralPath $sourcePath -ErrorAction Stop).Path
    if (-not (Test-Path -LiteralPath (Join-Path $resolved 'road_camera/capture_config.json'))) { throw 'Select an episode containing a road_camera capture.' }
    $arguments += @('--episode',(Convert-RoadObserverPath $resolved))
}
& wsl.exe @arguments
if ($LASTEXITCODE -ne 0) { throw 'Observation evaluation failed; inspect its report.json before retrying in a new directory.' }
Write-Host "Comparison replay: $(Join-Path $targetPath 'index.html')"
