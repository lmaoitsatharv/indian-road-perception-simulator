# Run the external clips through a configured bounded WSL GPU environment.
param(
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [switch]$Resize,
    [ValidateRange(60,3600)][int]$MaxWallSeconds = 1800,
    [string]$WslDistribution = '',
    [string]$WslPython = ''
)
$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $WslDistribution) { $WslDistribution = if ($env:PROJECT_WSL_DISTRIBUTION) { $env:PROJECT_WSL_DISTRIBUTION } else { 'Ubuntu-24.04' } }
if (-not $WslPython) { $WslPython = $env:PROJECT_WSL_PYTHON }
if (-not $WslPython) {
    $wslHome = (& wsl.exe -d $WslDistribution --exec /usr/bin/printenv HOME | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $wslHome.StartsWith('/') -or $wslHome -match '[\r\n]') { throw 'Could not resolve the selected WSL home; set -WslPython explicitly.' }
    $WslPython = $wslHome.TrimEnd('/') + '/.venvs/idd-tf220/bin/python'
}
if (-not $WslPython.StartsWith('/') -or $WslPython -match '[\r\n]') { throw 'WslPython must be an absolute Linux path.' }
function Convert-ToWslPath([string]$WindowsPath) {
    $absolutePath = [IO.Path]::GetFullPath($WindowsPath)
    if ($absolutePath -notmatch '^([A-Za-z]):[\\/]') { throw 'Use a local Windows drive path.' }
    $driveName = $Matches[1].ToLowerInvariant()
    '/mnt/' + $driveName + '/' + $absolutePath.Substring(3).Replace('\','/')
}
$targetPath = if ([IO.Path]::IsPathRooted($OutputDir)) { $OutputDir } else { Join-Path $projectDir $OutputDir }
if (Test-Path -LiteralPath $targetPath) { throw 'Choose a new output directory; existing results are preserved.' }
$arguments = @('-d', $WslDistribution, '--cd', (Convert-ToWslPath $projectDir), '--exec',
    $WslPython,
    (Convert-ToWslPath (Join-Path $projectDir 'predict_road_video.py')),
    '--output-dir', (Convert-ToWslPath $targetPath), '--max-wall-seconds', [string]$MaxWallSeconds)
if ($Resize) { $arguments += '--resize' }
& wsl.exe @arguments
if ($LASTEXITCODE -ne 0) { throw 'Video prediction stopped or failed. Partial evidence is preserved; inspect run_status.json.' }
