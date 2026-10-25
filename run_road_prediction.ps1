# Run one offline prediction through a configured WSL environment.
param(
    [Parameter(Mandatory=$true)][string[]]$Image,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [switch]$Resize,
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
if ($Image.Count -eq 0) { throw 'Select at least one image.' }
$arguments = @('-d', $WslDistribution, '--cd', (Convert-ToWslPath $projectDir), '--exec',
    $WslPython,
    (Convert-ToWslPath (Join-Path $projectDir 'predict_road.py')))
foreach ($imagePath in $Image) {
    if (-not (Test-Path -LiteralPath $imagePath -PathType Leaf)) { throw "Image file not found: $imagePath" }
    $resolved = (Resolve-Path -LiteralPath $imagePath -ErrorAction Stop).Path
    $arguments += @('--image', (Convert-ToWslPath $resolved))
}
$targetPath = if ([IO.Path]::IsPathRooted($OutputDir)) { $OutputDir } else { Join-Path $projectDir $OutputDir }
if (Test-Path -LiteralPath $targetPath) { throw 'Choose a new output directory; existing results are preserved.' }
$arguments += @('--output-dir', (Convert-ToWslPath $targetPath))
if ($Resize) { $arguments += '--resize' }
& wsl.exe @arguments
if ($LASTEXITCODE -ne 0) { throw 'Prediction failed. Read the error above and use a fresh output directory for a retry.' }
