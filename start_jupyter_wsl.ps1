# Run from PowerShell to open the project's Linux notebook server.
# Keep this terminal running while using notebooks.
param(
    [string]$WslDistribution = '',
    [string]$WslPython = '',
    [ValidateRange(1024,65535)][int]$Port = 8890
)
$ErrorActionPreference = 'Stop'
if (-not $WslDistribution) { $WslDistribution = if ($env:PROJECT_WSL_DISTRIBUTION) { $env:PROJECT_WSL_DISTRIBUTION } else { 'Ubuntu-24.04' } }
if (-not $WslPython) { $WslPython = $env:PROJECT_WSL_PYTHON }
if (-not $WslPython) {
    $wslHome = (& wsl.exe -d $WslDistribution --exec /usr/bin/printenv HOME | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $wslHome.StartsWith('/') -or $wslHome -match '[\r\n]') { throw 'Could not resolve the selected WSL home; set -WslPython explicitly.' }
    $WslPython = $wslHome.TrimEnd('/') + '/.venvs/idd-tf220/bin/python'
}
if (-not $WslPython.StartsWith('/') -or $WslPython -match '[\r\n]') { throw 'WslPython must be an absolute Linux path.' }
$projectPath = [IO.Path]::GetFullPath($PSScriptRoot)
if ($projectPath -notmatch '^([A-Za-z]):[\\/]') { throw 'Use a local Windows drive path.' }
$wslProject = '/mnt/' + $Matches[1].ToLowerInvariant() + '/' + $projectPath.Substring(3).Replace('\','/')
$arguments = @('-d',$WslDistribution,'--cd',$wslProject,'--exec',$WslPython,'-m','jupyterlab','--no-browser','--ip=127.0.0.1','--port',"$Port")
& wsl.exe @arguments
if ($LASTEXITCODE -ne 0) { throw 'Jupyter stopped with an error; inspect the terminal output.' }
