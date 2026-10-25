param([switch]$DownloadCarla)
$ErrorActionPreference = 'Stop'
$bootstrapRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Push-Location -LiteralPath $bootstrapRoot
try {
    $bootstrapPython = Join-Path $bootstrapRoot '.venv-carla/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $bootstrapPython)) {
        & py -3.12 -m venv .venv-carla
        if ($LASTEXITCODE -ne 0) { throw 'Install official 64-bit Python 3.12, including the py launcher, then retry.' }
    }
    & $bootstrapPython -m pip install -r setup/carla/requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'CARLA Python dependencies could not be installed.' }
    $bootstrapArchive = Join-Path $bootstrapRoot 'setup/carla/downloads/CARLA_0.9.16.zip'
    if ($DownloadCarla -and -not (Test-Path -LiteralPath 'setup/carla/installation.json')) {
        New-Item -ItemType Directory -Path (Split-Path $bootstrapArchive -Parent) -Force | Out-Null
        if (-not (Test-Path -LiteralPath $bootstrapArchive)) {
            & curl.exe --fail --location --output "$bootstrapArchive.part" 'https://downloads.carlasim.com/Windows/CARLA_0.9.16.zip'
            if ($LASTEXITCODE -ne 0) { throw 'Download incomplete. The partial file is preserved; retry or download the official package manually.' }
            Move-Item -LiteralPath "$bootstrapArchive.part" -Destination $bootstrapArchive
        }
        & $bootstrapPython setup/carla/install_package.py
        if ($LASTEXITCODE -ne 0) { throw 'CARLA extraction did not finish; inspect the error above.' }
    }
    Write-Host 'Python environment ready. For the native engine, use -DownloadCarla or follow docs/GETTING_STARTED.md.'
} finally { Pop-Location }
