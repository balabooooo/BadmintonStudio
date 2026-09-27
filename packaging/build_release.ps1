# Builds the Badminton Studio portable Windows release (one-folder -> zip).
#
# Usage:
#   pwsh -File packaging\build_release.ps1
#
# Outputs:
#   dist\BadmintonStudio\          the frozen app (run BadmintonStudio.exe)
#   dist\BadmintonStudio-<ver>-win64.zip   portable distribution archive
#
# Requirements on the build machine: Node.js + npm (frontend), a Python 3.14
# base interpreter (used to create the isolated build venv). The app bundles a
# CPU-only torch so it runs on any Windows machine.

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$Root = Split-Path -Parent $PSScriptRoot
$Version = "0.1.0"
$Venv = Join-Path $PSScriptRoot ".venv-cpu"
$Python = Join-Path $Venv "Scripts\python.exe"

Write-Host "==> Building frontend" -ForegroundColor Cyan
Push-Location (Join-Path $Root "frontend")
try {
    npm run build
    if ($LASTEXITCODE -ne 0) { throw "npm run build failed" }
} finally {
    Pop-Location
}

if (-not (Test-Path $Python)) {
    Write-Host "==> Creating CPU build venv" -ForegroundColor Cyan
    & (Join-Path $Root ".venv\Scripts\python.exe") -m venv $Venv
    & $Python -m pip install --upgrade pip
    & $Python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    & $Python -m pip install -r (Join-Path $Root "requirements.txt")
    & $Python -m pip install pyinstaller
}

Write-Host "==> Freezing with PyInstaller" -ForegroundColor Cyan
Push-Location $Root
try {
    & $Python -m PyInstaller --clean --noconfirm (Join-Path $PSScriptRoot "BadmintonStudio.spec")
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
} finally {
    Pop-Location
}

Write-Host "==> Creating portable zip" -ForegroundColor Cyan
$Dist = Join-Path $Root "dist"
$Zip = Join-Path $Dist "BadmintonStudio-$Version-win64.zip"
if (Test-Path $Zip) { Remove-Item $Zip -Force }
& tar.exe -a -c -f $Zip -C $Dist "BadmintonStudio"
if ($LASTEXITCODE -ne 0) { throw "zip failed" }

Write-Host "==> Done: $Zip" -ForegroundColor Green
