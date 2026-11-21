# Builds the Badminton Studio Windows release (one-folder -> zip -> optional installer).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File packaging\build_release.ps1 [version] [-SkipFrontend] [-Installer]
#
# Outputs:
#   dist\BadmintonStudio\                          the frozen app (run BadmintonStudio.exe)
#   dist\BadmintonStudio-<ver>-win64.zip           portable distribution archive
#   dist\BadmintonStudio-Setup-<ver>-win64.exe     installer (only with -Installer)
#
# -SkipFrontend reuses the existing frontend\dist (use it when only backend code changed).
# -Installer requires Inno Setup 6 (winget install --id JRSoftware.InnoSetup --silent).
#
# Requirements on the build machine: Node.js + npm (frontend), a Python 3.14
# base interpreter (used to create the isolated build venv). The app bundles a
# CPU-only torch so it runs on any Windows machine.

param(
    [string]$Version = "0.1.0",
    [switch]$SkipFrontend,
    [switch]$Installer
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $PSScriptRoot ".venv-cpu"
$Python = Join-Path $Venv "Scripts\python.exe"

if (-not $SkipFrontend) {
    Write-Host "==> Building frontend" -ForegroundColor Cyan
    Push-Location (Join-Path $Root "frontend")
    try {
        npm run build
        if ($LASTEXITCODE -ne 0) { throw "npm run build failed" }
    } finally {
        Pop-Location
    }
} else {
    Write-Host "==> Skipping frontend build (-SkipFrontend)" -ForegroundColor Yellow
    if (-not (Test-Path (Join-Path $Root "frontend\dist\index.html"))) {
        throw "frontend\dist is missing; run without -SkipFrontend first"
    }
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

if ($Installer) {
    $Iscc = Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"
    if (-not (Test-Path $Iscc)) {
        $Iscc = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
    }
    if (-not (Test-Path $Iscc)) {
        throw "Inno Setup 6 not found. Install it with: winget install --id JRSoftware.InnoSetup --silent"
    }

    Write-Host "==> Compiling installer with Inno Setup" -ForegroundColor Cyan
    & $Iscc "/DAppVersion=$Version" (Join-Path $PSScriptRoot "BadmintonStudio.iss")
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup compile failed" }

    $Setup = Join-Path $Dist "BadmintonStudio-Setup-$Version-win64.exe"
    if (-not (Test-Path $Setup)) { throw "installer was not produced: $Setup" }
    Write-Host "==> Done: $Setup" -ForegroundColor Green
}
