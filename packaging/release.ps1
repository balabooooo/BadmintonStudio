# Publishes a GitHub Release with the prebuilt portable zip.
#
# Prereqs:
#   - run packaging\build_release.ps1 first (produces dist\BadmintonStudio-<ver>-win64.zip)
#   - `gh` installed and authenticated (`gh auth login`)
#
# Usage:
#   pwsh -File packaging\release.ps1 [version]
#
# The GitHub device/browser login, `git push` and asset upload all require
# access to github.com / uploads.github.com, so run this on a network that can
# reach them (e.g. with a proxy/VPN if needed).

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$Version = if ($args.Count -ge 1) { $args[0] } else { "0.1.0" }
$Tag = "v$Version"
$Root = Split-Path -Parent $PSScriptRoot
$Zip = Join-Path $Root "dist\BadmintonStudio-$Version-win64.zip"

if (-not (Test-Path $Zip)) { throw "missing asset: $Zip (run build_release.ps1 first)" }

gh auth status

Write-Host "==> Committing release files if any are uncommitted" -ForegroundColor Cyan
$dirty = git -C $Root status --porcelain
if ($dirty) {
    git -C $Root add config.py README.md .gitignore CHANGELOG.md packaging
    git -C $Root commit -m "chore: 打包发布 v$Version"
}

Write-Host "==> Tagging and pushing $Tag" -ForegroundColor Cyan
git -C $Root tag -f $Tag
git -C $Root push origin $Tag

Write-Host "==> Creating release and uploading asset" -ForegroundColor Cyan
gh release create $Tag $Zip `
    --title "Badminton Studio v$Version" `
    --notes "See [CHANGELOG.md](CHANGELOG.md) for details." `
    --target (git -C $Root rev-parse HEAD)

Write-Host "==> Done" -ForegroundColor Green
