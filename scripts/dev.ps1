# 开发模式：后端（uvicorn --reload）+ 前端（Vite HMR）一起跑
# 用法:  pwsh -File scripts/dev.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "[错误] 找不到 .venv，请先创建虚拟环境" -ForegroundColor Red
    exit 1
}

$env:PYTHONPATH = Join-Path $root "backend"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "启动后端 http://127.0.0.1:8000 ..." -ForegroundColor Green
$backend = Start-Process -PassThru -NoNewWindow $py `
    -ArgumentList "-m", "uvicorn", "bms.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"

Start-Sleep -Seconds 2

Write-Host "启动前端 http://127.0.0.1:5273 ..." -ForegroundColor Green
Push-Location (Join-Path $root "frontend")
$frontend = Start-Process -PassThru -NoNewWindow "npm" -ArgumentList "run", "dev"
Pop-Location

Write-Host ""
Write-Host "  打开 http://127.0.0.1:5273 开发（支持热更新）" -ForegroundColor Cyan
Write-Host "  打开 http://127.0.0.1:8000 使用已构建版本" -ForegroundColor Cyan
Write-Host "  按 Ctrl+C 结束" -ForegroundColor DarkGray

try {
    Wait-Process -Id $backend.Id
} finally {
    Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $frontend.Id -Force -ErrorAction SilentlyContinue
}
