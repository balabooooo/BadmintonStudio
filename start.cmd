@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================
echo   羽毛球智能剪辑台 · Badminton Studio
echo ============================================
echo.

if not exist ".venv\Scripts\python.exe" (
  echo [错误] 没找到虚拟环境 .venv
  echo        请先执行:  python -m venv --system-site-packages .venv
  echo                   .venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)

if not exist "frontend\dist\index.html" (
  echo [提示] 前端还没构建，正在构建...
  pushd frontend
  call npm install
  call npm run build
  popd
)

set PYTHONPATH=%~dp0backend
set PYTHONIOENCODING=utf-8

".venv\Scripts\python.exe" "desktop\app.py" %*
if errorlevel 1 pause
endlocal
