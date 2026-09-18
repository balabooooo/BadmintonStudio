"""BadmintonStudio 桌面外壳。

把本地 FastAPI 服务跑在一个后台线程里，再用 pywebview（Windows 上是
WebView2）开一个无浏览器边框的原生窗口，观感就是一个普通桌面软件。

用法::

    python desktop/app.py [--port 8000] [--browser]

没有安装 pywebview 时自动回退成「启动服务 + 打开系统默认浏览器」。
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def free_port(preferred: int = 8000) -> int:
    for p in range(preferred, preferred + 40):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return 0


def wait_ready(url: str, timeout: float = 45.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url + "/api/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.35)
    return False


def serve(port: int) -> None:
    import uvicorn

    from bms.main import app

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--browser", action="store_true", help="强制用系统浏览器打开")
    ap.add_argument("--width", type=int, default=1500)
    ap.add_argument("--height", type=int, default=950)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    port = free_port(args.port)
    url = f"http://127.0.0.1:{port}"
    print(f"[BadmintonStudio] 启动本地服务 {url}")

    t = threading.Thread(target=serve, args=(port,), daemon=True, name="bms-server")
    t.start()

    if not wait_ready(url):
        print("[BadmintonStudio] 服务启动失败，请检查依赖是否安装完整。")
        return 1
    print("[BadmintonStudio] 服务就绪")

    use_browser = args.browser
    if not use_browser:
        try:
            import webview  # type: ignore
        except Exception:
            webview = None  # type: ignore
            use_browser = True
            print("[BadmintonStudio] 未安装 pywebview，改用系统浏览器打开。")
            print("                  安装桌面窗口： pip install pywebview")

    if use_browser:
        import webbrowser

        webbrowser.open(url)
        print(f"[BadmintonStudio] 已打开 {url}（关闭本窗口即停止服务）")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return 0

    import webview  # type: ignore

    window = webview.create_window(
        "羽毛球智能剪辑台 · Badminton Studio",
        url,
        width=args.width,
        height=args.height,
        min_size=(1100, 700),
        background_color="#06080b",
        text_select=False,
    )
    webview.start(debug=args.debug, private_mode=False, storage_path=str(ROOT / "data" / "webview"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
