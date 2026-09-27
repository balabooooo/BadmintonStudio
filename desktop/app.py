"""BadmintonStudio 桌面外壳。

把本地 FastAPI 服务跑在一个后台线程里，再用 pywebview（Windows 上是
WebView2）开一个无浏览器边框的原生窗口，观感就是一个普通桌面软件。

用法::

    python desktop/app.py [--port 8000] [--browser]

没有安装 pywebview 时自动回退成「启动服务 + 打开系统默认浏览器」。
"""

from __future__ import annotations

import argparse
import json
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


def frontend_version() -> str:
    """Cache-bust token for the webview URL, derived from the built frontend's mtime.

    The desktop window uses a persistent WebView2 cache; without a changing document URL it can
    keep rendering a bundle from a previous build. The backend also sends ``no-store`` for
    index.html, so this is belt-and-suspenders.
    """
    try:
        return str(int((ROOT / "frontend" / "dist" / "index.html").stat().st_mtime))
    except OSError:
        return "0"


def serve(port: int) -> None:
    import uvicorn

    from bms.logging_setup import setup_logging
    from bms.main import app

    setup_logging()
    # log_config=None keeps uvicorn from installing its own stdlib handlers, so uvicorn records
    # propagate to loguru through the root interceptor installed by setup_logging().
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning",
                access_log=False, log_config=None)


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
    page_url = f"{url}/?v={frontend_version()}"
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

        webbrowser.open(page_url)
        print(f"[BadmintonStudio] 已打开 {page_url}（关闭本窗口即停止服务）")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return 0

    import webview  # type: ignore

    window = webview.create_window(
        "羽毛球智能剪辑台 · Badminton Studio",
        page_url,
        width=args.width,
        height=args.height,
        min_size=(1100, 700),
        background_color="#06080b",
        text_select=False,
    )
    _enable_native_file_drop(window)
    webview.start(debug=args.debug, private_mode=False, storage_path=str(ROOT / "data" / "webview"))
    return 0


def _enable_native_file_drop(window) -> None:
    """把拖进窗口的文件按「完整本地路径」交给前端，而不是让浏览器复制上传。

    浏览器出于安全拿不到拖入文件的路径，只有内容（大文件会被复制一份）。pywebview
    5.0+ 在 drop 事件的 ``domTransfer.files[].pywebviewFullPath`` 里给出真实路径，
    桌面窗口因此可以像「选择视频」一样零拷贝导入，大文件也能拖。

    任何一步失败（pywebview 太老 / DOM 事件不可用）都不影响启动：前端检测不到
    ``__bmsNativeDropReady`` 时会自动退回浏览器上传。
    """

    def on_drop(event) -> None:
        # 官方文档写的是 domTransfer，但 pywebview 6.x 实际把 DataTransfer 序列化成
        # dataTransfer。两个键都认，免得升级/降级时静默收不到路径。
        try:
            ev = event or {}
            dt = ev.get("dataTransfer") or ev.get("domTransfer") or {}
            files = dt.get("files") or []
        except Exception:
            files = []
        paths: list[str] = []
        for f in files:
            try:
                p = f.get("pywebviewFullPath")
            except Exception:
                p = None
            if p:
                paths.append(p)
        if not paths:
            return
        payload = json.dumps(paths, ensure_ascii=False)
        try:
            window.evaluate_js(
                f"window.__bmsNativeDrop && window.__bmsNativeDrop({payload})"
            )
        except Exception:
            pass

    def on_loaded() -> None:
        try:
            window.dom.document.events.drop += on_drop
            # 告诉前端「原生按路径拖拽已就绪」：JS 的 drop 兜底逻辑据此让路，避免重复导入。
            window.evaluate_js("window.__bmsNativeDropReady = true")
        except Exception as e:  # noqa: BLE001
            print(f"[BadmintonStudio] 原生拖拽不可用，拖入将退回浏览器上传：{e}")

    try:
        window.events.loaded += on_loaded
    except Exception as e:  # noqa: BLE001
        print(f"[BadmintonStudio] 注册原生拖拽事件失败：{e}")


if __name__ == "__main__":
    raise SystemExit(main())
