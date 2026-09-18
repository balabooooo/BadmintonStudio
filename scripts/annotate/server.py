"""回合标注工具：本地 Web UI（零前端构建）。

用法::

    python scripts/annotate/server.py \
        --video "data/cache/proxies/20260913_羽毛球_clip9_m_aa6b31f616ee_960x540.mp4" \
        --auto  data/cache/frames/analysis_new.json

然后浏览器打开 http://127.0.0.1:8765 。

功能：
* 播放 / 逐帧步进 / 变速 / 循环；
* 键盘快捷键快速标出每回合的起止（``i`` / ``o``）；
* 把自动切分结果当作草稿，一键「确认当前自动回合」（``c``）快速半自动标注；
* 时间轴拖动边界、缩放（滚轮 / 按钮）、跳转；
* 自动保存到 ``data/annotations/<视频名>.anno.json``，可导出 CSV。

所有标注只写入 ``data/``（已被 .gitignore 忽略），不会改动源码。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from fastapi import Body, FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse  # noqa: E402
from bms.core.streaming import range_response  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_PROXIES = ROOT / "data" / "cache" / "proxies"
DEFAULT_AUTO = ROOT / "data" / "cache" / "frames" / "analysis_new.json"
DEFAULT_OUTDIR = ROOT / "data" / "annotations"

app = FastAPI(title="BadmintonStudio 回合标注", docs_url=None, openapi_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CFG: dict = {
    "video": None,
    "auto": None,
    "out": None,
    "focus": (0.0, 0.0),
    "index_html": HERE / "index.html",
}


# ---------------------------------------------------------------- 工具函数


def probe_video(path: Path) -> dict:
    """用 OpenCV 读取时长 / 帧率 / 分辨率（缺 cv2 时退回 0）。"""
    info = {"duration": 0.0, "fps": 0.0, "width": 0, "height": 0, "frames": 0}
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            info["fps"] = round(fps, 4)
            info["frames"] = frames
            info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            info["duration"] = round(frames / fps, 3) if fps > 0 else 0.0
        cap.release()
    except Exception:
        pass
    return info


def load_auto(path: Path | None) -> dict:
    """读取分析 JSON，取出自动切分的回合。"""
    if not path or not path.is_file():
        return {"rallies": [], "duration": 0.0, "fps": 0.0, "media_id": ""}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 无法读取 {path}: {e}")
        return {"rallies": [], "duration": 0.0, "fps": 0.0, "media_id": ""}
    rallies = []
    for r in data.get("rallies") or []:
        try:
            start = float(r["start"])
            end = float(r["end"])
        except Exception:
            continue
        rallies.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "shots": len(r.get("shots") or []),
            "score": round(float((r.get("scores") or {}).get("total", 0.0)), 1),
        })
    rallies.sort(key=lambda x: x["start"])
    sig = data.get("signals") or {}
    duration = float((sig.get("duration") or [0.0])[0] or 0.0)
    fps = float((sig.get("fps") or [0.0])[0] or 0.0)
    return {
        "rallies": rallies,
        "duration": duration,
        "fps": fps,
        "media_id": str(data.get("media_id") or ""),
    }


def find_default_video(media_id: str = "") -> Path | None:
    if not DEFAULT_PROXIES.is_dir():
        return None
    cands = [p for p in DEFAULT_PROXIES.glob("*.mp4") if p.is_file()]
    if not cands:
        return None
    if media_id:
        hit = [p for p in cands if media_id in p.name and "960x540" in p.name]
        if hit:
            return max(hit, key=lambda p: p.stat().st_size)
    # 优先 960x540 主代理，且避免再次派生（名字里有两个 _m_）
    main = [p for p in cands if p.name.endswith("_960x540.mp4") and "x540_m_" not in p.name]
    pool = main or cands
    return max(pool, key=lambda p: p.stat().st_size)


def annotations_path() -> Path:
    if CFG["out"]:
        return Path(CFG["out"])
    stem = Path(CFG["video"]).stem
    return DEFAULT_OUTDIR / f"{stem}.anno.json"


def load_annotations() -> dict:
    p = annotations_path()
    if not p.is_file():
        return {"rallies": [], "focus": None, "note": ""}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"rallies": [], "focus": None, "note": ""}
    data.setdefault("rallies", [])
    return data


def save_annotations(payload: dict) -> int:
    p = annotations_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    rallies = []
    for r in payload.get("rallies") or []:
        try:
            start = max(0.0, float(r["start"]))
            end = float(r["end"])
        except Exception:
            continue
        if end <= start:
            continue
        rallies.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "note": str(r.get("note") or ""),
            "source": str(r.get("source") or "manual"),
        })
    rallies.sort(key=lambda x: x["start"])
    video = Path(CFG["video"])
    doc = {
        "video": str(video),
        "video_name": video.name,
        "duration": CFG["probe"]["duration"],
        "fps": CFG["probe"]["fps"],
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(rallies),
        "focus": payload.get("focus") or CFG.get("focus"),
        "note": str(payload.get("note") or ""),
        "rallies": rallies,
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)
    return len(rallies)


# ---------------------------------------------------------------- 路由


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = CFG["index_html"].read_text(encoding="utf-8")
    banner = (
        f"<script>window.__BMS__ = {json.dumps({'videoName': Path(CFG['video']).name, 'hasAuto': bool(CFG['auto'])}, ensure_ascii=False)};</script>"
    )
    return HTMLResponse(html.replace("<!--BANNER-->", banner))


@app.get("/api/info")
def api_info() -> JSONResponse:
    auto = load_auto(CFG["auto"])
    ann = load_annotations()
    return JSONResponse({
        "videoName": Path(CFG["video"]).name,
        "probe": CFG["probe"],
        "auto": auto["rallies"],
        "autoMeta": {"media_id": auto["media_id"], "duration": auto["duration"], "fps": auto["fps"]},
        "annotations": ann["rallies"],
        "focus": ann.get("focus") or list(CFG["focus"]),
        "note": ann.get("note", ""),
        "out": str(annotations_path()),
    })


@app.get("/api/video")
def api_video(request: Request):
    return range_response(request, Path(CFG["video"]), cache_seconds=3600)


@app.post("/api/save")
def api_save(payload: dict = Body(...)) -> JSONResponse:
    n = save_annotations(payload)
    return JSONResponse({"ok": True, "count": n, "path": str(annotations_path())})


@app.get("/api/export.csv", response_class=PlainTextResponse)
def api_export_csv() -> PlainTextResponse:
    ann = load_annotations()
    fps = CFG["probe"]["fps"] or 0.0
    lines = ["index,start,end,duration,start_frame,end_frame,source,note"]
    for i, r in enumerate(ann.get("rallies") or [], 1):
        s, e = float(r["start"]), float(r["end"])
        sf = int(round(s * fps)) if fps else ""
        ef = int(round(e * fps)) if fps else ""
        note = str(r.get("note") or "").replace(",", " ")
        lines.append(f"{i},{s:.3f},{e:.3f},{e - s:.3f},{sf},{ef},{r.get('source','')},{note}")
    return PlainTextResponse("\n".join(lines), media_type="text/csv",
                             headers={"Content-Disposition": 'attachment; filename="rally_annotations.csv"'})


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------- 启动


def main() -> None:
    ap = argparse.ArgumentParser(description="BadmintonStudio 回合标注工具")
    ap.add_argument("--video", default=None, help="要标注的视频（默认取 data/cache/proxies 里最大的 960x540 代理）")
    ap.add_argument("--auto", default=str(DEFAULT_AUTO), help="自动切分结果 JSON（作草稿）")
    ap.add_argument("--out", default=None, help="标注输出路径（默认 data/annotations/<视频名>.anno.json）")
    ap.add_argument("--focus", nargs=2, type=float, default=None, metavar=("START", "END"),
                    help="建议标注的时间窗（秒），仅影响初始视图")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    auto_meta = load_auto(Path(args.auto) if args.auto else None)
    video = Path(args.video).expanduser() if args.video else find_default_video(auto_meta["media_id"])
    if not video or not video.is_file():
        print("找不到视频，请用 --video 指定。候选：")
        for p in sorted(DEFAULT_PROXIES.glob("*.mp4")):
            print("   ", p)
        raise SystemExit(2)

    CFG["video"] = str(video.resolve())
    CFG["auto"] = Path(args.auto).resolve() if args.auto and Path(args.auto).is_file() else None
    CFG["out"] = str(Path(args.out).resolve()) if args.out else None
    CFG["probe"] = probe_video(video)
    duration = CFG["probe"]["duration"] or auto_meta["duration"] or 0.0
    CFG["probe"]["duration"] = duration
    if args.focus:
        CFG["focus"] = (max(0.0, args.focus[0]), min(duration, args.focus[1]))
    else:
        CFG["focus"] = (0.0, duration)

    print("=" * 68)
    print("BadmintonStudio 回合标注工具")
    print(f"  视频 : {CFG['video']}")
    print(f"  时长 : {duration:.1f}s   {CFG['probe']['fps']:.3f} fps   "
          f"{CFG['probe']['width']}x{CFG['probe']['height']}")
    print(f"  自动 : {CFG['auto'] or '(无，纯手动)'}")
    if CFG["auto"]:
        print(f"         {len(auto_meta['rallies'])} 个自动回合（作为草稿）")
    print(f"  输出 : {annotations_path()}")
    print("=" * 68)
    print(f"  浏览器打开： http://{args.host}:{args.port}")

    if not args.no_browser:
        try:
            import threading
            import webbrowser

            threading.Timer(1.2, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()
        except Exception:
            pass

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
