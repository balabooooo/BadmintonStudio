"""The most basic regression tests: covering the places where "silent failures" are most likely.

Run with ``python -m pytest tests`` or directly with ``python tests/test_core.py``.
Deliberately avoids any pytest-specific features, so it also runs when executed directly.

Why only these: the accuracy of the speech/vision algorithms cannot be guaranteed by
unit tests, but **interface mismatches** can be -- and those are exactly the most
dangerous, because a failure in ``analyze_players`` gets swallowed by the pipeline's
try/except and only written to player_trace; the UI looks perfectly fine, the results
just get worse.
"""

from __future__ import annotations

import copy
import inspect
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from bms.analysis import court_calib as CC          # noqa: E402
from bms.analysis import pipeline as P             # noqa: E402
from bms.analysis import players as PL             # noqa: E402
from bms.analysis import rally as RA               # noqa: E402
from bms.analysis import rally_vision as RV        # noqa: E402
from bms.analysis import annotation as AN          # noqa: E402
from bms.core.models import AnalysisParams, AnalysisResult  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


def _find_ffmpeg() -> str:
    """Resolve ffmpeg the same way the app does, for the sample-synthesis test.

    Falls back to the in-project tools path (without checking existence) so a
    missing ffmpeg cannot break the whole suite at import time.
    """
    try:
        from bms.core.ffmpeg import find_ffmpeg
        return find_ffmpeg()
    except FileNotFoundError:
        return str(ROOT / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe")


FFMPEG = _find_ffmpeg()


# ------------------------------------------------------------------ Interface consistency


def test_player_pipeline_contract() -> None:
    """Every keyword argument the pipeline uses when calling the player module must exist in the signature.

    This one was bought with a real bug: ``_select_active_players`` gained a ``boxes``
    parameter but the signature was not updated, so player detection threw a TypeError
    that was eaten by the pipeline's try/except, resulting in "analysis succeeded but no
    player was detected at all" and completely broken rally boundaries.
    """
    print("\n接口一致性")
    sig = inspect.signature(PL.analyze_players)
    for kw in ("roi", "roi_poly", "size_filter", "max_seconds", "viewpoint",
               "on_progress", "cancel"):
        check(f"analyze_players 接受 {kw}", kw in sig.parameters)
    sel = inspect.signature(PL._select_active_players)
    for kw in ("viewpoint", "boxes", "aspect", "size_filter"):
        check(f"_select_active_players 接受 {kw}", kw in sel.parameters)
    probe = inspect.signature(PL.probe_boxes)
    for kw in ("count", "viewpoint", "roi_poly", "roi", "times", "save_frames"):
        check(f"probe_boxes 接受 {kw}", kw in probe.parameters)


def test_shuttle_pipeline_contract() -> None:
    from bms.analysis import shuttle as SH

    sig = inspect.signature(SH.analyze_shuttle)
    for kw in ("roi", "max_seconds", "sample_fps", "on_progress", "cancel"):
        check(f"analyze_shuttle 接受 {kw}", kw in sig.parameters)


def test_bytetrack_tracker() -> None:
    """Interface and behavior after player tracking switched to ultralytics' built-in ByteTrack.

    This one was bought with a real failure: if ``_ByteTracker.update(frame_idx, time, dets)``
    is mismatched with the calling convention of :func:`analyze_players`, the exception gets
    swallowed by the pipeline's try/except; the UI looks perfectly normal, except one player
    is silently split into several tracks. This also pins down "reconnecting to the same track
    after a low-confidence (occluded) frame".
    """
    print("\nByteTrack 跟踪适配器")
    if not hasattr(PL, "_ByteTracker"):
        check("存在 _ByteTracker 适配器", False)
        return
    try:
        import ultralytics  # noqa: F401
        from ultralytics.trackers.utils import matching  # noqa: F401  trigger the lap check
    except Exception as exc:  # Dependency is optional: degrade silently when missing, so skip here
        print(f"  skip  未安装 ultralytics/lap（{type(exc).__name__}: {exc}）")
        return

    sig = inspect.signature(PL._ByteTracker.update)
    check("update 接受 frame_idx/time/dets",
          all(k in sig.parameters for k in ("frame_idx", "time", "dets")))

    aspect = 16 / 9
    tr = PL._ByteTracker(fps=15.0, aspect=aspect)

    def person(cx, cy, conf, h=0.2, w=0.09):
        return ((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), conf)

    # Frames 8~11 drop below BT_HIGH_THRESH (simulating occlusion) and should be reconnected by stage two.
    for f in range(25):
        conf = 0.18 if 8 <= f <= 11 else 0.9
        tr.update(f, f / 15.0, [person(0.30 + 0.005 * f, 0.55, conf)])

    check("低分遮挡后仍是同一条轨迹", len(tr.tracks) == 1, f"tracks={len(tr.tracks)}")
    if tr.tracks:
        t = tr.tracks[0]
        check("轨迹覆盖遮挡前后", t.frames[0] == 0 and t.frames[-1] == 24,
              f"frames={t.frames[0]}..{t.frames[-1]}")
        check("框全部落在 [0,1]",
              all(0.0 <= v <= 1.0 for b in t.boxes for v in b))


# ------------------------------------------------------------------ Size filtering


def test_box_size_stats() -> None:
    """Size stats must give a sensible reference when "players exist" and degrade gracefully when "no players"."""
    print("\n框尺寸统计")
    boxes = [[(0, 0.10, 0.45, 0.22, 0.75)] for _ in range(20)]
    st = PL._box_size_stats(boxes, 1.7778)
    check("有球员时参考尺度在 0.05~0.5 之间", 0.05 < st["ref"] < 0.5, str(st))
    check("小目标下限为正", st["min_abs"] > 0, str(st))
    check("空输入退化", PL._box_size_stats([], 1.7778)["ref"] == 0.0)
    # In the pipeline det_frames is a **quadruple** (x1,y1,x2,y2); only frame_boxes carries a track id.
    # An implementation accepting only quintuples makes ref always 0 and silently breaks the adaptive threshold -- this case pins it down.
    boxes4 = [[(0.10, 0.45, 0.22, 0.75)] for _ in range(20)]
    st4 = PL._box_size_stats(boxes4, 1.7778)
    check("四元组框也能算出参考尺度", st4["ref"] > 0.05, str(st4))
    check("两种表示给出同一个参考尺度", abs(st4["ref"] - st["ref"]) < 1e-9, f"{st4} vs {st}")
    check("框表示归一化：四元组", PL._box_xyxy((0.1, 0.2, 0.3, 0.4)) == (0.1, 0.2, 0.3, 0.4))
    check("框表示归一化：五元组（带轨迹号）",
          PL._box_xyxy((7, 0.1, 0.2, 0.3, 0.4)) == (0.1, 0.2, 0.3, 0.4))
    check("坏输入返回 None", PL._box_xyxy((0.1, 0.2)) is None)


# ------------------------------------------------------------------ Person box size filtering


def test_size_filter() -> None:
    """The two size-filter modes + area lower/upper bounds + stats histogram.

    This is the entry point for "user-specified thresholds"; a wrong decision **silently**
    filters out real players (the analysis still completes, the rallies are just all messed up),
    so every mode needs a test case pinning it down.
    """
    print("\n人物框尺寸筛选")
    small = (0.40, 0.30, 0.45, 0.34)      # box height 0.04 (spectator / person far away)
    big = (0.40, 0.40, 0.60, 0.62)        # box height 0.22 (player)

    off = PL.size_filter_from({"mode": "off"})
    check("off 时不做筛选", off.keep(small, 0.22) and off.keep(big, 0.22) and not off.active)

    absf = PL.size_filter_from({"mode": "absolute", "min_height": 0.10})
    check("absolute：低于绝对下限的被筛掉", not absf.keep(small, 0.22))
    check("absolute：够大的保留", absf.keep(big, 0.22))

    absu = PL.size_filter_from({"mode": "absolute", "min_height": 0.0, "max_height": 0.15})
    check("absolute：超过上限的被筛掉", not absu.keep(big, 0.22))
    check("absolute：上限内保留", absu.keep(small, 0.22))

    rel = PL.size_filter_from({"mode": "relative", "min_height": 0.5})
    # same-frame max box height 0.22 -> effective threshold 0.11
    check("relative：同帧参考下小框被筛掉", not rel.keep(small, 0.22))
    check("relative：同帧参考下大框保留", rel.keep(big, 0.22))
    check("relative：没有可信参考时不做判断", rel.keep(small, 0.0))

    areaf = PL.size_filter_from({"mode": "absolute", "min_height": 0.0, "max_area": 0.01})
    check("面积上限生效", not areaf.keep(big, 0.22))
    check("面积上限内的保留", areaf.keep(small, 0.22))

    bad = PL.size_filter_from({"mode": "乱写", "min_height": 0.1})
    check("非法模式退化成 off", bad.mode == "off" and not bad.active)
    check("None 退化成 off", not PL.size_filter_from(None).active)
    check("已经是 SizeFilter 时原样返回", PL.size_filter_from(absf) is absf)

    # Stats: the histogram must cover **all** boxes before filtering (otherwise the UI cannot see what was cut)
    samples = [[0.04, 0.0025, 0.22], [0.04, 0.0025, 0.22], [0.22, 0.044, 0.22]]
    stats = PL._build_size_stats(absf, samples, 2, 2, 0.22)
    check("统计里的总数是筛选前的框数", stats["total"] == 3, str(stats["total"]))
    check("统计里的筛掉数正确", stats["dropped"] == 2 and stats["kept"] == 1, str(stats))
    check("直方图总和等于框数", sum(stats["hist"]) + stats["overflow"] == 3, str(stats["hist"]))
    check("样本是 [框高, 框面积, 同帧参考] 三元组",
          all(len(s) == 3 for s in stats["sample"]), str(stats["sample"]))
    check("空输入不炸", PL._build_size_stats(absf, [], 0, 0, 0.0)["total"] == 0)

    med_box = PL._median_box(PL.PlayerTrack(1, [], [], [(0.1, 0.1, 0.2, 0.2),
                                                        (0.1, 0.1, 0.2, 0.4)], [], []))
    check("轨迹代表框取逐坐标中位数", abs(med_box[3] - 0.3) < 1e-6, str(med_box))


def test_probe_frame_capture() -> None:
    """Probe frame capture: capturing at a given time must skip the black frame after a seek,
    and the frame image path must be stable and unique.

    This guards the "grab a frame in the UI to check the box selection" feature: if a black
    frame is captured, the user sees boxes drawn on a black field and cannot verify anything
    (and will think they marked it wrong themselves).
    """
    print("\n试测取帧（手动选帧）")
    p1 = PL._probe_frame_path("D:/v/a.mp4", 12.0, 0)
    p2 = PL._probe_frame_path("D:/v/a.mp4", 12.0, 0)
    p3 = PL._probe_frame_path("D:/v/a.mp4", 12.5, 0)
    p4 = PL._probe_frame_path("D:/v/b.mp4", 12.0, 0)
    check("同一素材同一时刻路径稳定", p1 == p2)
    check("不同时刻路径不同", p1 != p3)
    check("不同素材路径不同", p1 != p4)
    check("帧图落在缓存 frames 目录下", "frames" in str(p1).replace("\\", "/"))

    try:
        import cv2
        import tempfile
    except ImportError:  # pragma: no cover
        check("opencv 可用（跳过取帧用例）", False)
        return
    tmp = Path(tempfile.mkdtemp(prefix="bms_probe_"))
    vid = tmp / "clip.avi"
    # Synthesize a video: the first 5 frames are deliberately all black (simulating black frames returned after seek), followed by bright frames
    w, h = 160, 96
    try:
        vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (w, h))
        if not vw.isOpened():
            print("  skip 本机 OpenCV 不能写 MJPG 视频，跳过取帧用例")
            return
        rng = np.random.RandomState(0)
        for i in range(20):
            if i < 5:
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                frame = rng.randint(40, 200, (h, w, 3)).astype(np.uint8)
            vw.write(frame)
        vw.release()
    except Exception as e:  # pragma: no cover
        print(f"  skip 合成视频失败（{e}），跳过取帧用例")
        return

    cap = cv2.VideoCapture(str(vid))
    check("合成视频可读", cap.isOpened())
    if cap.isOpened():
        fr = PL._read_frame_at(cap, 10.0, 0.0)          # frame 0 is a black frame
        check("黑帧被跳过（拿到的是非纯色帧）",
              fr is not None and float(fr.std()) > 2.0,
              f"std={0.0 if fr is None else float(fr.std()):.2f}")
        fr2 = PL._read_frame_at(cap, 10.0, 1.5)         # frame 15 is normal
        check("正常时刻能取到帧", fr2 is not None and float(fr2.std()) > 2.0)
        fr3 = PL._read_frame_at(cap, 10.0, 999.0)       # beyond the clip length
        check("超出片长不抛异常", fr3 is None or float(fr3.std()) >= 0.0)
    cap.release()


def test_size_filter_selects_end_to_end() -> None:
    """Actually call _select_active_players with size_filter: the threshold must really take effect.

    An interface mismatch / a threshold not wired into candidate filtering cannot be caught
    at import time, and the consequence is "the user dragged the threshold but nothing changed".
    """
    print("\n尺寸筛选（真实调用）")
    tracks: list[PL.PlayerTrack] = []
    for tid, h, speed in ((1, 0.20, 0.05), (2, 0.05, 0.05)):
        n = 200
        tr = PL.PlayerTrack(
            track_id=tid,
            frames=list(range(n)),
            times=[i / 12.0 for i in range(n)],
            boxes=[(0.4, 0.4, 0.4 + h * 0.5, 0.4 + h)] * n,
            speeds=[speed] * n,
            confidences=[0.9] * n,
        )
        tr.mean_area = 0.5 * h * h
        tr.max_area = 0.5 * h * h
        tr.mean_speed = speed
        tr.max_speed = speed * 2
        tr.total_travel = speed * n
        tracks.append(tr)
    boxes = [[(1, 0.4, 0.4, 0.5, 0.6), (2, 0.4, 0.4, 0.45, 0.45)]] * 60

    sf = PL.size_filter_from({"mode": "absolute", "min_height": 0.12})
    picked = PL._select_active_players(tracks, 12.0, 200.0, viewpoint="rear",
                                       boxes=boxes, aspect=1.7778, size_filter=sf)
    ids = [t.track_id for t in picked]
    check("尺寸下限只留下够大的轨迹", ids == [1], str(ids))

    # Conversely, cap the upper bound: the threshold really is used -- and it can override the "adaptive size threshold"
    # (the adaptive scheme only cuts small tracks and can never cut a large one)
    upper = PL.size_filter_from({"mode": "absolute", "min_height": 0.0, "max_height": 0.12})
    ids2 = [t.track_id for t in PL._select_active_players(
        tracks, 12.0, 200.0, viewpoint="rear", boxes=boxes, aspect=1.7778, size_filter=upper)]
    check("尺寸上限能把大轨迹筛掉（阈值真的生效）", 1 not in ids2, str(ids2))


# ------------------------------------------------------------------ Court polygon


def test_polygon_geometry() -> None:
    """Polygon geometry: ring order, area, fitted quadrilateral, in-court test."""
    print("\n场地多边形几何")
    # A hexagon with "near edge bulging down, far edge bulging up": the typical shape for panoramic / fisheye footage
    hexa = np.array([[0.10, 0.96], [0.50, 1.00], [0.90, 0.96],
                     [0.80, 0.40], [0.50, 0.30], [0.20, 0.40]], dtype=np.float32)
    ring = CC.order_poly(hexa, aspect=1.6)
    check("环序点数不变", ring.shape == hexa.shape)
    check("环序起点是最靠近画面的点（y 最大且最靠左）",
          abs(ring[0][1] - 0.96) < 1e-6 and abs(ring[0][0] - 0.10) < 1e-6, str(ring[0]))
    same = {tuple(np.round(p, 5)) for p in ring} == {tuple(np.round(p, 5)) for p in hexa}
    check("环序不增删点", same)

    quad = CC.order_poly(np.array([[0.9, 0.98], [0.5, 0.3], [0.1, 0.98], [0.5, 0.4]],
                                  dtype=np.float32), aspect=1.6)  # shuffled quadrilateral
    expect = CC.order_quad(np.array([[0.1, 0.98], [0.9, 0.98], [0.5, 0.4], [0.5, 0.3]],
                                    dtype=np.float32))
    check("四点输入与 order_quad 结果一致",
          np.allclose(np.asarray(quad, dtype=np.float32), expect, atol=1e-5),
          f"{np.asarray(quad).tolist()} vs {expect.tolist()}")

    fitted = CC.fit_quad_from_poly(hexa)
    check("六边形能拟合出四边形", fitted is not None and fitted.shape == (4, 2))
    if fitted is not None:
        on_poly = all(any(np.allclose(f, p, atol=1e-5) for p in hexa) for f in fitted)
        check("拟合点取自多边形顶点", on_poly)
        area = CC._contour_area(fitted)
        best_other = max(CC._contour_area(hexa[[a, b, c, d]])
                         for a in range(6) for b in range(a + 1, 6)
                         for c in range(b + 1, 6) for d in range(c + 1, 6))
        check("拟合四边形是面积最大的那个", abs(area - best_other) < 1e-6,
              f"{area:.5f} vs {best_other:.5f}")

    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    check("单位方框面积为 1", abs(CC.poly_area_norm(square) - 1.0) < 1e-6)
    check("三角形面积正确",
          abs(CC.poly_area_norm(np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)) - 0.5) < 1e-6)

    inside = CC.point_in_poly(
        np.array([[0.5, 0.5], [1.5, 0.5], [-0.01, 0.5], [0.99, 0.99]], dtype=np.float32), square)
    check("多边形内 / 外判定正确", inside.tolist() == [True, False, False, True], str(inside.tolist()))
    # Expansion: a point just outside the sideline also counts as in-court under margin (a player standing on the line must not be missed)
    edge = CC.point_in_poly(np.array([[0.5, 1.004]], dtype=np.float32), square)
    check("精确判定：边线外侧算场外", edge.tolist() == [False], str(edge.tolist()))
    edge_m = CC.point_in_poly(np.array([[0.5, 1.004]], dtype=np.float32), square, margin=0.02)
    check("带外扩后边线外侧算场内", edge_m.tolist() == [True], str(edge_m.tolist()))
    # Curved-edge hexagon: the two corners of the bounding rectangle must be judged "out of court" -- this is exactly the value of the polygon representation
    corner = CC.point_in_poly(np.array([[0.02, 0.02], [0.98, 0.02], [0.5, 0.65]],
                                       dtype=np.float32), hexa)
    check("弯边以外（矩形内）的点被判成场外", corner[0] == False and corner[1] == False,  # noqa: E712
          str(corner.tolist()))


def test_polygon_calibration() -> None:
    """Polygon calibration: homography, distortion metric, ROI, payload fields."""
    print("\n场地多边形标定")
    fw, fh = 1920, 1080
    hexa = np.array([[0.10, 0.96], [0.50, 1.00], [0.90, 0.96],
                     [0.80, 0.40], [0.50, 0.30], [0.20, 0.40]], dtype=np.float32)
    cal = CC.build_calibration(hexa * np.array([fw, fh], dtype=np.float32), fw, fh, source="manual")
    check("六边形标定成功", cal.ok)
    check("保留全部多边形顶点", len(cal.polygon_norm) == 6, str(len(cal.polygon_norm)))
    check("仍给出四点单应", cal.quad_norm is not None and len(cal.quad_norm) == 4)
    check("畸变度量 > 0（弯边）", cal.distortion > 0.02, f"{cal.distortion:.3f}")
    check("机位判定用的 foreshortening 有值", 0.0 < cal.foreshortening <= 1.0)
    roi = cal.court_mask_roi(0.0)
    check("ROI 覆盖整个多边形", roi is not None and roi[0] <= 0.101 and roi[2] >= 0.899, str(roi))

    got = cal.contains(np.array([[0.5, 0.6], [0.05, 0.05]], dtype=np.float32))
    check("contains：场内 / 场外判定正确", got.tolist() == [True, False], str(got.tolist()))
    got2 = cal.contains_box_bottom(np.array([[0.45, 0.5, 0.55, 0.62], [0.0, 0.0, 0.1, 0.05]]))
    check("contains_box_bottom：按底边中心判定", got2.tolist() == [True, False], str(got2.tolist()))

    payload = cal.as_payload()
    for key in ("polygon", "quad", "point_count", "source", "distortion"):
        check(f"payload 带 {key}", key in payload)
    check("payload 的 source 是 manual", payload["source"] == "manual")
    check("payload 的 polygon 点数正确", len(payload["polygon"]) == 6)

    # Degenerate input must be blocked (otherwise the ROI shrinks to a point and filters out everyone)
    tiny = np.array([[0.5, 0.5], [0.501, 0.5], [0.501, 0.501], [0.5, 0.501]], dtype=np.float32)
    bad = CC.build_calibration(tiny * np.array([fw, fh], dtype=np.float32), fw, fh)
    check("面积过小的多边形被拒", not bad.ok, str(bad.notes))
    tri = CC.build_calibration(np.array([[100, 900], [900, 900], [500, 200]], dtype=np.float32),
                               fw, fh)
    check("少于四点的多边形被拒", not tri.ok)

    # Auto path: a curved-edge mask should fit more than 4 points (instead of being forced into a quadrilateral)
    try:
        import cv2

        mask = np.zeros((540, 960), dtype=np.uint8)
        pts: list[list[float]] = []
        for i in range(41):                      # near edge: bulge downward
            t = i / 40.0
            pts.append([60 + t * 840, 480 + 40 * np.sin(np.pi * t)])
        for i in range(41):                      # right edge: bulge rightward
            t = i / 40.0
            pts.append([900 + 40 * np.sin(np.pi * t), 480 - t * 380])
        for i in range(41):                      # far edge: bulge upward
            t = 1.0 - i / 40.0
            pts.append([900 - t * 840, 100 - 40 * np.sin(np.pi * t)])
        for i in range(41):                      # left edge: bulge leftward
            t = 1.0 - i / 40.0
            pts.append([60 - 40 * np.sin(np.pi * t), 100 + t * 380])
        cv2.fillPoly(mask, [np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)], 255)
        poly = CC.find_court_poly(mask, min_area_ratio=0.02)
        check("弯边 mask 能拟合出边界", poly is not None and poly.shape[0] >= 4)
        if poly is not None:
            check("弯边被保留成多边形（不是硬压成四边）", poly.shape[0] > 4,
                  f"点数 {poly.shape[0]}")
            check("边界面积与 mask 接近",
                  CC._contour_area(poly) > 0.9 * float(np.count_nonzero(mask)))
        quad = CC.find_court_quad(mask, min_area_ratio=0.02)
        check("旧的 find_court_quad 仍然可用", quad is not None and quad.shape == (4, 2))

        # Conversely: a straight-edge court must still yield only 4 points (otherwise the calibration
        # the downstream gets becomes needlessly complex, and "auto-added points" just add noise to every clip)
        straight = np.zeros((540, 960), dtype=np.uint8)
        cv2.fillPoly(straight, [np.array([[120, 520], [840, 520], [700, 120], [260, 120]],
                                         dtype=np.int32)], 255)
        poly2 = CC.find_court_poly(straight, min_area_ratio=0.02)
        check("直边场地仍然拟合成四边形", poly2 is not None and poly2.shape[0] == 4,
              f"点数 {0 if poly2 is None else poly2.shape[0]}")
    except ImportError:  # pragma: no cover - environment without opencv installed
        check("opencv 可用（跳过弯边拟合用例）", False)


def test_select_active_players_signature_end_to_end() -> None:
    """Actually call _select_active_players: an interface mismatch cannot be caught at import time."""
    print("\n球员筛选（真实调用）")
    tracks: list[PL.PlayerTrack] = []
    for tid, h, speed in ((1, 0.18, 0.05), (2, 0.17, 0.04), (3, 0.03, 0.05)):
        n = 200
        tr = PL.PlayerTrack(
            track_id=tid,
            frames=list(range(n)),
            times=[i / 12.0 for i in range(n)],
            boxes=[(0.4, 0.5, 0.6, 0.5 + h)] * n,
            speeds=[speed] * n,
            confidences=[0.9] * n,
        )
        tr.mean_area = 0.2 * h
        tr.max_area = 0.2 * h
        tr.mean_speed = speed
        tr.max_speed = speed * 2
        tr.total_travel = speed * n
        tracks.append(tr)
    boxes = [[(1, 0.4, 0.5, 0.6, 0.68), (2, 0.4, 0.4, 0.6, 0.57)],
             [(1, 0.4, 0.5, 0.6, 0.68), (3, 0.4, 0.3, 0.5, 0.33)]] * 60
    picked = PL._select_active_players(tracks, 12.0, 200.0, viewpoint="rear",
                                      boxes=boxes, aspect=1.7778)
    ids = [t.track_id for t in picked]
    check("选出至少一名球员", len(ids) >= 1, str(ids))
    check("过小的轨迹被尺寸门限排除", 3 not in ids, str(ids))
    check("真实球员被选中", 1 in ids, str(ids))


# ------------------------------------------------------------------ Manual calibration


def test_manual_quad_parsing() -> None:
    print("\n手动标定多边形")
    good = [[0.1, 0.99], [0.9, 0.99], [0.8, 0.2], [0.2, 0.2]]
    check("合法四边形可用", P._manual_quad(good) is not None)
    check("点数不对被拒", P._manual_quad(good[:3]) is None)
    check("像素坐标被拒", P._manual_quad([[100, 900], [900, 900], [800, 200], [200, 200]]) is None)
    check("四点重合被拒", P._manual_quad([[0.5, 0.5]] * 4) is None)
    check("None 被拒", P._manual_quad(None) is None)
    # Multi-point (panoramic / fisheye) path
    hexa = [[0.10, 0.96], [0.50, 1.00], [0.90, 0.96], [0.80, 0.40], [0.50, 0.30], [0.20, 0.40]]
    check("六点多边形可用", P._manual_poly(hexa) is not None)
    check("五点可用", P._manual_poly(hexa[:5]) is not None)
    check("点数超过上限被拒", P._manual_poly(hexa + [[0.3, 0.5]] * 30) is None)
    check("环形但面积为零被拒", P._manual_poly([[0.5, 0.5], [0.5001, 0.5], [0.5001, 0.5001],
                                                [0.5, 0.5001]]) is None)
    cal = CC.build_calibration(np.asarray(good, dtype=np.float32) * np.array([960, 540], dtype=np.float32),
                               960, 540)
    check("手动四角能建出标定", cal.ok)
    check("标定给出 ROI", cal.court_mask_roi() is not None)
    check("params 能带上手动四角",
          AnalysisParams(court_quad=good).court_quad == good)
    check("params 能带多点多边形",
          len(AnalysisParams(court_poly=hexa).court_poly or []) == 6)
    check("多边形标定走的是多边形判定",
          CC.build_calibration(np.asarray(hexa, dtype=np.float32) * np.array([960, 540],
                                                                             dtype=np.float32),
                               960, 540).contains(np.array([[0.5, 0.62]]))[0])


# ------------------------------------------------------------------ Segmentation


def test_quiet_spans_and_segmentation() -> None:
    """Synthesize a "rally -> pause -> rally" clip; segmentation must produce two spans with a gap between them."""
    print("\n球员运动切分（合成信号）")
    fps = 12.0
    dur = 120.0
    n = int(dur * fps)
    m = np.zeros(n, dtype=np.float32)
    for a, b in ((10, 40), (60, 95)):
        m[int(a * fps):int(b * fps)] = 1.0
    m += np.random.RandomState(0).normal(0, 0.02, n).astype(np.float32)
    spans = RV.find_quiet_spans(m, fps, min_quiet=1.0, prominence_ratio=0.3)
    # This synthetic signal has one pause between 40~60s; the start and end are continuous motion,
    # so there is exactly 1 "quiet span" (the rallies at the head and tail are filled in by segment_by_player_motion).
    check("找到 40~60s 的停顿", len(spans) == 1, str([(round(s.start / fps, 1), round(s.end / fps, 1)) for s in spans]))
    if spans:
        check("停顿位置正确", 39 <= spans[0].start / fps <= 42 and 58 <= spans[0].end / fps <= 61,
              f"{spans[0].start / fps:.1f}-{spans[0].end / fps:.1f}")
    segs = RV.segment_by_player_motion(
        RV.SegmentSignals(fps=fps, duration=dur, player_motion=m, has_players=True),
        RV.SegmentOptions(min_rally=2.0, max_rally=120.0, pre_roll=1.0, post_roll=1.4),
    )
    check("切出 2 个回合", len(segs) == 2, str([(round(s.start, 1), round(s.end, 1)) for s in segs]))
    if len(segs) == 2:
        s = sorted(segs, key=lambda x: x.start)
        # The first rally should start near 10s and end near 40s (including padding)
        check("首回合起点落在 7~12s", 7 <= s[0].start <= 12, f"{s[0].start:.1f}")
        check("首回合终点落在 38~43s", 38 <= s[0].end <= 43, f"{s[0].end:.1f}")
        check("两回合之间有间隔（不相接）", s[1].start - s[0].end > 1.0,
              f"gap={s[1].start - s[0].end:.1f}")


def test_detection_coverage() -> None:
    print("\n球员检测覆盖率")
    fps = 12.0
    boxes: list[list] = [[] for _ in range(600)]
    for i in range(120, 360):
        boxes[i] = [(1, 0.4, 0.5, 0.6, 0.68)]
    cov = RV.detection_coverage(boxes, fps)
    check("长度一致", cov.size == 600)
    check("有球员的段落被判为有效", float(np.mean(cov[150:330])) > 0.9)
    check("空洞被填掉（<1.5s）", float(np.mean(cov[130:350])) > 0.9)
    check("长空洞保持无效", float(np.mean(cov[0:60])) < 0.1)
    check("覆盖率数值合理", abs(float(np.mean(cov > 0.5)) - 0.4) < 0.15,
          f"{float(np.mean(cov > 0.5)):.2f}")


def test_match_format_stats() -> None:
    """singles/doubles recommendation from per-frame active-player counts."""
    print("\n单/双打识别（仅推荐，不自动切预设）")
    rng = np.random.default_rng(1)
    n = 6000
    # Singles: 1-2 players with a few 3-player ID-split noise frames.
    singles = rng.integers(1, 3, n).astype(np.float32)
    singles[rng.random(n) < 0.02] = 3
    s = PL.match_format_stats(singles, fps=12.0)
    check("单打：format=single", s["format"] == PL.MATCH_FORMAT_SINGLE, s["format"])
    check("frac_ge3 处于噪声带", 0.0 <= s["frac_ge3"] <= PL.MATCH_FRAC_GE3_MAX_SINGLE + 0.03)
    # Doubles: 4 players most frames, 3 in some, detection gaps excluded by denominator.
    doubles = np.zeros(n, dtype=np.float32)
    doubles[:] = 4
    doubles[rng.random(n) < 0.25] = 3
    doubles[rng.random(n) < 0.1] = 0  # detection gaps
    d = PL.match_format_stats(doubles, fps=12.0)
    check("双打：format=doubles", d["format"] == PL.MATCH_FORMAT_DOUBLES, d["format"])
    check("检测空洞不计入分母", abs(d["present_frac"] - 0.9) < 0.03, str(d["present_frac"]))
    check("frac_ge4 证据充足", d["frac_ge4"] >= PL.MATCH_FRAC_GE4_DOUBLES - 0.03)
    # Ambiguous: three players present 40% of the time (between the two thresholds).
    amb = np.full(n, 2, dtype=np.float32)
    amb[rng.random(n) < 0.4] = 3
    a = PL.match_format_stats(amb, fps=12.0)
    check("ge3 证据落在模糊带 -> unknown", a["format"] == PL.MATCH_FORMAT_UNKNOWN,
          f"{a['format']} ge3={a['frac_ge3']}")
    # Insufficient evidence -> unknown.
    weak = PL.match_format_stats(np.full(100, 2, dtype=np.float32))
    check("证据帧数不足 -> unknown", weak["format"] == PL.MATCH_FORMAT_UNKNOWN)
    check("空输入安全降级",
          PL.match_format_stats(np.zeros(0, dtype=np.float32))["format"]
          == PL.MATCH_FORMAT_UNKNOWN)
    check("稳定 code 集合",
          {PL.MATCH_FORMAT_SINGLE, PL.MATCH_FORMAT_DOUBLES, PL.MATCH_FORMAT_UNKNOWN}
          == {"single", "doubles", "unknown"})


def test_join_abutting() -> None:
    print("\n消除首尾相接")
    ivs = [RA.RallyInterval(start=0, end=10), RA.RallyInterval(start=10, end=20),
           RA.RallyInterval(start=30, end=40)]
    out = P._join_abutting(ivs)
    check("相接的两段合成一段", len(out) == 2, str([(x.start, x.end) for x in out]))
    check("合并后区间正确", out[0].start == 0 and out[0].end == 20)


def test_activity_segmentation_has_gaps() -> None:
    """Activity-valley segmentation must not produce abutting intervals."""
    print("\n活跃度谷值切分")
    fps = 12.0
    dur = 200.0
    n = int(dur * fps)
    a = np.full(n, 0.2, dtype=np.float32)
    for s, e in ((20, 50), (70, 100), (120, 150)):
        a[int(s * fps):int(e * fps)] = 0.9
    a += np.random.RandomState(1).normal(0, 0.01, n).astype(np.float32)
    segs = RV.segment_activity(np.clip(a, 0, 1), fps, dur,
                               RV.SegmentOptions(min_rally=2.0, max_rally=120.0))
    check("切出 3 个回合", len(segs) == 3, str([(round(s.start, 1), round(s.end, 1)) for s in segs]))
    ss = sorted(segs, key=lambda x: x.start)
    touching = sum(1 for i in range(len(ss) - 1)
                   if ss[i + 1].start - ss[i].end < 0.3)
    check("没有首尾相接", touching == 0, f"{touching}")


def test_split_by_hit_gaps() -> None:
    """A large gap in the hit sequence must split the interval; a small gap must not.

    This corresponds to a bug found in testing: a 30-minute clip produced a 69.35-second,
    72-shot "rally" (0.0~69.35s) that actually contained two or three rallies. The hit
    intervals had a 6.93-second gap, yet segmentation made no use of that information at all.
    """
    print("\n击球空档切分")
    # 0~29s dense hits (one shot every 0.6s), 29~35s gap, 35~50s dense again
    times = list(np.arange(0.5, 29.0, 0.6)) + list(np.arange(35.0, 50.0, 0.6))
    times = np.asarray(times, dtype=np.float64)
    hits = RA.HitDetection(
        times=times,
        strength=np.full(times.size, 0.6, dtype=np.float32),
        confidence=np.full(times.size, 0.8, dtype=np.float32),
        envelope=np.zeros(10, dtype=np.float32), env_fps=250.0,
    )
    ivs = [RA.RallyInterval(start=0.0, end=50.0)]
    out = RA.split_by_hit_gaps(ivs, hits)
    check("大空档处被切开", len(out) == 2, f"{[(round(x.start, 1), round(x.end, 1)) for x in out]}")
    if len(out) == 2:
        check("切点落在空档内", 29.0 <= out[0].end <= 35.0, f"{out[0].end:.1f}")

    # Splitting is not allowed when there are only small gaps (all within MAX_INTRA_HIT_GAP)
    small = np.asarray([0.5, 1.2, 1.9, 4.5, 5.2, 5.9, 6.6], dtype=np.float64)
    h2 = RA.HitDetection(
        times=small,
        strength=np.full(small.size, 0.6, dtype=np.float32),
        confidence=np.full(small.size, 0.8, dtype=np.float32),
        envelope=np.zeros(10, dtype=np.float32), env_fps=250.0,
    )
    out2 = RA.split_by_hit_gaps([RA.RallyInterval(start=0.0, end=7.0)], h2)
    check("小空档不切", len(out2) == 1, f"{len(out2)}")

    # Large gap but only 1 shot on each side: a missed detection, not two rallies
    thin = np.asarray([0.5, 20.0], dtype=np.float64)
    h3 = RA.HitDetection(
        times=thin,
        strength=np.full(thin.size, 0.6, dtype=np.float32),
        confidence=np.full(thin.size, 0.8, dtype=np.float32),
        envelope=np.zeros(10, dtype=np.float32), env_fps=250.0,
    )
    out3 = RA.split_by_hit_gaps([RA.RallyInterval(start=0.0, end=25.0)], h3)
    check("两侧拍数不足时不切（防漏检误切）", len(out3) == 1, f"{len(out3)}")


def test_refine_trims_tail() -> None:
    """The end must be tighten-able to "last hit + tail", not only pushed later.

    The old implementation wrote ``iv.end = max(iv.end, last + tail)``, so no matter how long
    the interval from segmentation was, it could never be shortened -- this is the direct cause
    of "a long stretch is left after the shuttle lands".
    """
    print("\n终点锚定到最后一拍")
    times = np.asarray([10.0, 10.8, 11.6, 12.4, 13.2], dtype=np.float64)
    hits = RA.HitDetection(
        times=times,
        strength=np.full(times.size, 0.7, dtype=np.float32),
        confidence=np.full(times.size, 0.9, dtype=np.float32),
        envelope=np.zeros(10, dtype=np.float32), env_fps=250.0,
    )
    # The end from segmentation is far after the last hit (13.2 -> 40.0); it must be tightened to 13.2 + 0.9
    ivs = [RA.RallyInterval(start=9.0, end=40.0)]
    out = RA.refine_with_hits(ivs, hits, pre_roll=1.2, post_roll=0.5, tail_seconds=0.9)
    check("终点被收紧到最后一拍 + 尾巴",
          abs(out[0].end - (13.2 + 0.9)) < 0.01, f"{out[0].end:.2f}")
    check("起点对齐到第一次击球前 pre_roll",
          abs(out[0].start - (10.0 - 1.2)) < 0.01, f"{out[0].start:.2f}")

    # Conversely: there are hits right after the last hit (< gap_limit), meaning the shuttle is still in flight, so do not tighten
    times2 = np.asarray([10.0, 10.8, 11.6, 16.5, 17.3], dtype=np.float64)
    hits2 = RA.HitDetection(
        times=times2,
        strength=np.full(times2.size, 0.7, dtype=np.float32),
        confidence=np.full(times2.size, 0.9, dtype=np.float32),
        envelope=np.zeros(10, dtype=np.float32), env_fps=250.0,
    )
    ivs2 = [RA.RallyInterval(start=9.0, end=25.0)]
    out2 = RA.refine_with_hits(ivs2, hits2, pre_roll=1.2, post_roll=0.5,
                               tail_seconds=0.9, split=False)
    check("后面还有击球时不收紧", out2[0].end > 17.3, f"{out2[0].end:.2f}")

    # One hit can belong to only one rally: the two sides of a split point must not both count the same shot.
    # (The start is allowed to extend back by pre_roll to cover the serve preparation, so the **boundaries**
    #  of the two segments naturally overlap; the real constraint is "no duplicate shots", and dedupe_overlaps trims the overlap.)
    ivs3 = [RA.RallyInterval(start=9.0, end=13.0), RA.RallyInterval(start=13.0, end=20.0)]
    out3 = RA.refine_with_hits(ivs3, hits, pre_roll=1.2, post_roll=0.5,
                               tail_seconds=0.9, split=False)
    s0, s1 = set(out3[0].hit_indices), set(out3[1].hit_indices)
    check("两段不共用同一拍", not (s0 & s1), f"{sorted(s0)} / {sorted(s1)}")
    fixed = RA.dedupe_overlaps(out3)
    check("dedupe 之后不再重叠",
          all(fixed[i].end <= fixed[i + 1].start + 1e-6 for i in range(len(fixed) - 1)),
          f"{[(round(x.start, 2), round(x.end, 2)) for x in fixed]}")


def test_join_abutting_keeps_shots() -> None:
    """Merging adjacent intervals must preserve hit indices (the old code cleared them, making merged rallies show "0 shots")."""
    print("\n合并相邻区间保留拍数")
    a = RA.RallyInterval(start=0, end=10)
    a.hit_indices = [1, 2, 3]
    b = RA.RallyInterval(start=10, end=20)
    b.hit_indices = [4, 5]
    out = P._join_abutting([a, b])
    check("合成为一段", len(out) == 1, f"{len(out)}")
    check("拍数被保留", len(out[0].hit_indices) == 5, f"{out[0].hit_indices}")


def test_finish_intervals_padding_once() -> None:
    """The canonical finishing pass must add rolls exactly once.

    Regression for the P0口径分叉: rally_vision segments already carry
    pre_roll/post_roll internally, but run_analysis/resegment added them a second
    time when no hit sequence existed, while the annotation optimizer added them
    zero times. Only the legacy hysteresis state machine emits tight intervals and
    still needs the post-hoc rolls.
    """
    print("\n收尾口径：padding 只生效一次")
    params = AnalysisParams(pre_roll=1.0, post_roll=0.6,
                            hit_tail_seconds=0.9, min_rally_seconds=2.0)
    act = np.full(int(120 * 12), 0.5, dtype=np.float32)
    fused = RA.FusedSignal(fps=12.0, duration=120.0, activity=act)

    # rally_vision path: the interval already includes the rolls; anchoring must not touch it,
    # and running the finishing pass again must be idempotent (no accumulated double padding).
    out = P._finish_intervals([RA.RallyInterval(start=9.0, end=35.6)], hits=None,
                              params=params, fused=fused, duration=120.0,
                              method="player_motion+activity")
    check("vision 路径不二次 padding",
          len(out) == 1 and abs(out[0].start - 9.0) < 1e-9 and abs(out[0].end - 35.6) < 1e-9,
          str([(x.start, x.end) for x in out]))
    out2 = P._finish_intervals([RA.RallyInterval(start=out[0].start, end=out[0].end)], hits=None,
                               params=params, fused=fused, duration=120.0,
                               method="player_motion+activity")
    check("vision 路径重复收尾结果不变",
          abs(out2[0].start - 9.0) < 1e-9 and abs(out2[0].end - 35.6) < 1e-9,
          str([(x.start, x.end) for x in out2]))

    # Empty HitDetection must be treated like "no hits" (the old run_analysis path called
    # refine_with_hits with an empty sequence and got its internal padding branch — another
    # source of double padding on the vision path).
    empty_hits = RA.HitDetection(
        times=np.zeros(0, dtype=np.float64),
        strength=np.zeros(0, dtype=np.float32),
        confidence=np.zeros(0, dtype=np.float32),
        envelope=np.zeros(0, dtype=np.float32), env_fps=250.0)
    out3 = P._finish_intervals([RA.RallyInterval(start=9.0, end=35.6)], hits=empty_hits,
                               params=params, fused=fused, duration=120.0,
                               method="activity_valleys")
    check("空击球序列等价于无击球（vision 不二次 pad）",
          abs(out3[0].start - 9.0) < 1e-9 and abs(out3[0].end - 35.6) < 1e-9,
          str([(x.start, x.end) for x in out3]))

    # Legacy hysteresis path: tight intervals get the rolls exactly once.
    legacy = P._finish_intervals([RA.RallyInterval(start=10.0, end=35.0)], hits=None,
                                 params=params, fused=fused, duration=120.0,
                                 method="activity_state_machine")
    check("legacy 路径补一次 pre/post roll",
          len(legacy) == 1 and abs(legacy[0].start - 9.0) < 1e-9
          and abs(legacy[0].end - 35.6) < 1e-9,
          str([(x.start, x.end) for x in legacy]))

    # With hit evidence the canonical helper anchors like refine_with_hits: the end must be
    # pullable back to the last shot + tail even on the vision path.
    times = np.asarray([10.0, 10.8, 11.6, 12.4, 13.2], dtype=np.float64)
    hits = RA.HitDetection(
        times=times,
        strength=np.full(times.size, 0.7, dtype=np.float32),
        confidence=np.full(times.size, 0.9, dtype=np.float32),
        envelope=np.zeros(10, dtype=np.float32), env_fps=250.0)
    anchored = P._finish_intervals([RA.RallyInterval(start=9.0, end=40.0)], hits=hits,
                                   params=params, fused=fused, duration=120.0,
                                   method="player_motion+activity")
    check("有击球时统一收尾做终点锚定",
          abs(anchored[0].end - (13.2 + 0.9)) < 0.01, f"{anchored[0].end:.2f}")


# ------------------------------------------------------------------ P3 boundary evidence


def _boundary_motion_fixture() -> tuple:
    """Square-wave player motion (3 rallies, 4 s gaps) at 12 fps, 60 s."""
    from bms.analysis import boundary as BD
    fps = 12.0
    n = int(60 * fps)
    motion = np.zeros(n, dtype=np.float32)
    swing = np.zeros(n, dtype=np.float32)
    rallies = [(10.0, 18.0), (22.0, 30.0), (34.0, 42.0)]
    for s, e in rallies:
        a, b = int(s * fps), int(e * fps)
        motion[a:b] = 1.0
        swing[a:b] = 1.2
    motion = np.convolve(motion, np.ones(5) / 5, mode="same").astype(np.float32)
    return BD, fps, n, motion, swing, rallies


def test_boundary_evidence_direction() -> None:
    """Feature channels must point "boundary-like" at onsets/decays, not inside pauses."""
    print("\n边界证据：onset/decay 方向正确")
    BD, fps, n, motion, swing, rallies = _boundary_motion_fixture()
    ev = BD.build_evidence(
        fps=fps, duration=60.0, motion=motion,
        coverage=np.ones(n, dtype=np.float32),
        swing=swing, swing_fps=fps, overhead=np.zeros(n, dtype=np.float32),
        pose_coverage=0.9, swing_quiet=0.3, hit_times=[])
    check("证据可用（pose coverage 充足）", ev.available_for_refine())
    # onset slope: max near the labeled start, ~0 mid-pause
    for s, e in rallies:
        around = ev.channels["start"]["motion_slope"][
            ev.frame(s - 0.3):ev.frame(s + 0.3)].max()
        midgap = ev.channels["start"]["motion_slope"][
            ev.frame(s - 2.0):ev.frame(s - 1.2)].max()
        check(f"onset slope 在 {s}s 起跳处高于前段", around > midgap + 0.2,
              f"{around:.3f} vs {midgap:.3f}")
        dec = ev.channels["end"]["motion_slope"][
            ev.frame(e - 0.3):ev.frame(e + 0.3)].max()
        after = ev.channels["end"]["motion_slope"][
            ev.frame(e + 1.2):ev.frame(e + 2.0)].max()
        check(f"decay slope 在 {e}s 收尾处高于后段", dec > after + 0.2,
              f"{dec:.3f} vs {after:.3f}")
    # hit proximity is 1 right at a hit and 0 far away
    ev2 = BD.build_evidence(
        fps=fps, duration=60.0, motion=motion,
        swing=swing, swing_fps=fps, pose_coverage=0.9,
        hit_times=[11.0, 12.0])
    check("hit 特征：起跳点紧邻首击", abs(ev2.hit_feature("start", ev2.frame(11.0)) - 1.0) < 1e-6)
    check("hit 特征：4s 外归零", ev2.hit_feature("start", ev2.frame(7.0)) == 0.0)


def test_boundary_fit_weights() -> None:
    """fit_template keeps only directionally separating channels; degenerate classes refuse."""
    print("\n边界模板：权重方向与退化保护")
    BD, *_ = _boundary_motion_fixture()
    rows_pos = [{"motion_contrast": 0.8, "motion_slope": 0.7, "swing": 0.8,
                 "overhead": 0.2, "hit": 0.7}] * 8
    rows_neg = [{"motion_contrast": 0.1, "motion_slope": 0.9, "swing": 0.1,
                 "overhead": 0.2, "hit": 0.1}] * 8
    fit = BD.fit_template(rows_pos, rows_neg, "start", name="unit")
    tpl = fit.template
    check("正分离通道有权重", tpl.weights["motion_contrast"] > 0
          and tpl.weights["hit"] > 0)
    check("反方向通道权重为 0", tpl.weights["motion_slope"] == 0.0
          and tpl.weights["overhead"] == 0.0)
    check("阈值落在 0..1", 0.0 <= tpl.thr_start <= 1.0)
    acc_pos = BD.acceptance(tpl, "start", rows_pos)["accepted"]
    acc_neg = BD.acceptance(tpl, "start", rows_neg)["accepted"]
    check("正样本高接受、负样本低接受", acc_pos >= 0.8 and acc_neg <= 0.2,
          f"pos={acc_pos} neg={acc_neg}")
    bad = BD.fit_template(rows_pos, [], "start", name="unit")
    check("缺负样本时拒绝拟合（返回中性占位）",
          bad.template.name.endswith("insufficient")
          and bad.template.weights["motion_contrast"] == 1.0)
    merged = BD.merge_templates(tpl, tpl, name="m")
    check("merge 保留双侧阈值", merged.thr_start == tpl.thr_start
          and merged.thr_end == tpl.thr_end)


def _snap_evidence(bottom_frame: int, win: int = 5):
    """Evidence whose end-side channels peak at one quiet-span bottom frame."""
    BD, fps, n, motion, swing, _rallies = _boundary_motion_fixture()
    tpl = BD.Template(
        ranges={"motion_contrast": (0.0, 1.0), "motion_slope": (0.0, 1.0),
                "swing": (0.0, 1.0), "overhead": (0.0, 1.0), "hit": (0.0, 1.0)},
        weights={"motion_contrast": 1.0, "motion_slope": 1.0, "swing": 0.0,
                 "overhead": 0.0, "hit": 0.0},
        thr_start=0.55, thr_end=0.5, name="snap-fixture")
    ev = BD.build_evidence(
        fps=fps, duration=60.0, motion=motion,
        coverage=np.ones(n, dtype=np.float32),
        swing=swing, swing_fps=fps, overhead=np.zeros(n, dtype=np.float32),
        pose_coverage=0.9, swing_quiet=0.3, hit_times=[], template=tpl)
    for ch in ("motion_contrast", "motion_slope"):
        band = ev.channels["end"][ch]
        band[:] = 0.0
        a, b = max(0, bottom_frame - win), min(n, bottom_frame + win + 1)
        band[a:b] = 1.0
    ev.curves["end"] = BD._pack_curve(ev.channels["end"], tpl)
    return BD, ev, tpl


def test_boundary_refine_guardrails() -> None:
    """refine_boundaries snaps only through threshold/margin/channel/neighbor guardrails."""
    print("\n边界 refine：guardrail 全套")
    BD, fps, n, motion, swing, rallies = _boundary_motion_fixture()
    spans = RV.find_quiet_spans(motion, fps, 0.6, prominence_ratio=0.10)
    # quiet span between rally 1 and 2
    bottom_t = spans[0].bottom / fps
    BD, ev, tpl = _snap_evidence(spans[0].bottom)

    def run(ivs, params, duration=60.0):
        return BD.refine_boundaries(
            [RA.RallyInterval(start=a, end=b) for a, b in ivs],
            ev, params, duration, method="player_motion")

    # current end placed 1 s past the quiet bottom (still inside the gap); candidate wins
    p = AnalysisParams(use_boundary_refine=True, boundary_max_move=3.0,
                       boundary_score_margin=0.1)
    ivs = [(10.0, bottom_t + 1.0), (22.0, 33.0), (34.0, 45.0)]
    _out, tr = run(ivs, p)
    check("高分谷底触发 end snap", tr["moved"] == 1 and tr["status"] == "applied",
          str(tr))
    moved = tr["snaps"][0]
    check("snap 落在谷底且向内（提前终点）",
          abs(moved["to"] - bottom_t) < 0.12 and moved["to"] < moved["frm"],
          str(moved))

    # margin too high -> refuse (candidate gain is ~1.0 here)
    p_hi = AnalysisParams(use_boundary_refine=True, boundary_max_move=3.0,
                          boundary_score_margin=1.01)
    _, tr_hi = run(ivs, p_hi)
    check("score margin 不足不动", tr_hi["moved"] == 0)

    # max_move too small -> candidate out of legal window
    p_small = AnalysisParams(use_boundary_refine=True, boundary_max_move=0.5,
                             boundary_score_margin=0.1)
    _, tr_small = run(ivs, p_small)
    check("超出 max_move 不动", tr_small["moved"] == 0, str(tr_small))

    # neighbor guard: the next rally starts just before the candidate bottom, so snapping
    # would make interval 0 overlap interval 1 -> blocked.
    ivs_cross = [(10.0, bottom_t + 1.0), (bottom_t - 0.3, 33.0), (34.0, 45.0)]
    _, tr_cross = run(ivs_cross, p)
    check("不跨越相邻回合起点", all(s["index"] != 0 or s["side"] != "end"
                                    for s in tr_cross.get("snaps", [])),
          str(tr_cross.get("snaps")))

    # switch off -> same object, untouched; no pose -> status pose_coverage_low
    iv_objs = [RA.RallyInterval(start=a, end=b) for a, b in ivs]
    same, tr_off = BD.refine_boundaries(iv_objs, ev, AnalysisParams(), 60.0)
    check("开关关闭原样返回", tr_off["status"] == "off" and same is iv_objs)
    ev_nopose = BD.build_evidence(
        fps=fps, duration=60.0, motion=motion, coverage=np.ones(n, np.float32),
        swing=None, pose_coverage=0.0, hit_times=[])
    _, tr_nopose = BD.refine_boundaries(iv_objs, ev_nopose, p, 60.0)
    check("低 pose coverage 不动", tr_nopose["status"] == "pose_coverage_low")
    _, tr_empty = BD.refine_boundaries([], ev, p, 60.0)
    check("空区间安全", tr_empty["status"] == "empty")


def test_boundary_pack_roundtrip() -> None:
    """Packed curves survive the signals dict; missing keys rebuild; defaults stay off."""
    print("\n边界证据：pack 往返 / 缺键重建 / 默认关")
    BD, fps, n, motion, swing, _rallies = _boundary_motion_fixture()
    tpl = BD.NEUTRAL_TEMPLATE
    ev = BD.build_evidence(
        fps=fps, duration=60.0, motion=motion,
        coverage=np.ones(n, dtype=np.float32),
        swing=swing, swing_fps=fps, overhead=np.zeros(n, dtype=np.float32),
        pose_coverage=0.9, swing_quiet=0.3, hit_times=[])
    packed = BD.pack_signals(ev)
    check("pack 版本键", packed["boundary_version"] == [BD.BOUNDARY_EVIDENCE_VERSION])
    sig = {
        "activity_full": [round(float(v), 4) for v in motion],
        "fps": [fps], "duration": [60.0],
        "player_motion_full": [round(float(v), 4) for v in motion],
        "player_coverage_full": [1.0] * n, "player_fps": [fps],
        "pose_swing_full": [round(float(v), 4) for v in swing],
        "pose_overhead_full": [0.0] * n,
        "pose_fps": [fps], "pose_coverage": [0.9], "pose_quiet": [0.3],
    }
    sig.update(packed)
    ev2 = BD.evidence_from_signals(sig, None)
    check("packed 曲线逐帧复用",
          np.allclose(ev2.curves["start"], ev.curves["start"], atol=2e-4)
          and np.allclose(ev2.curves["end"], ev.curves["end"], atol=2e-4))
    raw = {k: v for k, v in sig.items()
           if not k.startswith("boundary_")}
    ev3 = BD.evidence_from_signals(raw, None)
    check("缺 boundary 键时从原始数组重建",
          ev3 is not None and ev3.n == n and ev3.available_for_refine())
    # activity-only fallback (old project without player curves)
    old = {"activity_full": sig["activity_full"], "fps": [fps], "duration": [60.0]}
    ev4 = BD.evidence_from_signals(old, None)
    check("无 player/pose 曲线也能降级建证据（但不可 refine）",
          ev4 is not None and not ev4.available_for_refine())
    check("AnalysisParams 默认关闭边界 refine",
          AnalysisParams().use_boundary_refine is False
          and AnalysisParams().boundary_max_move == BD.DEFAULT_MAX_MOVE
          and AnalysisParams().boundary_score_margin == BD.DEFAULT_SCORE_MARGIN)
    meta = BD.boundary_meta(ev)
    check("stats 元数据带 available/version",
          meta["available"] is True and meta["version"] == BD.BOUNDARY_EVIDENCE_VERSION)


def _fuse_fixture() -> RA.FusedSignal:
    """Synthetic fused signal with rally-like bursts on players + motion channels."""
    rng = np.random.default_rng(0)
    fps, dur = 12.0, 60.0
    n = int(dur * fps)
    base = np.zeros(n, dtype=np.float32)
    for s, e in ((10, 18), (22, 30), (34, 42)):
        base[int(s * fps):int(e * fps)] = 1.0
    motion_arr = np.clip(base + rng.normal(0, 0.1, n), 0, None).astype(np.float32)
    return RA.fuse(
        fps=fps, duration=dur,
        motion={"fps": fps, "court_motion": motion_arr},
        players={"fps": fps, "active_count": base.copy(),
                 "active_speed": motion_arr, "max_speed": motion_arr * 1.5})


def test_fuse_weight_defaults() -> None:
    """Params defaults stay the single source of truth shared with rally fuse constants."""
    print("\n融合权重：默认值与 rally 常量一致")
    p = AnalysisParams()
    base = p.fuse_weight_base()
    check("5 个组件键齐全",
          tuple(sorted(base)) == tuple(sorted(RA.FUSE_COMPONENT_KEYS)))
    check("默认权重 == DEFAULT_BASE_WEIGHTS",
          all(abs(base[k] - RA.DEFAULT_BASE_WEIGHTS[k]) < 1e-12
              for k in RA.FUSE_COMPONENT_KEYS))
    p2 = p.model_copy(update={"fuse_weight_players": 2.0})
    check("model_copy 覆盖生效且不污染默认实例",
          p2.fuse_weight_base()["players"] == 2.0 and p.fuse_weight_base()["players"] == 1.35)


def test_fuse_components_roundtrip() -> None:
    """Re-fusing fuse()'s own components must reproduce the activity exactly (default weights)."""
    print("\n融合分量：往返一致 / 覆盖改变方向 / 长度漂移守卫")
    f = _fuse_fixture()
    g = RA.fuse_from_components(
        f.components, fps=f.fps, duration=f.duration, audio_rel=f.audio_reliability)
    check("默认权重重融合逐帧一致",
          np.array_equal(f.activity, g.activity)
          and abs(f.threshold_hi - g.threshold_hi) < 1e-9
          and abs(f.threshold_lo - g.threshold_lo) < 1e-9)
    check("5 条分量齐全",
          set(f.components) == set(RA.FUSE_COMPONENT_KEYS))
    wb = dict(RA.DEFAULT_BASE_WEIGHTS)
    wb["players"] = 0.0
    h = RA.fuse_from_components(
        f.components, fps=f.fps, duration=f.duration,
        audio_rel=f.audio_reliability, weight_base=wb)
    check("权重覆盖确实改变活动度曲线",
          not np.array_equal(f.activity, h.activity))
    # Rounded (packed) components stay numerically close at default weights.
    rounded = {k: np.round(v, 4) for k, v in f.components.items()}
    g3 = RA.fuse_from_components(
        rounded, fps=f.fps, duration=f.duration, audio_rel=f.audio_reliability)
    check("4 位小数落盘分量重融合误差 < 5e-4",
          float(np.abs(f.activity - g3.activity).max()) < 5e-4)
    # Unknown weight keys are ignored; length drift is resampled.
    g4 = RA.fuse_from_components(
        {k: v[:-7] for k, v in f.components.items()},
        fps=f.fps, duration=f.duration, audio_rel=f.audio_reliability,
        weight_base={"players": 1.35, "bogus": 9.0})
    check("漂移长度自动 resample、未知权重键忽略",
          g4.activity.size == f.activity.size)


def test_shuttle_in_flight_build() -> None:
    """Validated-track coverage bridges association gaps and takes max confidence."""
    print("\n羽毛球在飞曲线：跨丢帧覆盖 / 置信度取大 / 空输入")
    from bms.analysis import shuttle as SH
    t1 = SH.ShuttleTrack(
        points=[SH.ShuttlePoint(f, f / 12.0, 0.1, 0.2, 0.9) for f in (10, 11, 13)],
        start=10 / 12.0, end=13 / 12.0, confidence=0.8)
    t2 = SH.ShuttleTrack(
        points=[SH.ShuttlePoint(12, 1.0, 0.1, 0.2, 0.9),
                SH.ShuttlePoint(99, 8.0, 0.1, 0.2, 0.9)],
        start=1.0, end=8.0, confidence=0.4)
    flight = SH.build_in_flight([t1, t2], 20)
    check("丢帧 12 被桥接覆盖", flight[12] == 0.8)
    check("整段跨度连续在飞", all(flight[10:14] == 0.8))
    check("重叠轨迹取最大置信度", flight[12] == 0.8)
    check("独立点轨迹正常覆盖、越界点忽略",
          flight[12] == 0.8 and flight.size == 20 and float(flight[:10].sum()) == 0.0)
    check("空轨迹 / 空长度安全",
          SH.build_in_flight([], 5).sum() == 0.0 and SH.build_in_flight([t1], 0).size == 0)


def test_fuse_shuttle_in_flight_blend() -> None:
    """Fuse shuttle comp: legacy presence/speed blend unchanged; in_flight only adds when nonzero."""
    print("\n融合 shuttle 分量：缺省零差异 / 在飞曲线生效")
    fps, dur, n = 12.0, 20.0, 240
    burst = np.zeros(n, dtype=np.float32)
    burst[60:120] = 1.0
    base_sh = {"fps": fps, "presence": np.zeros(n, np.float32),
               "max_candidate_speed": np.zeros(n, np.float32)}
    f0 = RA.fuse(fps=fps, duration=dur, shuttle=dict(base_sh))
    fz = RA.fuse(fps=fps, duration=dur,
                 shuttle={**base_sh, "in_flight": np.zeros(n, np.float32)})
    check("零在飞曲线与缺省逐帧一致",
          np.array_equal(f0.components["shuttle"], fz.components["shuttle"]))
    check("无在飞信息时 shuttle 分量恒零", float(np.abs(f0.components["shuttle"]).max()) == 0.0)
    ff = RA.fuse(fps=fps, duration=dur,
                 shuttle={**base_sh, "in_flight": burst})
    check("非零在飞曲线进入 shuttle 分量",
          ff.components["shuttle"][90] > 0.0 and ff.components["shuttle"][10] == 0.0)
    check("在飞曲线改变最终活动度", not np.array_equal(ff.activity, f0.activity))


def _write_synthetic_shuttle_video(path: Path, *, w: int = 640, h: int = 360,
                                   n: int = 45, fps: float = 30.0) -> Path:
    """Write a near-lossless AVI: flat green court, a static white court-line bar,
    and a 3x3 white dot moving diagonally across frames 5..40."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert vw.isOpened(), "cv2.VideoWriter MJPG unavailable"
    try:
        vw.set(cv2.VIDEOWRITER_PROP_QUALITY, 100.0)
    except Exception:
        pass
    bg = np.zeros((h, w, 3), np.uint8)
    bg[:] = (40, 130, 40)                       # BGR saturated green court
    bg[300:306, 80:560] = 235                   # constantly-white line: exercises static-white mask
    for f in range(n):
        fr = bg.copy()
        if 5 <= f <= 40:
            x = 60 + (f - 5) * 6
            y = 60 + (f - 5) * 3
            fr[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = 255
        vw.write(fr)
    vw.release()
    return path


def test_shuttle_backend_selection() -> None:
    """Backend resolver honors force/env and torch CUDA availability."""
    print("\nshuttle 后端选择：force/env 与 CUDA 可用性")
    from bms.analysis import shuttle as SH

    old = os.environ.pop("BMS_SHUTTLE_BACKEND", None)
    try:
        check("force=cpu 永远选 cpu", SH._resolve_backend("cpu") == "cpu")
        have_gpu = bool(SH._gpu_available())
        check("auto 依 CUDA 可用性选择", SH._resolve_backend("auto") == ("gpu" if have_gpu else "cpu"))
        os.environ["BMS_SHUTTLE_BACKEND"] = "cpu"
        check("env=cpu 覆盖 auto", SH._resolve_backend(None) == "cpu")
        os.environ["BMS_SHUTTLE_BACKEND"] = "garbage"
        check("env 非法值按 auto 处理", SH._resolve_backend(None) == ("gpu" if have_gpu else "cpu"))
        if not have_gpu:
            os.environ["BMS_SHUTTLE_BACKEND"] = "gpu"
            check("env=gpu 但 CUDA 不可用静默回退 cpu", SH._resolve_backend(None) == "cpu")
            raised = False
            try:
                SH._resolve_backend("gpu")
            except RuntimeError:
                raised = True
            check("force=gpu 但 CUDA 不可用硬失败", raised)
    finally:
        if old is None:
            os.environ.pop("BMS_SHUTTLE_BACKEND", None)
        else:
            os.environ["BMS_SHUTTLE_BACKEND"] = old


def test_shuttle_gpu_morph_equivalence() -> None:
    """GPU binary morphology kernels must match cv2's exact structuring elements."""
    from bms.analysis import shuttle as SH
    if not SH._gpu_available():
        print("\nshuttle GPU 形态学等价：skip（无可用 CUDA）")
        return
    print("\nshuttle GPU 形态学与 cv2 逐像素等价")
    import cv2

    rng = np.random.default_rng(7)
    m = (rng.random((60, 80)) > 0.85).astype(np.uint8)
    cross = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cpu_open = cv2.morphologyEx(m, cv2.MORPH_OPEN, cross)
    gpu_open = SH._gpu_binary_open(m, "cuda")
    check("cross 3x3 open 逐像素一致", np.array_equal(cpu_open, gpu_open))
    cpu_dil = cv2.dilate(m, np.ones((3, 3), np.uint8))
    gpu_dil = SH._gpu_binary_dilate(m, "cuda")
    check("square 3x3 dilate 逐像素一致", np.array_equal(cpu_dil, gpu_dil))


def test_shuttle_gpu_cpu_parity() -> None:
    """Same synthetic video through both backends: same candidate frames and sub-2px centers."""
    from bms.analysis import shuttle as SH
    if not SH._gpu_available():
        print("\nshuttle GPU/CPU 合成片一致性：skip（无可用 CUDA）")
        return
    print("\nshuttle GPU/CPU 合成片候选一致性")
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        vp = _write_synthetic_shuttle_video(Path(td) / "syn.avi")
        kw = dict(sample_fps=30.0, window=2, work_width=640, sensitivity=0.5)
        sig_c, dbg_c = SH.analyze_shuttle_debug(str(vp), backend="cpu", **kw)
        sig_g, dbg_g = SH.analyze_shuttle_debug(str(vp), backend="gpu", **kw)

    kc = {f for f, v in dbg_c["candidates"].items() if v}
    kg = {f for f, v in dbg_g["candidates"].items() if v}
    check("两后端都检出移动白点（>=12 帧）", len(kc) >= 12 and len(kg) >= 12)
    union = kc | kg
    jacc = len(kc & kg) / max(1, len(union))
    check(f"候选帧集合 Jaccard>=0.8（cpu={len(kc)} gpu={len(kg)} J={jacc:.2f}）", jacc >= 0.8)
    dists: list[float] = []
    for f in kc & kg:
        pc = max(dbg_c["candidates"][f], key=lambda p: p[2])
        pg = max(dbg_g["candidates"][f], key=lambda p: p[2])
        dists.append(float(np.hypot((pc[0] - pg[0]) * 640.0, (pc[1] - pg[1]) * 360.0)))
    check("公共帧最强候选位置中位差<=1px 最大<=2px",
          bool(dists) and float(np.median(dists)) <= 1.0 and max(dists) <= 2.0)
    check("信号记录实际后端", sig_c.backend == "cpu" and sig_g.backend == "gpu")


def test_shuttle_gpu_runtime_error_falls_back() -> None:
    """GPU scan exception degrades silently to the full CPU scan with the reason recorded."""
    from bms.analysis import shuttle as SH
    if not SH._gpu_available():
        print("\nshuttle GPU 异常回退 CPU：skip（无可用 CUDA）")
        return
    print("\nshuttle GPU 异常时整段静默回退 CPU")
    import tempfile

    orig = SH._scan_candidates_gpu

    def _boom(*_a, **_k):
        raise RuntimeError("synthetic gpu failure")

    SH._scan_candidates_gpu = _boom
    try:
        with tempfile.TemporaryDirectory() as td:
            vp = _write_synthetic_shuttle_video(Path(td) / "syn.avi")
            sig, _dbg = SH.analyze_shuttle_debug(
                str(vp), backend="gpu", sample_fps=30.0, window=2, work_width=640)
    finally:
        SH._scan_candidates_gpu = orig
    check("回退后结果可用且后端标记为 cpu", sig.backend == "cpu")
    check("fallback 原因写入 backend_fallback", "synthetic gpu failure" in (sig.backend_fallback or ""))


def test_shuttle_budget_seconds_frame_unit() -> None:
    """max_seconds must cap reads in SAMPLED frames, not source frames.

    The scan limit used to be ``int(max_seconds * src_fps)`` while the loop counter
    only increments on sampled frames (one per ``step`` grabs), so the analyzed span
    was stretched ``step``-fold (0.5 s budget -> 1.5 s analyzed at step=3). pipeline.py
    reports ``coverage = min(duration, budget) / duration`` from this same budget, so
    an over-scanned signal makes that reported coverage wrong as well; the pipeline
    side itself needs no change.
    """
    from bms.analysis import shuttle as SH
    import tempfile

    # Synthetic clip: 30 fps / 45 frames = 1.5 s. sample_fps=10 -> step=3, eff_fps=10,
    # so max_seconds=0.5 must read round(0.5*10)=5 sampled frames (~0.5 s), instead of
    # all 15 sampled frames (1.5 s) allowed by the old source-frame limit.
    print("\nshuttle max_seconds budget honored in sampled frames (cpu/gpu)")
    with tempfile.TemporaryDirectory() as td:
        vp = _write_synthetic_shuttle_video(Path(td) / "syn.avi")
        kw = dict(sample_fps=10.0, window=2, work_width=640,
                  sensitivity=0.5, max_seconds=0.5)
        sig_c, _dbg_c = SH.analyze_shuttle_debug(str(vp), backend="cpu", **kw)
        check(f"cpu duration within one sampled interval of 0.5s (got {sig_c.duration:.3f}s)",
              0.25 < sig_c.duration < 0.75)
        if SH._gpu_available():
            sig_g, _dbg_g = SH.analyze_shuttle_debug(str(vp), backend="gpu", **kw)
            check(f"gpu duration within one sampled interval of 0.5s (got {sig_g.duration:.3f}s)",
                  0.25 < sig_g.duration < 0.75)
        else:
            print("  (gpu assertion skipped: no CUDA)")


def test_rebuild_fused_weight_overrides() -> None:
    """Offline rebuild: default weights reuse stored activity verbatim; override re-fuses."""
    print("\n离线重建：缺省逐帧复用 / 覆盖重融合 / 旧工程静默降级")
    f = _fuse_fixture()
    sig = {
        "fps": [f.fps], "duration": [f.duration],
        "activity_full": [round(float(x), 4) for x in f.activity],
    }
    sig.update({f"component_{k}_full": [round(float(x), 4) for x in v]
                for k, v in f.components.items()})
    p_def = AnalysisParams()
    rb = P._rebuild_fused(sig, p_def, act=f.activity, fps=f.fps,
                          duration=f.duration, audio_rel=f.audio_reliability)
    check("默认权重：存储 activity 原样返回（逐帧相等）",
          np.array_equal(rb.activity, f.activity))
    p_ov = p_def.model_copy(update={"fuse_weight_players": 0.0})
    ro = P._rebuild_fused(sig, p_ov, act=f.activity, fps=f.fps,
                          duration=f.duration, audio_rel=f.audio_reliability)
    check("覆盖权重：走重融合路径",
          not np.array_equal(ro.activity, f.activity))
    old = {"fps": [f.fps], "duration": [f.duration],
           "activity_full": sig["activity_full"]}
    ro2 = P._rebuild_fused(old, p_ov, act=f.activity, fps=f.fps,
                           duration=f.duration, audio_rel=f.audio_reliability)
    check("旧工程无 component_*_full：覆盖静默降级到存储曲线",
          np.array_equal(ro2.activity, f.activity))


def _optimizer_consistency_fixture(with_hits: bool) -> AnalysisResult:
    fps = 12.0
    dur = 120.0
    n = int(dur * fps)
    act = np.full(n, 0.4, dtype=np.float32)
    pm = np.zeros(n, dtype=np.float32)
    for a, b in ((10, 35), (60, 90)):
        pm[int(a * fps):int(b * fps)] = 1.0
        act[int(a * fps):int(b * fps)] = 0.9
    signals = {
        "activity_full": act.tolist(),
        "player_motion_full": pm.tolist(),
        "player_coverage_full": np.ones(n, dtype=np.float32).tolist(),
        "fps": [fps], "duration": [dur], "player_fps": [fps],
    }
    if with_hits:
        hit_times = [t for a, b in ((10, 35), (60, 90)) for t in np.arange(a, b, 1.2)]
        signals.update({
            "hit_times": [round(float(t), 3) for t in hit_times],
            "hit_strength": [0.8] * len(hit_times),
            "hit_confidence": [0.9] * len(hit_times),
        })
    return AnalysisResult(
        media_id="m_consistency", status="done",
        params=AnalysisParams(min_rally_seconds=2.0, pre_roll=0.5, post_roll=0.5),
        signals=signals, stats={"audio_reliability": 0.8})


def test_resegment_matches_optimizer_finishing() -> None:
    """resegment and the annotation optimizer must emit the same interval set for the same archive.

    The P0 fix routes both through ``_finish_intervals``; historically the optimizer skipped the
    no-hit padding while production double-padded vision intervals, so the F1 seen while tuning
    did not correspond to the F1 production shipped.
    """
    print("\n收尾口径：resegment 与优化器一致")
    for with_hits in (False, True):
        res = _optimizer_consistency_fixture(with_hits)
        prod = P.resegment(copy.deepcopy(res), res.params, "balanced")
        prod_ivs = [(round(r.start, 4), round(r.end, 4)) for r in prod.rallies]
        ctx = AN._build_context(copy.deepcopy(res), 0.0, 1e9)
        pred_ivs = sorted((round(a, 4), round(b, 4)) for a, b in AN._predict(ctx, res.params))
        tag = "有击球" if with_hits else "无击球"
        check(f"{tag}：两路径区间集合一致（{len(prod_ivs)} 段）",
              prod_ivs == pred_ivs,
              f"prod={prod_ivs[:4]} pred={pred_ivs[:4]}")


def _synthetic_pose(fps: float = 12.0, dur: float = 30.0, peaks=(3.0, 8.0, 15.0),
                    coverage: float = 1.0):
    """Build a synthetic pose signal: place one swing peak at each given time."""
    from bms.analysis import pose as POSE

    n = int(dur * fps)
    sw = np.full(n, 0.4, dtype=np.float32)
    over = np.zeros(n, dtype=np.float32)
    ok = np.ones(n, dtype=np.float32)
    for t in peaks:
        i = int(t * fps)
        # One swing is about 0.25 seconds: spike up and back within three frames
        for k, v in ((-2, 0.9), (-1, 3.4), (0, 5.6), (1, 2.1), (2, 0.6)):
            if 0 <= i + k < n:
                sw[i + k] = v
    if coverage < 1.0:
        cut = int(n * (1.0 - coverage))
        ok[cut:] = 0.0
        sw[cut:] = 0.0
    return POSE.PoseSignal(fps=fps, duration=dur, swing=sw, ok=ok, overhead=over,
                           coverage=coverage, quiet=0.4)


def _hits(times) -> "RA.HitDetection":
    t = np.asarray(times, dtype=np.float64)
    return RA.HitDetection(
        times=t, strength=np.full(t.size, 0.6, dtype=np.float32),
        confidence=np.full(t.size, 0.8, dtype=np.float32),
        envelope=np.zeros(10, dtype=np.float32), env_fps=250.0,
    )


def test_pose_swing_peaks() -> None:
    """Swing-peak detection must land on the positions of the synthetic peaks."""
    print("\n姿态：挥拍峰检测")
    from bms.analysis import pose as POSE

    pose = _synthetic_pose(peaks=(3.0, 8.0, 15.0))
    idx, prom = POSE.swing_peaks(pose)
    got = sorted(round(float(i) / pose.fps, 2) for i in idx)
    check("检出 3 个挥拍峰", idx.size == 3, f"{got}")
    check("峰位置正确", all(any(abs(g - t) < 0.2 for g in got) for t in (3.0, 8.0, 15.0)),
          f"{got}")
    check("显著度为正", bool(np.all(prom > 0)), f"{prom}")


def test_pose_hit_gating_one_to_one() -> None:
    """One swing peak can explain only one hit; a hit with no swing must be judged as "from another court"."""
    print("\n姿态：击球归属门控")
    from bms.analysis import pose as POSE

    pose = _synthetic_pose(peaks=(3.0, 8.0, 15.0))
    # Both the 3.0 and 3.08 shots are near one peak -> keep only the closer one; there is no swing near 5.0
    hits = _hits([3.0, 3.08, 5.0, 8.02, 15.0])
    ev = POSE.hit_swing_evidence(pose, hits.times, strengths=hits.strength)
    check("贴着峰的击球证据高", ev[0] > 0.5 and ev[3] > 0.5 and ev[4] > 0.5, f"{np.round(ev, 3)}")
    check("没有挥拍的击球证据为 0", ev[2] == 0.0, f"{ev[2]:.3f}")
    check("同一峰上只保留一个", int(np.sum(ev[:2] > 0.5)) == 1, f"{np.round(ev[:2], 3)}")

    mask, trace = POSE.gate_hits(hits, pose, threshold=0.3)
    check("门控生效", mask is not None, str(trace))
    if mask is not None:
        check("5.0 那一拍被剔除", not bool(mask[2]), f"{mask.tolist()}")
        check("91% 的保留比例在合理区间内", 0.4 < trace["keep_ratio"] < 0.8,
              str(trace))
        kept = POSE.filter_hits(hits, mask)
        check("filter_hits 长度正确", kept.times.size == int(mask.sum()),
              f"{kept.times.size} vs {int(mask.sum())}")


def test_pose_gate_degrades() -> None:
    """When pose is unreliable it must pass through unchanged, not cut the hits in half."""
    print("\n姿态：不可用时降级")
    from bms.analysis import pose as POSE

    hits = _hits([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # Coverage too low
    low = _synthetic_pose(coverage=0.2)
    mask, trace = POSE.gate_hits(hits, low)
    check("覆盖率低时门控不生效", mask is None, str(trace))
    # No peaks at all (signal completely flat)
    flat = _synthetic_pose(peaks=())
    mask2, trace2 = POSE.gate_hits(hits, flat)
    check("没有挥拍峰时候降级（保留比例过低）", mask2 is None, str(trace2))
    # pose is None
    mask3, _ = POSE.gate_hits(hits, None)
    check("pose 为 None 时返回 None", mask3 is None)
    check("filter_hits 收到 None 掩码时原样返回", POSE.filter_hits(hits, None) is hits)


def test_pose_keypoint_crop_inverse_map() -> None:
    """Patch-space keypoints must map back to frame coords by MULTIPLYING with the crop scale.

    The 192px pose patch is ``cv2.resize`` of the crop window: patch pixel p corresponds to frame
    ``x0 + p * crop_w / 192``. Dividing instead amplifies every offset and projects skeletons
    onto empty floor away from the player (the visual-cache bug seen on clip1).
    """
    print("\n姿态：关键点裁剪逆映射正确")
    from bms.analysis import pose as POSE

    x0, y0, x1, y1 = 607, 225, 742, 360
    kp_raw = np.asarray([[0.0, 0.0], [192.0, 192.0], [96.0, 96.0]], dtype=np.float32)
    mapped = POSE._map_keypoints(kp_raw, x0, y0, (x1 - x0) / 192.0, (y1 - y0) / 192.0)
    # Patch corners land exactly on the crop-window corners; center on the center.
    check("左上角映射到裁剪窗左上角",
          abs(mapped[0][0] - x0) < 1e-3 and abs(mapped[0][1] - y0) < 1e-3)
    check("右下角映射到裁剪窗右下角",
          abs(mapped[1][0] - x1) < 1e-3 and abs(mapped[1][1] - y1) < 1e-3)
    check("中心点映射到裁剪窗中心",
          abs(mapped[2][0] - (x0 + x1) / 2) < 1e-3
          and abs(mapped[2][1] - (y0 + y1) / 2) < 1e-3)


def test_pose_boxes_signature_quantization_stable() -> None:
    """Float boxes and their uint16 npz roundtrip must hash to the same pose cache signature.

    rebuild_visual_cache resolves the pose path from boxes loaded back from the compact npz;
    without quantization-stable hashing the tiny dequant error flips the %.4f rounding and the
    v2 pose cache misses, rerunning GPU pose on every rebuild.
    """
    print("\n姿态：框缓存量化前后签名一致")
    from bms.analysis import pose as POSE

    rng = np.random.RandomState(123)
    frames = [
        [(1, float(rng.rand()), float(rng.rand()), float(rng.rand()), float(rng.rand())),
         (2, float(rng.rand()), float(rng.rand()), float(rng.rand()), float(rng.rand()))]
        for _ in range(120)
    ]
    # Mirror players.save_boxes_cache: round(clip(coord, 0, 1) * 65535) on the uint16 grid
    def _deq(x: float) -> float:
        return round(min(1.0, max(0.0, x)) * 65535.0) / 65535.0

    quant = [
        [(int(track), _deq(x1), _deq(y1), _deq(x2), _deq(y2))
         for (track, x1, y1, x2, y2) in fr]
        for fr in frames
    ]
    check("量化前后 boxes 签名一致",
          POSE._boxes_signature(frames) == POSE._boxes_signature(quant))


def test_boxes_cache_v2_roundtrip_and_v1_compat() -> None:
    """v2 npz roundtrip (per-row conf + raw detections) and v1 back-compat loading.

    The v2 layout adds ``conf`` (uint8 ×255) aligned with the tracked-box rows and the
    ``raw_frame`` / ``raw_boxes`` / ``raw_conf`` detection layer. v1 files (written before the
    conf/raw columns existed) must keep loading with ``frame_confs=None, raw_dets=None`` —
    existing analyses reference them and the overlay must degrade, not break.
    """
    print("\n球员：框缓存 v2 读写与 v1 兼容")
    import tempfile

    rng = np.random.RandomState(7)
    n_frames = 40
    frame_boxes = []
    frame_confs = []
    for i in range(n_frames):
        if i % 3 == 2:
            frame_boxes.append([])
            frame_confs.append([])
            continue
        row = []
        conf_row = []
        for tid in (11, 12):
            x1, y1 = rng.rand() * 0.5, rng.rand() * 0.5
            row.append((tid, x1, y1, x1 + 0.1, y1 + 0.3))
            conf_row.append(round(float(rng.uniform(0.1, 0.9)), 3))
        frame_boxes.append(row)
        frame_confs.append(conf_row)
    raw_dets = [
        [((0.2, 0.3, 0.4, 0.8), 0.12), ((0.6, 0.3, 0.7, 0.8), 0.55)]
        if i % 2 == 0 else []
        for i in range(n_frames)
    ]

    with tempfile.TemporaryDirectory() as td:
        ok = PL.save_boxes_cache("unittest_v2", frame_boxes, 12.0, n_frames / 12.0,
                                 (11, 12), cache_dir=td,
                                 frame_confs=frame_confs, raw_dets=raw_dets)
        check("v2 保存成功", ok)
        doc = PL.load_boxes_cache("unittest_v2", cache_dir=td)
        check("v2 加载成功", doc is not None)
        if doc is None:
            return
        check("frame_boxes 行对齐", len(doc["frame_boxes"]) == n_frames
              and all(len(a) == len(b) for a, b in zip(doc["frame_boxes"], frame_boxes)))
        check("frame_boxes track/坐标一致",
              all(tuple(r)[:1] == tuple(o)[:1] and all(abs(a - b) < 1e-4 for a, b in zip(r[1:], o[1:]))
                  for got, want in zip(doc["frame_boxes"], frame_boxes)
                  for r, o in zip(got, want)))
        got_confs = doc["frame_confs"]
        check("v2 带 conf 列",
              got_confs is not None
              and all(abs(a - b) <= 1 / 255 + 1e-6
                      for got, want in zip(got_confs, frame_confs)
                      for a, b in zip(got, want)))
        got_raw = doc["raw_dets"]
        check("v2 带原始检测层",
              got_raw is not None and len(got_raw) == n_frames
              and len(got_raw[0]) == 2
              and abs(got_raw[0][0][4] - 0.12) <= 1 / 255 + 1e-6
              and all(abs(a - b) < 1e-4
                      for a, b in zip(got_raw[0][0][:4], (0.2, 0.3, 0.4, 0.8))))

        # ---- v1 compatibility: a file without the conf / raw arrays must still load ----
        v1_path = Path(td) / "boxes_unittest_v1.npz"
        rows = [(i, tid, *b[1:]) for i, fr in enumerate(frame_boxes) for tid, *b in [(r[0], r) for r in fr]]
        np.savez_compressed(
            v1_path,
            version=np.int64(1),
            fps=np.float64(12.0), duration=np.float64(n_frames / 12.0), n=np.int64(n_frames),
            active_ids=np.asarray([11, 12], dtype=np.int32),
            frame=np.asarray([r[0] for r in rows], dtype=np.int32),
            track=np.asarray([r[1] for r in rows], dtype=np.int32),
            boxes=np.round(np.clip(np.asarray([r[2:] for r in rows], dtype=np.float64), 0, 1)
                           * 65535.0).astype(np.uint16),
        )
        doc1 = PL.load_boxes_cache("unittest_v1", cache_dir=td)
        check("v1 文件可加载", doc1 is not None)
        if doc1 is not None:
            check("v1 无 conf（前端不过滤）", doc1["frame_confs"] is None)
            check("v1 无原始检测层", doc1["raw_dets"] is None)
            check("v1 frame_boxes 数量一致", len(doc1["frame_boxes"]) == n_frames)

        # ---- unknown future version must be a miss, not a crash ----
        v3_path = Path(td) / "boxes_unittest_v3.npz"
        np.savez_compressed(v3_path, version=np.int64(99), n=np.int64(0))
        check("未知版本视为未命中", PL.load_boxes_cache("unittest_v3", cache_dir=td) is None)

    # ---- cache tag must embed the detection floor so v1/v2 key spaces split ----
    tag_a = PL.boxes_cache_tag("some/video.mp4", sample_fps=12.0)
    tag_b = PL.boxes_cache_tag("some/video.mp4", sample_fps=13.0)
    check("不同采样率产生不同 tag", tag_a != tag_b and len(tag_a) == 16)
    orig_low = PL.LOW_CONF
    try:
        PL.LOW_CONF = 0.15
        tag_old = PL.boxes_cache_tag("some/video.mp4", sample_fps=12.0)
    finally:
        PL.LOW_CONF = orig_low
    check("tag 含检测下限分量（v1/v2 键空间分流）", tag_a != tag_old)
    check("检测下限为 0.10（v2 行为）", abs(PL.LOW_CONF - 0.10) < 1e-9)


def test_pose_gate_force() -> None:
    """The forced switch must override the retention-ratio guard, and mark the trace as forced."""
    print("\n姿态：强制应用")
    from bms.analysis import pose as POSE

    hits = _hits([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    flat = _synthetic_pose(peaks=())          # coverage ok, but no swings at all -> ratio 0
    mask, trace = POSE.gate_hits(hits, flat)
    check("默认保留率越界时放行", mask is None, str(trace))
    mask2, trace2 = POSE.gate_hits(hits, flat, force=True)
    check("force 强制应用（返回掩码）", mask2 is not None, str(trace2))
    check("trace 标注被强制", bool(trace2.get("forced")), str(trace2))


def _hit_gate_result(threshold: float = 0.22):
    """Synthetic analysis result with pre-gate candidates and a full-rate pose signal.

    Swings at ``peaks`` (our hits), plus extra "neighbor" hits with no sway under them.
    """
    fps, dur = 12.0, 60.0
    n = int(dur * fps)
    act = np.full(n, 0.4, dtype=np.float32)
    pm = np.zeros(n, dtype=np.float32)
    for a, b in ((4, 26), (34, 56)):
        pm[int(a * fps):int(b * fps)] = 1.0
        act[int(a * fps):int(b * fps)] = 0.9
    peaks = [5.0, 10.0, 15.0, 20.0, 36.0, 41.0, 46.0, 51.0]
    neighbors = [7.5, 12.5, 38.5]
    raw = sorted(peaks + neighbors)
    pose = _synthetic_pose(fps=fps, dur=dur, peaks=tuple(peaks))
    sig = {
        "activity_full": act.tolist(),
        "player_motion_full": pm.tolist(),
        "player_coverage_full": np.ones(n, dtype=np.float32).tolist(),
        "fps": [fps], "duration": [dur], "player_fps": [fps],
        "hit_times_raw": raw,
        "hit_strength_raw": [0.8] * len(raw),
        "hit_confidence_raw": [0.9] * len(raw),
        "hit_times": raw,
        "hit_strength": [0.8] * len(raw),
        "hit_confidence": [0.9] * len(raw),
        "pose_swing_full": [round(float(v), 5) for v in pose.swing],
        "pose_quiet": [float(pose.quiet)],
        "pose_fps": [fps],
        "pose_coverage": [1.0],
    }
    res = AnalysisResult(
        media_id="m_gate", status="done",
        params=AnalysisParams(min_rally_seconds=2.0, pre_roll=0.5, post_roll=0.5,
                              pose_gate_threshold=threshold),
        signals=sig, stats={"audio_reliability": 0.8})
    return res, peaks, neighbors


def test_resegment_regates_hits() -> None:
    """Neighboring-court hits must be removable by resegmentation alone (no AI re-run), idempotently."""
    print("\n重切分：邻场击球重门控")
    res, peaks, neighbors = _hit_gate_result()

    def near(a: float, ts, tol: float = 0.25) -> bool:
        return any(abs(a - t) <= tol for t in ts)

    out = P.resegment(res, res.params)
    kept = list(out.signals.get("hit_times") or [])
    check("邻场击球被剔除", all(not near(t, kept) for t in neighbors), str(kept))
    check("我方击球保留", all(near(t, kept) for t in peaks), str(kept))
    raw_after = list(out.signals.get("hit_times_raw") or [])
    check("原始击球序列未被改写", raw_after == sorted(peaks + neighbors), str(raw_after))

    out2 = P.resegment(out, out.params)
    check("重门控幂等", list(out2.signals.get("hit_times") or []) == kept,
          f"{out2.signals.get('hit_times')}")


def test_resegment_gate_degrades_without_raw() -> None:
    """An old analysis without raw candidates must resegment without crashing and report it."""
    print("\n重切分：缺原始击球时降级")
    res, _peaks, _neighbors = _hit_gate_result()
    res.signals.pop("hit_times_raw", None)
    res.signals.pop("hit_strength_raw", None)
    res.signals.pop("hit_confidence_raw", None)
    out = P.resegment(res, res.params)
    check("无原始序列时不崩", out is not None and out.status == "done")


def test_resegment_respects_use_audio() -> None:
    """Turning hit audio off must make fast resegmentation run the visual-only path, reversibly.

    A full analysis with ``use_audio = false`` already skips the hit path; resegmentation must mirror
    that so toggling the switch and resegmenting is a valid A/B instead of silently reusing the stored
    hit sequence. The raw candidates must stay untouched so switching back restores the gated result.
    """
    print("\n重切分：尊重击球声开关")
    res, peaks, neighbors = _hit_gate_result()

    def near(a: float, ts, tol: float = 0.25) -> bool:
        return any(abs(a - t) <= tol for t in ts)

    params_on = res.params.model_copy(deep=True)
    on = P.resegment(res, params_on)
    kept_on = list(on.signals.get("hit_times") or [])
    check("开音频时保留我方击球", all(near(t, kept_on) for t in peaks), str(kept_on))

    params_off = params_on.model_copy(deep=True)
    params_off.use_audio = False
    hits, trace = P.regate_hits(res.signals, params_off)
    check("regate_hits 关音频返回 None 并标记 audio_off",
          hits is None and trace.get("regate") == "audio_off", str(trace))

    off = P.resegment(res, params_off)
    check("关音频后展示击球序列为空", list(off.signals.get("hit_times") or []) == [],
          str(off.signals.get("hit_times")))
    check("原始击球序列未被改写",
          list(off.signals.get("hit_times_raw") or []) == sorted(peaks + neighbors),
          str(off.signals.get("hit_times_raw")))
    check("关音频后回合不带拍数",
          all(r.features.shot_count == 0 for r in off.rallies),
          str([r.features.shot_count for r in off.rallies]))
    check("门控诊断标记 audio_off",
          (off.stats.get("pose_trace") or {}).get("regate") == "audio_off",
          str(off.stats.get("pose_trace")))

    back = P.resegment(off, params_on)
    check("切回开音频后恢复门控序列",
          list(back.signals.get("hit_times") or []) == kept_on,
          f"{back.signals.get('hit_times')}")


def test_segment_rallies_reuses_player_signal() -> None:
    """When a saved player motion curve is provided, _segment_rallies must use player segmentation.

    This one was bought with a real bug: fast re-segmentation (resegment) has no player boxes,
    and the old version degraded directly to "activity only", so the same project's "re-segment"
    and "re-run analysis" gave different rally counts -- and re-segmentation is the operation
    most often used when tuning segmentation parameters.
    """
    print("\n重切分复用球员信号")
    fps = 12.0
    dur = 120.0
    n = int(dur * fps)
    act = np.full(n, 0.5, dtype=np.float32)          # activity deliberately made flat
    pm = np.zeros(n, dtype=np.float32)
    cov = np.ones(n, dtype=np.float32)
    for a, b in ((10, 40), (60, 95)):
        pm[int(a * fps):int(b * fps)] = 1.0
    pm += np.random.RandomState(0).normal(0, 0.02, n).astype(np.float32)
    pm = np.clip(pm, 0, 1)
    fused = RA.FusedSignal(fps=fps, duration=dur, activity=act,
                           threshold_hi=0.7, threshold_lo=0.5)
    params = AnalysisParams(min_rally_seconds=2.0, pre_roll=0.5, post_roll=0.5)
    ivs, trace = P._segment_rallies(fused, params, dur, player_sig=None,
                                    shuttle_sig=None,
                                    player_motion=pm, player_coverage=cov)
    check("走了球员切分路径", "player_motion" in str(trace.get("method", "")),
          str(trace.get("method")))
    check("标记为复用了存下来的球员信号", trace.get("player_signal") == "reused",
          str(trace.get("player_signal")))
    check("切出 2 个回合", len(ivs) == 2,
          str([(round(x.start, 1), round(x.end, 1)) for x in ivs]))
    check("球员覆盖率被算出来", float(trace.get("player_coverage", 0)) > 0.9,
          str(trace.get("player_coverage")))


def test_merge_by_availability_windows() -> None:
    """All three cases of "pick the best by time span" must be correct.

    This one was bought with measured data. The old implementation wrote "if player segmentation
    has candidates use them, otherwise fall back to activity segmentation" and computed the
    ``valid_ratio`` used for the decision but never used it. The consequence: places where player
    segmentation **deliberately left a gap** (which is exactly the real pause between two rallies)
    got filled in by activity candidates and the rally was glued longer again -- the longest rally
    in testing was 6 seconds longer than after the fix. But conversely, "always leave a gap when
    there are no candidates" treats player-segmentation misses as pauses, and recall dropped by
    6 percentage points in testing. So the third case (player is moving -> cover it) is required.
    """
    print("\n按时间段择优（三种情况）")
    fps = 12.0
    dur = 120.0
    n = int(dur * fps)
    cov = np.ones(n, dtype=np.float32)
    pm = np.zeros(n, dtype=np.float32)                      # default: player not moving

    # 10~30s: player is moving, but player segmentation gave no candidate (simulating a miss) -> should be covered by activity
    # 50~70s: player is not moving, and player segmentation gave no candidate (a real pause) -> should stay empty
    # 90~110s: player segmentation gave a candidate -> should use it directly
    for a, b in ((10, 30), (90, 110)):
        pm[int(a * fps):int(b * fps)] = 1.0

    vis = [RV.RawSegment(start=90.0, end=110.0, score=0.8)]
    act = [RV.RawSegment(start=10.0, end=30.0, score=0.5),
           RV.RawSegment(start=50.0, end=70.0, score=0.5),
           RV.RawSegment(start=90.0, end=110.0, score=0.5)]
    out = P._merge_by_availability(vis, act, [], 0.0, fps, dur,
                                   coverage=cov, player_motion=pm)
    spans = [(round(s.start, 1), round(s.end, 1)) for s in out]
    covered = [(a, b) for a, b in spans]
    has = lambda a, b: any(x[0] <= a + 1 and x[1] >= b - 1 for x in covered)  # noqa: E731

    check("球员在动但球员切分漏检 → 用活跃度兜住", has(12, 28), str(spans))
    check("球员不动且无候选 → 留空（不补）", not has(52, 68), str(spans))
    check("球员切分有候选 → 用它", has(92, 108), str(spans))


def test_annotation_evidence_and_metrics() -> None:
    """Pure functions for annotation evaluation + hit-density evidence.

    Hit density is the only discriminative channel on this clip (measured AUC ~= 0.78); once
    it is computed wrong, the whole "use annotations to calibrate parameters" effort loses meaning.
    """
    print("\n标注评估与击球证据")
    fps = 12.0
    n = int(60 * fps)
    # 10~15s has 6 consecutive shots, no hits for the rest of the time
    hd = RA.hit_density_signal(_hits([10, 10.5, 11, 12, 13, 14]), fps, n)
    check("击球密度长度正确", hd.size == n)
    check("连打的时段密度高", float(hd[int(10 * fps):int(15 * fps)].mean()) > 0.4,
          str(float(hd[int(10 * fps):int(15 * fps)].mean())))
    check("没击球的时段密度低", float(hd[int(40 * fps):int(55 * fps)].mean()) < 0.1,
          str(float(hd[int(40 * fps):int(55 * fps)].mean())))

    # Evidence is 0 when the player is not moving; evidence is high when the player is moving + hitting
    pm = np.zeros(n, dtype=np.float32)
    pm[int(10 * fps):int(15 * fps)] = 1.0
    ev = RV.audio_visual_evidence(pm, hd)
    check("证据曲线长度正确", ev.size == n)
    check("球员不动 → 证据为 0", float(ev[int(40 * fps):int(55 * fps)].max()) < 0.05)
    check("球员在动且连打 → 证据明显", float(ev[int(10 * fps):int(15 * fps)].mean()) > 0.4,
          str(float(ev[int(10 * fps):int(15 * fps)].mean())))

    check("IoU 匹配：完全重合 F1=1", AN.metrics([(0.0, 10.0)], [(0.0, 10.0)])["f1"] == 1.0)
    check("IoU 匹配：不重合 F1=0", AN.metrics([(0.0, 10.0)], [(20.0, 30.0)])["f1"] == 0.0)
    clean = AN.normalize_rallies([{"start": 5, "end": 3}, {"start": -1, "end": 2}, {"start": 8, "end": 9}])
    check("清洗：丢掉倒置、夹取负起点", len(clean) == 2 and clean[0]["start"] == 0.0, str(clean))
    s = AN.suggest([(0.0, 4.0), (10.0, 15.0), (20.0, 23.0)])
    check("建议：给出时长统计", s.get("duration_median") == 4.0 and s.get("count") == 3.0, str(s))


def test_annotation_boundary_metrics() -> None:
    """Boundary-localization stats: paired start/end error quantiles + tolerance-band hit rates/F1.

    Pairing is deliberately loose (IoU>=0.3) so nearly-correct intervals still contribute boundary
    statistics. Within a band, a matched pair counts TP; a pair whose error exceeds the band counts
    on both the FP and FN side (it "moved" the boundary); unmatched predictions/GT add FP/FN.
    """
    print("\n标注边界容差指标")
    bm = AN.boundary_metrics([(0.0, 10.0)], [(0.0, 10.0)], bands=(0.5, 1.0))
    check("完全重合：误差为 0", bm["start"]["errors"] == [0.0] and bm["end"]["errors"] == [0.0])
    check("完全重合：带命中率 1", bm["start"]["bands"]["0.5"] == 1.0
          and bm["start"]["bands_f1"]["0.5"]["f1"] == 1.0)

    # One shifted pair (start +0.6s, end -0.6s, IoU still high) plus one spurious pred and one
    # missed GT; each side: 1 matched, 1 unmatched pred, 1 unmatched GT.
    preds = [(10.6, 29.4), (40.0, 50.0)]
    gt = [(10.0, 30.0), (60.0, 70.0)]
    bm = AN.boundary_metrics(preds, gt, bands=(0.5, 1.0))
    check("宽松配对：配到 1 对", bm["matched"] == 1 and bm["start"]["matched"] == 1)
    check("起点带符号误差正确", len(bm["start"]["errors"]) == 1
          and abs(bm["start"]["errors"][0] - 0.6) < 1e-9, str(bm["start"]["errors"]))
    check("终点带符号误差正确", abs(bm["end"]["errors"][0] - (-0.6)) < 1e-9)
    check("0.5s 带外：命中率 0", bm["start"]["bands"]["0.5"] == 0.0)
    f105 = bm["start"]["bands_f1"]["0.5"]
    check("0.5s 带外：F1=0 且配对外各计 FP/FN", f105 == {"tp": 0, "fp": 2, "fn": 2, "f1": 0.0},
          str(f105))
    f110 = bm["start"]["bands_f1"]["1.0"]
    # The pair is inside the 1.0s band: tp=1, no moved-pair penalty; the unmatched pred and
    # unmatched GT add one FP / one FN: F1 = 2/(2+1+1)=0.5.
    check("1.0s 带内：F1=2/(2+1+1)", f110["tp"] == 1 and f110["fp"] == 1 and f110["fn"] == 1
          and abs(f110["f1"] - 0.5) < 1e-9, str(f110))
    check("分位数给出中位/p90", bm["start"]["median"] == 0.6 and bm["start"]["p90"] == 0.6)

    empty = AN.boundary_metrics([], [])
    check("空输入安全降级", empty["matched"] == 0
          and empty["start"]["median"] is None
          and empty["start"]["bands"] == {"0.5": None, "1.0": None, "1.5": None})

    m = AN.metrics(preds, gt, 0.3, include_pairs=True)
    check("metrics 可选返回配对", len(m["pairs"]) == 1 and len(m["pairs"][0]) == 3)
    m_plain = AN.metrics(preds, gt, 0.3)
    check("metrics 默认不含配对（向后兼容）", "pairs" not in m_plain)


def _lq_result(plateaus, dur, fps=12.0, valleys=()):
    """Synthetic AnalysisResult: 0.25 base player motion, 0.9 plateaus, V-shaped valleys."""
    n = int(dur * fps)
    pm = np.full(n, 0.25, dtype=np.float32)
    for a, b in plateaus:
        pm[int(a * fps):int(b * fps)] = 0.9
    for c in valleys:
        h = int(1.0 * fps)
        ci = int(c * fps)
        for k in range(-h, h + 1):
            j = ci + k
            if 0 <= j < n:
                pm[j] = min(float(pm[j]), 0.25 * abs(k) / h)
    hits = [t for a, b in plateaus for t in np.arange(a + 0.5, b, 1.5)]
    return AnalysisResult(
        media_id="m_lq", status="done", params=AnalysisParams(),
        signals={
            "activity_full": pm.tolist(),
            "player_motion_full": pm.tolist(),
            "player_coverage_full": np.ones(n, dtype=np.float32).tolist(),
            "fps": [fps], "duration": [dur], "player_fps": [fps],
            "hit_times": [round(float(t), 3) for t in hits],
            "hit_strength": [0.8] * len(hits),
            "hit_confidence": [0.9] * len(hits),
        },
    )


def test_annotation_objective_and_grid() -> None:
    """Blended IoU+boundary-band objective and data-driven grid clamps/superset behavior."""
    print("\n容忍带目标与动态网格")
    check("组合分：IoU 权重占主", abs(AN.combo_score(1.0, 0.0) - 0.7) < 1e-9)
    check("组合分：完全正确为 1", AN.combo_score(1.0, 1.0) == 1.0)
    # Same IoU F1 (both pair at thr 0.5, no unmatched) but different boundary localization:
    # at the 0.5s band the 0.1s shift is correct and the 0.9s shift misses.
    g1 = [(10.0, 30.0)]
    p_centered = [(10.1, 29.9)]
    p_shifted = [(10.9, 29.1)]
    b1 = AN.boundary_band_f1(p_centered, g1, band=0.5)
    b2 = AN.boundary_band_f1(p_shifted, g1, band=0.5)
    s1 = AN.combo_score(AN.metrics(p_centered, g1)["f1"], b1)
    s2 = AN.combo_score(AN.metrics(p_shifted, g1)["f1"], b2)
    check("IoU 打平时边界更准者胜", s1 > s2, f"{s1} vs {s2}")
    check("边界带 F1 随误差下降", b1 > b2)

    gt = [(0.0, 3.5), (4.5, 9.0), (10.0, 14.0), (15.0, 19.0),
          (20.0, 24.0), (25.0, 35.0)]  # short 3.5s rally, ~1.0s tight gaps
    grid = dict(AN.dynamic_grid(gt))
    check("动态网格覆盖静态候选", all(set(dict(AN.SEARCH_GRID)[k]) <= set(grid[k])
                                    for k in dict(AN.SEARCH_GRID)))
    for k, (lo, hi) in AN._GRID_CLAMP.items():
        check(f"{k} 候选全部安全夹取", all(lo <= v <= hi for v in grid[k]), str(grid[k]))
    check("短回合拉低 min_rally 候选", min(grid["min_rally_seconds"]) <= 2.5,
          str(grid["min_rally_seconds"]))
    check("紧 gap 拉低 min_rest 候选", min(grid["seg_min_rest"]) < 0.6,
          str(grid["seg_min_rest"]))


def test_annotation_label_quality() -> None:
    """Label audit: structural + signal warnings with guarded snap suggestions; never rewrites."""
    print("\n标注质量审计")

    # --- no signals at all: structural checks still work, no exception
    empty_res = AnalysisResult(media_id="m0", status="done", params=AnalysisParams(),
                               signals={"activity_full": []})
    gt_struct = [(0.0, 18.0), (17.0, 37.0), (40.0, 62.0), (70.0, 90.0), (95.0, 97.0)]
    q = AN.label_quality(empty_res, gt_struct)
    check("无信号：条目数正确", q["count"] == 5)
    codes = {w["code"] for w in q["items"][1]["warnings"]}
    check("无信号：重叠仍报", "overlap" in codes, str(codes))
    check("无信号：时长离群仍报（MAD z）",
          any(w["code"] == "duration_outlier" for w in q["items"][4]["warnings"]),
          str(q["items"][4]["warnings"]))
    check("审计不回写标注", gt_struct[1][0] == 17.0)

    # --- signals: plateaus [12,40] and [44,62], gap valleys at 10/42/64 plus an INTRA-rally
    # valley at 25 which must never become a boundary reference or snap target.
    res = _lq_result([(12, 40), (44, 62)], 70.0, valleys=(10.0, 25.0, 42.0, 64.0))
    # Rally 0 start sits 2.5s off its external-gap valley (warn + snap); rally 1 starts 4s late.
    q = AN.label_quality(res, [(12.5, 40.0), (46.0, 62.0)])
    w0 = {(w["code"], w.get("side")): w for w in q["items"][0]["warnings"]}
    check("起点远离 quiet 谷 → 告警", ("boundary_off_quiet", "start") in w0, str(w0))
    check("给出吸附建议", w0[("boundary_off_quiet", "start")].get("snap_t") == 10.0
          and w0[("boundary_off_quiet", "start")].get("snap_kind") == "quiet",
          str(w0[("boundary_off_quiet", "start")]))
    check("回合内谷绝不作为吸附点",
          all(w.get("snap_t") != 25.0 for w in q["items"][0]["warnings"]),
          str(q["items"][0]["warnings"]))
    w1 = {(w["code"], w.get("side")): w for w in q["items"][1]["warnings"]}
    late = w1.get(("boundary_off_quiet", "start"))
    check("超出吸附半径：告警但无 snap_t", late is not None and late.get("snap_t") is None,
          str(late))
    check("严重级别汇总", q["severity_counts"]["warn"] >= 1)
    check("严重条目标 warn", q["items"][0]["severity"] == "warn")

    # --- snap guard: a candidate valley before the previous rally's end must not be suggested
    res2 = _lq_result([(0, 8), (12, 30)], 40.0, valleys=(10.0,))
    q2 = AN.label_quality(res2, [(12.0, 30.0)])
    # Only one label -> no neighbor guard here; instead verify min-length guard directly.
    cand = AN._snap_candidate(20.0, [(10.0, 1.0)], lo_guard=19.8, hi_guard=30.0)
    check("吸附守卫：候选在合法区间外 → None", cand is None)
    check("审计结果结构稳定", set(q2.keys()) == {"count", "severity_counts", "items"})


def test_annotation_optimizer_runs() -> None:
    """ "Search parameters from annotations" must run through and return usable best parameters (offline, without rerunning AI)."""
    print("\n标注搜参")
    fps = 12.0
    dur = 120.0
    n = int(dur * fps)
    act = np.full(n, 0.4, dtype=np.float32)
    pm = np.zeros(n, dtype=np.float32)
    for a, b in ((10, 35), (60, 90)):
        pm[int(a * fps):int(b * fps)] = 1.0
        act[int(a * fps):int(b * fps)] = 0.9
    hit_times = [t for a, b in ((10, 35), (60, 90)) for t in np.arange(a, b, 1.2)]
    res = AnalysisResult(
        media_id="m_test", status="done",
        params=AnalysisParams(min_rally_seconds=2.0, pre_roll=0.5, post_roll=0.5),
        signals={
            "activity_full": act.tolist(),
            "player_motion_full": pm.tolist(),
            "player_coverage_full": np.ones(n, dtype=np.float32).tolist(),
            "fps": [fps], "duration": [dur], "player_fps": [fps],
            "hit_times": [round(float(t), 3) for t in hit_times],
            "hit_strength": [0.8] * len(hit_times),
            "hit_confidence": [0.9] * len(hit_times),
        },
        stats={"audio_reliability": 0.8},
    )
    out = AN.optimize(res, [(10.0, 34.0), (61.0, 89.0)], focus=(0.0, dur),
                      grid=[("seg_min_core", [0.8, 1.8]), ("min_rally_seconds", [2.0])])
    check("搜到了结果", out["tried"] > 0, str(out.get("tried")))
    check("有最优参数", out["best"] is not None and "seg_min_core" in (out["best"] or {}).get("params", {}))
    check("给出了 baseline 指标", "f1" in out["baseline"])
    check("最优 F1 不低于 baseline F1", out["best"]["f1"] >= out["baseline"]["f1"] - 1e-9,
          f"{out['best']['f1']} < {out['baseline']['f1']}")


def test_annotation_weights_stage() -> None:
    """Stage E runs only with component_*_full; acceptance never regresses the objective."""
    print("\n标注搜参：融合权重 stage")
    f = _fuse_fixture()
    gt = [(10.0, 18.0), (22.0, 30.0), (34.0, 42.0)]
    sig = {
        "activity_full": [round(float(x), 4) for x in f.activity],
        "fps": [f.fps], "duration": [f.duration],
    }
    sig.update({f"component_{k}_full": [round(float(x), 4) for x in v]
                for k, v in f.components.items()})
    res = AnalysisResult(
        media_id="m_w", status="done",
        params=AnalysisParams(min_rally_seconds=2.0, pre_roll=0.3, post_roll=0.3,
                              sample_fps=f.fps),
        signals=sig, stats={"audio_reliability": float(f.audio_reliability)})
    out = AN.optimize(res, gt, focus=(0.0, f.duration),
                      grid=[("min_rally_seconds", [2.0])])
    check("P4-1+ 工程出现 weights stage", "weights" in out["stages"])
    w = out["stages"]["weights"]
    check("stage tried>0 且 trace 键稳定",
          w["tried"] > 0 and w["accepted_moves"] >= 0
          and set(w["search_fields"]) == set(AN.WEIGHT_FIELDS.values()))
    check("best.params 带 5 个 fuse_weight_*",
          all(fld in out["best"]["params"] for fld in AN.WEIGHT_FIELDS.values()))
    check("最终 score 不低于 baseline（接受守卫）",
          out["best"]["score"] >= out["baseline"]["score"] - 0.02,
          f"{out['best']['score']} < {out['baseline']['score']}")
    # Old project: stage skipped entirely and no weight fields leak into best params.
    old_sig = {"activity_full": sig["activity_full"],
               "fps": [f.fps], "duration": [f.duration]}
    res_old = res.model_copy(update={"signals": old_sig})
    out_old = AN.optimize(res_old, gt, focus=(0.0, f.duration),
                          grid=[("min_rally_seconds", [2.0])])
    check("旧工程跳过 weights stage", "weights" not in out_old["stages"])
    check("旧工程 best.params 无 fuse_weight_*",
          not any(k.startswith("fuse_weight_") for k in out_old["best"]["params"]))


def test_annotation_eval_window_intersection() -> None:
    """The evaluation window must be ``focus ∩ annotation bbox``; unannotated regions are unscored.

    Regression for a real bug: with focus set to the whole video and annotations covering only
    the middle section, every AI-detected rally outside the annotated span was counted as a false
    positive, biasing the parameter search toward over-suppressed segmentation. The fix tightens
    the window to the intersection of the saved focus and the annotation bounding box; regions
    without ground truth are simply not scored.
    """
    print("\n标注：评估窗 = focus ∩ 标注外接区间")
    fps = 12.0
    dur = 120.0
    n = int(dur * fps)
    act = np.full(n, 0.4, dtype=np.float32)
    pm = np.zeros(n, dtype=np.float32)
    # Two real rallies: one inside the annotated span, one outside it.
    for a, b in ((10, 35), (60, 90)):
        pm[int(a * fps):int(b * fps)] = 1.0
        act[int(a * fps):int(b * fps)] = 0.9
    hit_times = [t for a, b in ((10, 35), (60, 90)) for t in np.arange(a, b, 1.2)]
    res = AnalysisResult(
        media_id="m_win", status="done",
        params=AnalysisParams(min_rally_seconds=2.0, pre_roll=0.5, post_roll=0.5),
        signals={
            "activity_full": act.tolist(),
            "player_motion_full": pm.tolist(),
            "player_coverage_full": np.ones(n, dtype=np.float32).tolist(),
            "fps": [fps], "duration": [dur], "player_fps": [fps],
            "hit_times": [round(float(t), 3) for t in hit_times],
            "hit_strength": [0.8] * len(hit_times),
            "hit_confidence": [0.9] * len(hit_times),
        },
        stats={"audio_reliability": 0.8},
    )
    # User annotated only the first rally; focus covers the whole video (the old default).
    gt = [(10.0, 34.0)]
    focus = (0.0, dur)
    out = AN.optimize(res, gt, focus=focus,
                      grid=[("seg_min_core", [1.8]), ("min_rally_seconds", [2.0])])
    lo, hi = out["focus"]
    # Window must be the intersection of focus [0,120] and the annotation bbox [10,34].
    check("评估窗收紧到 focus ∩ 标注外接区间", abs(lo - 10.0) < 1e-6 and abs(hi - 34.0) < 1e-6,
          f"focus={out['focus']}")

    # The second rally (60-90) is correctly detected by the AI but lives outside the annotation
    # window. It must NOT appear in predictions that get scored, so it is not a false positive.
    base = out["baseline"]
    preds_inside = all(10.0 - 5 <= p[0] and p[1] <= 34.0 + 5 for p in out["results"][0].get("preds", [])) \
        if "preds" in out["results"][0] else True
    # Directly check the baseline n (prediction count) is small: with the window filter, the
    # 60-90 rally is dropped, so n should be ~1, not 2.
    check("未标注区域的正确预测不计入 FP（baseline n 不大）", int(base.get("n", 0)) <= 1,
          f"n={base.get('n')} fp={base.get('fp')}")
    check("baseline precision 不为 0（未被未标注区拉低）",
          float(base.get("precision", 0.0)) > 0.0, str(base))

    # Edge case: focus does not overlap the annotations at all -> fall back to the bbox.
    out2 = AN.optimize(res, gt, focus=(100.0, 110.0),
                       grid=[("seg_min_core", [1.8]), ("min_rally_seconds", [2.0])])
    lo2, hi2 = out2["focus"]
    check("focus 与标注无交集时回退到标注外接区间",
          abs(lo2 - 10.0) < 1e-6 and abs(hi2 - 34.0) < 1e-6, f"focus={out2['focus']}")

    # Edge case: no focus given -> window defaults to the annotation bbox.
    out3 = AN.optimize(res, gt, focus=None,
                       grid=[("seg_min_core", [1.8]), ("min_rally_seconds", [2.0])])
    lo3, hi3 = out3["focus"]
    check("无 focus 时窗口等于标注外接区间",
          abs(lo3 - 10.0) < 1e-6 and abs(hi3 - 34.0) < 1e-6, f"focus={out3['focus']}")


def test_annotation_hit_gate_stage() -> None:
    """Hit-level labels must unlock the gate stage and produce hit-level precision/recall."""
    print("\n标注：击球门控分阶段搜索")
    res, peaks, neighbors = _hit_gate_result()
    gt = [(4.0, 26.0), (34.0, 56.0)]
    labels = [(t, True) for t in peaks] + [(t, False) for t in neighbors]
    out = AN.optimize(res, gt, grid=[("seg_min_core", [1.0]), ("min_rally_seconds", [2.0])],
                      hit_labels=labels)
    stages = out.get("stages") or {}
    check("返回分阶段结果", "segment" in stages and "gate" in stages, str(list(stages.keys())))
    check("best 含门控阈值", "pose_gate_threshold" in (out.get("best") or {}).get("params", {}),
          str((out.get("best") or {}).get("params")))
    check("best 带击球级指标", (out.get("best") or {}).get("hit") is not None,
          str(out.get("best")))
    base_hit = (out.get("baseline") or {}).get("hit") or {}
    best_hit = (out.get("best") or {}).get("hit") or {}
    check("击球 F1 不低于 baseline", best_hit.get("f1", 0.0) >= base_hit.get("f1", 0.0) - 1e-9,
          f"{best_hit.get('f1')} < {base_hit.get('f1')}")
    check("统计了击球标注数", int(out.get("hit_label_count", 0)) == len(labels),
          str(out.get("hit_label_count")))


def test_preset_hit_params() -> None:
    """Scene presets must carry the calibrated gate params (including booleans) and fit metadata."""
    print("\n场景预设：门控参数")
    from bms.core import presets as PRE

    p = PRE._norm_params({
        "pose_gate_threshold": 0.3, "pose_gate_force": True,
        "pose_gate_one_to_one": False, "bogus": 1.0,
    })
    check("保留门控阈值", p.get("pose_gate_threshold") == 0.3, str(p))
    check("保留布尔字段", p.get("pose_gate_force") is True and p.get("pose_gate_one_to_one") is False,
          str(p))
    check("丢弃未知字段", "bogus" not in p, str(p))
    f = PRE._norm_fit({"source": "annotation", "hit_f1": 0.8, "junk": 123})
    check("fit 白名单", f.get("source") == "annotation" and f.get("hit_f1") == 0.8 and "junk" not in f,
          str(f))


def test_api_filters_and_id_safety() -> None:
    """Regression for two places in the API layer that "silently become match-all".

    * The bulk filter sends camelCase from the frontend; if the backend only recognizes
      snake_case, every condition except tags is ignored -- one "bulk exclude" hits the
      entire project;
    * project/media ids are concatenated into file paths; illegal characters must be rejected,
      otherwise one can escape data/projects.
    """
    print("\n批量筛选字段与 id 安全")
    from bms.core import store as ST
    from bms.core.models import Rally, RallyFeatures, RallyScores
    from bms.main import _match_filter

    r = Rally(duration=12.0, keep=False, starred=True, tags=["多拍"],
              scores=RallyScores(total=77.0),
              features=RallyFeatures(shot_count=9, confidence=0.8))
    check("筛选：空条件匹配", _match_filter(r, {}))
    check("筛选：camelCase minScore 生效", not _match_filter(r, {"minScore": 90}))
    check("筛选：camelCase minScore 通过", _match_filter(r, {"minScore": 70}))
    check("筛选：camelCase minDuration 生效", not _match_filter(r, {"minDuration": 20}))
    check("筛选：camelCase minShots 生效", not _match_filter(r, {"minShots": 12}))
    check("筛选：camelCase minConfidence 生效", not _match_filter(r, {"minConfidence": 0.95}))
    check("筛选：starredOnly 生效", not _match_filter(Rally(starred=False), {"starredOnly": True}))
    check("筛选：keepOnly 排除 keep=False", not _match_filter(r, {"keepOnly": True}))
    check("筛选：snake_case 仍兼容", not _match_filter(r, {"min_score": 90}))

    for bad in ("../x", "a/b", "a\\b", "..", "p_ok.analysis"):
        try:
            ST._path(bad)
            check(f"拒绝非法工程 id {bad!r}", False)
        except ValueError:
            check(f"拒绝非法工程 id {bad!r}", True)
    try:
        ST._analysis_path("p_ok", "../x")
        check("拒绝非法素材 id", False)
    except ValueError:
        check("拒绝非法素材 id", True)
    check("合法 id 正常", ST._path("p_abc123").name == "p_abc123.json",
          ST._path("p_abc123").name)


def test_job_media_binding() -> None:
    """The prepare job must carry media_id so the UI can attach generation progress to the matching media card."""
    print("\n任务与素材绑定")
    from bms.core.jobs import JobManager

    mgr = JobManager()
    job = mgr.submit("prepare", "准备素材 t", lambda j: {"ok": True}, media_id="m_demo")
    job.join(5)
    check("任务带上 media_id", job.info.media_id == "m_demo", str(job.info.media_id))
    check("任务正常完成", job.info.status == "done", job.info.status)
    check("任务列表保留 media_id", mgr.list()[0].media_id == "m_demo", str(mgr.list()[0].media_id))
    plain = mgr.submit("export", "导出", lambda j: None)
    check("未指定时不带 media_id", plain.info.media_id is None, str(plain.info.media_id))


def test_analyze_batch_scope() -> None:
    """Two key conventions for batch analysis: each media uses its own court calibration, and the batch route exists.

    If ``_stored_court_poly`` reads the wrong key, batch analysis treats every media as uncalibrated;
    manual calibration silently stops working and results get worse without any error.
    """
    print("\n批量分析：范围与场地标定")
    from bms.core.models import Project
    from bms.main import _stored_court_poly, app

    proj = Project(name="t")
    check("无标定返回 None", _stored_court_poly(proj, "m_a") is None)
    proj.ui = {"court_polys": {"m_a": [[0, 0], [1, 0], [1, 1], [0, 1]]}}
    check("读到新键 court_polys",
          _stored_court_poly(proj, "m_a") == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
          str(_stored_court_poly(proj, "m_a")))
    check("其他素材仍为 None", _stored_court_poly(proj, "m_b") is None)
    proj.ui = {"court_quads": {"m_c": [[0, 0], [2, 0], [2, 2], [0, 2]]}}
    check("兼容旧键 court_quads", _stored_court_poly(proj, "m_c") is not None)
    proj.ui = {"court_polys": {"m_d": [[0, 0], [1, 0], [1, 1]]}}
    check("点数不足被忽略", _stored_court_poly(proj, "m_d") is None)

    routes = {getattr(r, "path", None) for r in app.routes}
    check("批量分析路由存在", "/api/projects/{pid}/analyze-batch" in routes)


def test_preset_helpers() -> None:
    """Scene preset parameter whitelist / polygon validation / id path safety."""
    print("\n场景预设：参数与标定校验")
    from bms.core import presets as PRE
    from bms.config import DATA_DIR, PRESETS_DIR

    p = PRE._norm_params({
        "seg_min_core": 1.2,
        "min_rally_seconds": "3",
        "bogus": 9,
        "seg_min_rest": float("nan"),
    })
    check("只保留白名单字段", set(p) == {"seg_min_core", "min_rally_seconds"}, str(p))
    check("字符串数字被转成 float", p["min_rally_seconds"] == 3.0)

    good = [[0.1, 0.9], [0.9, 0.9], [0.6, 0.5], [0.4, 0.5]]
    check("合法多边形保留", PRE._norm_poly(good) == good)
    check("点数不足被拒", PRE._norm_poly(good[:3]) is None)
    check("越界坐标被拒", PRE._norm_poly([[0.1, 0.9], [1.4, 0.9], [0.6, 0.5], [0.4, 0.5]]) is None)
    check("面积为零被拒", PRE._norm_poly([[0.2, 0.2], [0.2, 0.2], [0.2, 0.2], [0.2, 0.2]]) is None)
    check("空多边形为 None", PRE._norm_poly(None) is None)

    for bad in ("../x", "ps_a/b", "x", "ps_..", "ps_a.b"):
        try:
            PRE._path(bad)
            check(f"拒绝非法预设 id {bad!r}", False)
        except ValueError:
            check(f"拒绝非法预设 id {bad!r}", True)
    check("合法 id 正常", PRE._path("ps_abc123").name == "ps_abc123.json", PRE._path("ps_abc123").name)
    check("预设目录在 data 下", PRESETS_DIR.parent == DATA_DIR, str(PRESETS_DIR))


def test_bulk_media_purge() -> None:
    """Bulk removal of media: media / analysis results / timeline clips must all be cleared, returning only the delete count."""
    print("\n批量移除素材")
    from bms.core.models import AnalysisResult, Clip, MediaInfo, Project, Timeline, Track
    from bms.main import _purge_media

    p = Project(name="t")
    ma = MediaInfo(path="C:/v/a.mp4", name="a")
    mb = MediaInfo(path="C:/v/b.mp4", name="b")
    mc = MediaInfo(path="C:/v/c.mp4", name="c")
    p.media = [ma, mb, mc]
    p.analyses = {ma.id: AnalysisResult(media_id=ma.id), mb.id: AnalysisResult(media_id=mb.id)}
    p.timeline = Timeline(
        tracks=[
            Track(
                clips=[
                    Clip(media_id=ma.id, src_in=0, src_out=3, tl_start=0),
                    Clip(media_id=mb.id, src_in=0, src_out=5, tl_start=10),
                ]
            )
        ]
    )

    removed = _purge_media(p, {mb.id, mc.id})
    check("删除数量正确", removed == 2, str(removed))
    check("media 只剩 a", [m.id for m in p.media] == [ma.id])
    check("分析结果清掉 b、保留 a", set(p.analyses) == {ma.id}, str(set(p.analyses)))
    check("时间线只剩 a 的片段", [c.media_id for c in p.timeline.tracks[0].clips] == [ma.id])
    check("不存在的 id 不报错", _purge_media(p, {"m_nope"}) == 0)


def test_cancel_jobs_for_media() -> None:
    """When media is deleted, queued/running jobs for it must be cancelled."""
    print("\n按素材取消任务")
    from bms.core.jobs import JobManager

    mgr = JobManager()
    j1 = mgr.submit("prepare", "p1", None, media_id="m_a")  # fn=None -> stays queued
    j2 = mgr.submit("prepare", "p2", None, media_id="m_b")
    check("取消命中 1 个", mgr.cancel_for_media({"m_a"}) == 1)
    check("m_a 任务已取消", j1.info.status == "cancelled", j1.info.status)
    check("m_b 任务不受影响", j2.info.status == "queued", j2.info.status)
    check("空集合返回 0", mgr.cancel_for_media(set()) == 0)


def test_cancel_suppresses_error_status() -> None:
    """A job cancelled mid-run that then raises must report 'cancelled', not a spurious 'error'.

    Regression: ``cancel()`` sets the status immediately, but ``Job._run``'s except branch used to
    overwrite it with ``error`` when the worker raised because of the cancellation (e.g. ffmpeg
    killed by the cancel flag inside ``ensure_proxy``), so the UI showed a failure for a cancel.
    """
    print("\n取消任务不误报错误")
    from bms.core.jobs import Job

    def fn(job: Job) -> None:
        job.cancel()
        raise RuntimeError("worker raised because it was cancelled")

    j = Job("analyze", "t", fn)
    j.start()
    j.join(5.0)
    check("取消优先于异常", j.info.status == "cancelled", j.info.status)
    check("取消后无错误信息", not j.info.error, j.info.error or "")


def test_cross_media_rescore() -> None:
    """Cross-media batch rescoring: scores enter the same distribution instead of being normalized per media.

    Scoring uses within-batch percentiles. If ``cross_media`` is ignored and each media is scored
    as its own batch, a single rally gets full marks on its own and is not comparable across media
    at all; this test case targets exactly "looks successful but not batched".
    """
    print("\n跨素材统一重算评分")
    from bms.analysis import scoring as SC
    from bms.core.models import AnalysisResult, Project, Rally, RallyFeatures, RallyScores
    from bms.main import _rescore_cross_media, _rescore_features, _rescore_per_media, _rescore_quality

    def mk(speed: float) -> Rally:
        return Rally(
            duration=10.0,
            clip_start=0.0,
            clip_end=10.0,
            # Simulate the per-media scores already computed during analysis: switching back to per-media must restore them unchanged
            scores=RallyScores(total=42.0, length=30.0, intensity=40.0, technique=50.0,
                               excitement=60.0, production=70.0),
            features=RallyFeatures(
                duration=10.0,
                shot_count=10,
                tempo=2.0,
                finish_intensity=2.0,
                motion_energy=0.5,
                motion_peak=1.0,
                shuttle_speed_p95=5.0,
                confidence=1.0,
                hit_strength_p90=0.5,
                player_speed_mean=speed,
                player_speed_max=speed + 1.0,
                shuttle_presence=1.0,
                quality_sharpness=0.8,
                quality_shake=0.1,
                quality_subject_size=0.3,
            ),
        )

    p = Project(name="t")
    ra = AnalysisResult(media_id="m_a", status="done", rallies=[mk(10.0)])
    rb = AnalysisResult(media_id="m_b", status="done", rallies=[mk(1.0)])
    # Per-media mode during analysis: stats.weights is recorded as balanced
    ra.stats["weights"] = "balanced"
    rb.stats["weights"] = "balanced"
    p.analyses = {"m_a": ra, "m_b": rb}

    n = _rescore_cross_media(p, "balanced", SC.PRESETS["balanced"])
    check("合批覆盖全部素材", n == 2, str(n))
    cross_a = ra.rallies[0].scores.total
    cross_b = rb.rallies[0].scores.total
    check("快的一方分数更高（进了同一分布）", cross_a > cross_b, f"{cross_a} vs {cross_b}")
    check("写入 cross 缓存键", "cross:balanced" in ra.stats.get("score_cache", {}))
    check(
        "权重标记为 cross 口径",
        ra.stats.get("weights") == "cross:balanced",
        str(ra.stats.get("weights")),
    )
    check(
        "覆盖前保存了逐素材原分",
        ra.stats["score_cache"]["balanced"][0]["total"] == 42.0,
        str(ra.stats["score_cache"]["balanced"][0]["total"]),
    )

    # Switch back to per-media mode: should restore from cache unchanged, not approximately recompute from features
    _rescore_per_media(p, "balanced", SC.PRESETS["balanced"], None)
    check("切回逐素材恢复原分", ra.rallies[0].scores.total == 42.0, str(ra.rallies[0].scores.total))
    check("权重标记回 balanced", ra.stats.get("weights") == "balanced", str(ra.stats.get("weights")))

    # Switch back to all media: cross cache hits, result matches the first run
    _rescore_cross_media(p, "balanced", SC.PRESETS["balanced"])
    check(
        "二次统一重算命中缓存、结果一致",
        ra.rallies[0].scores.total == cross_a and rb.rallies[0].scores.total == cross_b,
        f"{ra.rallies[0].scores.total}/{rb.rallies[0].scores.total}",
    )

    # Control in the other direction: when computed per media, a single rally is normalized on its own
    # and the two sides tie -- which is exactly why batching is required instead of per-media recomputation.
    solo_a = SC.score_rallies(
        [_rescore_features(mk(10.0))], SC.PRESETS["balanced"], [_rescore_quality(mk(10.0))]
    )[0]["total"]
    solo_b = SC.score_rallies(
        [_rescore_features(mk(1.0))], SC.PRESETS["balanced"], [_rescore_quality(mk(1.0))]
    )[0]["total"]
    check(
        "逐素材重算时双方同分（说明合批才有区分度）",
        abs(solo_a - solo_b) < 1e-6,
        f"{solo_a} vs {solo_b}",
    )


def test_exports_registry() -> None:
    """Export registry: register / list / lookup by id / illegal id does not leak the path."""
    print("\n导出登记表")
    import tempfile
    from pathlib import Path

    from bms.core import exports as EXP

    tmp = Path(tempfile.mkdtemp(prefix="bms_exports_"))
    f1 = tmp / "a.mp4"
    f2 = tmp / "b_01.mp4"
    f1.write_bytes(b"x" * 100)
    f2.write_bytes(b"y" * 200)

    # Point the registry at a temp directory; do not touch the real data/exports
    old_index = EXP._INDEX
    EXP._INDEX = tmp / "index.json"
    try:
        entries = EXP.register([f1, f2], "separate", "grp")
        check("登记返回两条", len(entries) == 2, str(len(entries)))
        ids = {e["id"] for e in entries}
        check("id 以 e_ 开头", all(i.startswith("e_") for i in ids), str(ids))

        listed = EXP.list_all()
        paths = {x["path"] for x in listed}
        check("列表包含登记文件", str(f1) in paths and str(f2) in paths, str(paths))

        check("按 id 命中真实路径", EXP.find(entries[0]["id"]) == f1, str(EXP.find(entries[0]["id"])))
        check("非法 id 返回 None", EXP.find("e_notexist") is None)
        check("第二个 id 也能命中", EXP.find(entries[1]["id"]) == f2)

        # Re-registering the same path must not create a duplicate entry
        EXP.register([f1], "merge", "grp")
        check("同路径不重复登记", len(EXP.list_all()) == 2, str(len(EXP.list_all())))
        check("登记表文件已写入", EXP._INDEX.exists())
    finally:
        EXP._INDEX = old_index

    from bms.config import EXPORT_DIR
    from bms.main import _resolve_output_dir
    from fastapi import HTTPException

    check("缺省导出目录", _resolve_output_dir(None) == EXPORT_DIR)
    sub = tmp / "out"
    check("自定义目录会自动创建", _resolve_output_dir(str(sub)) == sub and sub.is_dir())
    try:
        _resolve_output_dir("relative/dir")
        check("相对路径被拒", False)
    except HTTPException:
        check("相对路径被拒", True)


def test_speech_phrase_sanitize() -> None:
    """Voice command sanitizing: at most 2, each <=3 characters, strip whitespace and dedupe; the params layer must enforce the same rules."""
    print("\n语音口令：清洗")
    from bms.analysis import speech as SP

    check("去空白", SP.sanitize_phrases([" 好 球 "]) == ["好球"])
    check("超长截断到 3 字", SP.sanitize_phrases(["好球啊啊啊"]) == ["好球啊"])
    check("去重", SP.sanitize_phrases(["好球", "好球", "漂亮"]) == ["好球", "漂亮"])
    check("最多 2 个", SP.sanitize_phrases(["一", "二", "三"]) == ["一", "二"])
    check("空串 / 非字符串被丢", SP.sanitize_phrases(["", "  ", 123, "好球"]) == ["好球"])
    check("None / 非列表退化为空", SP.sanitize_phrases(None) == [] and SP.sanitize_phrases("好球") == [])

    p = AnalysisParams(speech_phrases=["好球", "漂亮", "加油"])
    check("参数层限 2 个", p.speech_phrases == ["好球", "漂亮"], str(p.speech_phrases))
    check("参数层截断到 3 字", AnalysisParams(speech_phrases=["非常漂亮"]).speech_phrases == ["非常漂"])
    check("加分值封顶 30", AnalysisParams(speech_bonus_points=999).speech_bonus_points == 30.0)
    check("加分值不为负", AnalysisParams(speech_bonus_points=-5).speech_bonus_points == 0.0)
    check("默认关闭", AnalysisParams().use_speech is False)


def test_speech_contract_and_degrade() -> None:
    """Speech module interface contract + silent degradation (must not raise / must not load a model for a bad path)."""
    print("\n语音口令：接口与降级")
    from bms.analysis import speech as SP

    sig = inspect.signature(SP.detect_phrases)
    for kw in ("phrases", "on_progress", "cancel", "model", "fuzzy"):
        check(f"detect_phrases 接受 {kw}", kw in sig.parameters)

    empty = SP.detect_phrases("nope.wav", [])
    check("没有短语时直接空结果", not empty.available and empty.times.size == 0)

    # Unreadable audio must return available=False instead of raising, and must fail before
    # loading a (possibly large, possibly downloading) model.
    det = SP.detect_phrases("nope.wav", ["好球"])
    check("缺失时降级不抛异常", det.available is False, str(det.trace))
    check("降级时 trace 有说明", bool(det.trace), str(det.trace))
    check("读音频失败不加载模型", "error" in det.trace and "model" not in det.trace, str(det.trace))

    model = SP.resolve_model()
    check("resolve_model 不抛异常且返回字符串", isinstance(model, str) and bool(model), str(model))


def test_speech_near_match() -> None:
    """Near-homophone matching: 到球/倒球 count as 好球, but 打球/要求 must not."""
    print("\n语音口令：近音匹配")
    from bms.analysis import speech as SP

    check("精确命中", SP._matches("好球", "好球", False))
    check("关掉近音时 到球 不命中", not SP._matches("到球", "好球", False))

    if SP._pinyin("好球") is None:
        print("  （未安装 pypinyin，跳过近音断言）")
        return
    check("近音：到球 -> 好球", SP._matches("到球", "好球", True))
    check("近音：倒球 -> 好球", SP._matches("倒球", "好球", True))
    check("近音：打球 -> 好球（韵母不同，拒绝）", not SP._matches("打球", "好球", True))
    check("近音：要求 -> 好球（末字不同，拒绝）", not SP._matches("要求", "好球", True))
    check("近音：好好 -> 好球（末字不同，拒绝）", not SP._matches("好好", "好球", True))
    check("近音：长度不同拒绝", not SP._matches("好球啊", "好球", True))


def test_speech_refine_windows_and_merge() -> None:
    """Short-window recall pass: windows cover each region, and merging keeps the most confident event per shout."""
    print("\n语音口令：短窗复核")
    from bms.analysis import speech as SP

    # Candidate windows cover [10, 20] with 3 s windows, 1.5 s hop.
    ws = SP._candidate_windows([(10.0, 20.0)], window_s=3.0, hop_s=1.5, pad_s=0.0)
    check("窗口非空", bool(ws), str(ws))
    check("首窗从区间起点开始", ws[0][0] == 10.0 and abs(ws[0][1] - 13.0) < 1e-6, str(ws[0]))
    check("末窗覆盖区间末尾", max(w[1] for w in ws) >= 20.0 - 1e-6, str(ws[-1]))
    # The final window is a tail-cover window and may not follow the hop.
    check("步长 1.5s", all(abs(ws[i + 1][0] - ws[i][0] - 1.5) < 1e-6 for i in range(len(ws) - 2)), str(ws))
    check("pad 不越过 0", SP._candidate_windows([(1.0, 2.0)], window_s=3.0, hop_s=1.5, pad_s=1.0)[0][0] == 0.0)
    check("空区间返回空", SP._candidate_windows([]) == [])

    # Scored matching exposes the anchor (last-char) probability for de-duplication.
    scored = SP._match_tokens_scored([("好", 5.0, 0.8), ("球", 5.5, 0.9)], ["好球"], False, 0.4)
    check("带分数匹配返回锚概率", len(scored) == 1 and abs(scored[0][2] - 0.9) < 1e-9, str(scored))

    # Merge: within 1.5 s only the most confident survives; far events are kept; a trusted base event wins.
    base = [(12.0, "好球")]
    refined = [(12.3, "好球", 0.9, 0.1), (12.6, "好球", 0.5, 0.1), (20.0, "好球", 0.8, 0.2)]
    check("同一声喊去重后保留基线与远端",
          SP._merge_events(base, refined, dedup_s=1.5) == [(12.0, "好球"), (20.0, "好球")],
          str(SP._merge_events(base, refined, dedup_s=1.5)))
    check("无基线时保留锚概率最高",
          SP._merge_events([], refined, dedup_s=1.5) == [(12.3, "好球"), (20.0, "好球")],
          str(SP._merge_events([], refined, dedup_s=1.5)))

    # Bad audio degrades instead of raising.
    det = SP.refine_phrases("nope.wav", ["好球"], [(10.0, 20.0)])
    check("复核缺失音频时降级", det.available is False, str(det.trace))

    # Base events arrive from pipeline._speech_events as {"t","phrase"} dicts; the two shapes must
    # both normalize. Regression: the old tuple-only comprehension crashed with float('t') whenever
    # the main pass had found any phrase, failing the whole analysis at stage=speech.
    check("dict 形基础事件归一化",
          SP._normalize_base_events([{"t": 1.5, "phrase": "好球"}]) == [(1.5, "好球")],
          str(SP._normalize_base_events([{"t": 1.5, "phrase": "好球"}])))
    check("tuple 形基础事件归一化",
          SP._normalize_base_events([(1.5, "好球")]) == [(1.5, "好球")],
          str(SP._normalize_base_events([(1.5, "好球")])))
    check("非法基础事件跳过", SP._normalize_base_events([("x", "y")]) == [],
          str(SP._normalize_base_events([("x", "y")])))
    det_dict = SP.refine_phrases("nope.wav", ["好球"], [(10.0, 20.0)],
                                 base_events=[{"t": 1.0, "phrase": "好球"}])
    check("dict 形基础事件不崩溃", det_dict.available is False, str(det_dict.trace))


def test_speech_bonus_and_scoring() -> None:
    """Voice commands add bonus points to rallies when matched by interval; scoring includes it in the total and tags it."""
    print("\n语音口令：加分与评分")
    from bms.analysis import scoring as SC
    from bms.analysis import speech as SP  # noqa: F401

    params = AnalysisParams(use_speech=True, speech_phrases=["好球"], speech_bonus_points=10.0)
    ivs = [RA.RallyInterval(start=10.0, end=15.0), RA.RallyInterval(start=30.0, end=35.0)]
    events = [{"t": 12.0, "phrase": "好球"}, {"t": 15.5, "phrase": "好球"},
              {"t": 60.0, "phrase": "漂亮"}]
    P._apply_speech_bonus(ivs, events, params)
    check("命中短语写入 features", ivs[0].features["speech_phrases"] == ["好球"],
          str(ivs[0].features.get("speech_phrases")))
    check("同一短语喊多次只加一次（含容差命中）", ivs[0].features["speech_bonus"] == 10.0,
          str(ivs[0].features["speech_bonus"]))
    check("没命中的回合为 0", ivs[1].features["speech_bonus"] == 0.0,
          str(ivs[1].features["speech_bonus"]))

    # Switch to two phrases: the bonus doubles
    p2 = AnalysisParams(speech_phrases=["好球", "漂亮"], speech_bonus_points=8.0)
    iv2 = [RA.RallyInterval(start=10.0, end=15.0)]
    P._apply_speech_bonus(iv2, [{"t": 11.0, "phrase": "好球"}, {"t": 13.0, "phrase": "漂亮"}], p2)
    check("两个不同短语加两倍", iv2[0].features["speech_bonus"] == 16.0,
          str(iv2[0].features["speech_bonus"]))

    # Reused on re-segmentation: clearing events must clear the old bonus
    P._apply_speech_bonus(iv2, [], p2)
    check("无事件时清零", iv2[0].features["speech_bonus"] == 0.0)

    # _build_rally maps the bonus/phrases into RallyFeatures
    P._apply_speech_bonus(iv2, [{"t": 11.0, "phrase": "好球"}], p2)
    r = P._build_rally(0, iv2[0], None, P._MediaStub(60.0), {"total": 50.0}, p2)
    check("RallyFeatures 带口令加分", r.features.speech_bonus == 8.0, str(r.features.speech_bonus))
    check("RallyFeatures 带口令短语", r.features.speech_phrases == ["好球"], str(r.features.speech_phrases))

    # Scoring: a rally with a bonus has a higher total, capped at 100, and is tagged with the voice command
    base = {
        "duration": 20.0, "shot_count": 15.0, "tempo": 1.5, "finish_tempo": 1.5,
        "activity_mean": 0.5, "activity_peak": 0.9, "player_speed_mean": 0.5,
        "player_speed_max": 0.8, "motion_mean": 0.5, "motion_peak": 0.7,
        "hit_strength_p90": 0.6, "shuttle_speed_p90": 0.5, "shuttle_presence": 0.5,
        "confidence": 1.0,
    }
    f0 = dict(base, speech_bonus=0.0)
    f1 = dict(base, speech_bonus=10.0, speech_phrases=["好球"])
    scores = SC.score_rallies([f0, f1], SC.PRESETS["balanced"])
    check("命中口令的回合总分更高", scores[1]["total"] > scores[0]["total"],
          f"{scores[1]['total']} vs {scores[0]['total']}")
    check("总分封顶 100", scores[1]["total"] <= 100.0, str(scores[1]["total"]))
    check("命中短语成为标签", "好球" in scores[1]["tags"], str(scores[1]["tags"]))


def test_i18n_catalogs() -> None:
    """中英目录键集合一致、插值与回退可用；rally 标签使用稳定代码并可迁移旧中文。"""
    print("\ni18n 双语目录")
    from bms import i18n
    from bms.locales import en as EN
    from bms.locales import zh as ZH
    from bms.analysis import scoring as SC

    zh_keys = set(ZH.MESSAGES)
    en_keys = set(EN.MESSAGES)
    check("中文目录非空", len(zh_keys) > 50, str(len(zh_keys)))
    check("中英键集合一致", zh_keys == en_keys,
          f"only-zh={sorted(zh_keys - en_keys)[:5]} only-en={sorted(en_keys - zh_keys)[:5]}")
    check("所有键都有非空文案", all(ZH.MESSAGES[k] for k in zh_keys) and all(EN.MESSAGES[k] for k in en_keys))

    i18n.set_lang("en")
    check("英文插值", i18n.tr("job.done", seconds=1.5) == "Done (1.5s)", i18n.tr("job.done", seconds=1.5))
    i18n.set_lang("zh")
    check("中文插值", i18n.tr("job.done", seconds=1.5) == "完成（1.5s）", i18n.tr("job.done", seconds=1.5))
    check("缺失键回退为键名", i18n.tr("no.such.key") == "no.such.key")
    check("归一化语言码", i18n.parse_lang("en-US,en;q=0.9") == "en" and i18n.parse_lang("zh-CN") == "zh")

    check("tag 为稳定 ASCII 代码",
          all(t.isascii() for t in (
              SC.TAG_ULTRA_LONG, SC.TAG_MANY_SHOTS, SC.TAG_FAST_TEMPO, SC.TAG_HIGH_SCORE)))
    check("旧中文标签可迁移", SC.migrate_tags(["多拍", "快节奏"]) == [SC.TAG_MANY_SHOTS, SC.TAG_FAST_TEMPO],
          str(SC.migrate_tags(["多拍", "快节奏"])))
    check("迁移保留未知（口令）标签", SC.migrate_tags(["好球", "多拍"]) == ["好球", SC.TAG_MANY_SHOTS],
          str(SC.migrate_tags(["好球", "多拍"])))


def test_proxy_source_fallback() -> None:
    """A deleted derived file must not be handed to cv2/ffmpeg: proxy_source falls back to the source."""
    print("\n代理路径回退")
    import tempfile

    from bms.core import media as M
    from bms.core.models import MediaInfo

    tmp = Path(tempfile.mkdtemp(prefix="bms_proxy_"))
    src = tmp / "clip.mp4"
    src.write_bytes(b"x")
    proxy = tmp / "clip_proxy.mp4"
    proxy.write_bytes(b"x")

    m = MediaInfo(id="m_test", path=str(src), name="clip", duration=1.0, width=3840, height=2160)
    m.proxy_path = str(proxy)
    check("代理存在时用代理", M.proxy_source(m) == proxy, str(M.proxy_source(m)))

    proxy.unlink()
    check("代理被删后回退到源片", M.proxy_source(m) == src, str(M.proxy_source(m)))

    m.proxy_path = None
    check("无代理字段时用源片", M.proxy_source(m) == src, str(M.proxy_source(m)))

    # Annotations are keyed on the deterministic proxy name; clearing proxy_path must not change it.
    from bms.analysis.annotation import annotation_path

    stem = M.proxy_stem(m)
    check("代理名为确定性公式", stem == "clip_m_test_960x540", stem)
    at_without = annotation_path(m)
    m.proxy_path = str(tmp / f"{stem}.mp4")
    at_with = annotation_path(m)
    check("清缓存前后标注文件名一致",
          at_with.name == at_without.name == f"{stem}.anno.json", f"{at_with.name} / {at_without.name}")


def test_clear_derived_paths_on_cache_clear() -> None:
    """Clearing the cache must drop the stored derived paths, or the stale ones keep breaking consumers."""
    print("\n清缓存清理派生路径")
    from bms.core.models import MediaInfo
    from bms.core.store import clear_derived_paths

    m = MediaInfo(id="m_x", path="C:/v/clip.mp4")
    m.proxy_path = "C:/cache/proxies/clip.mp4"
    m.proxy_fps = 30.0
    m.proxy_width = 960
    m.proxy_height = 540
    m.audio_path = "C:/cache/audio/clip.wav"
    m.poster = "C:/cache/thumbs/clip.jpg"

    check("仅清代理时返回 True", clear_derived_paths(m, {"proxies"}) is True)
    check("代理字段被清空",
          m.proxy_path is None and m.proxy_fps is None and m.proxy_width is None and m.proxy_height is None)
    check("音轨/封面保留", m.audio_path is not None and m.poster is not None)

    check("清音轨与封面返回 True", clear_derived_paths(m, {"audio", "thumbs"}) is True)
    check("音轨与封面被清空", m.audio_path is None and m.poster is None)
    check("再次清空无变化返回 False", clear_derived_paths(m, {"proxies", "audio", "thumbs"}) is False)
    check("源片路径不受影响", m.path == "C:/v/clip.mp4")


def test_speech_regions_use_interval_bounds() -> None:
    """The speech recall pass must read start/end from a RallyInterval; clip_start does not exist.

    Regression: the new pass used ``iv.clip_start``/``iv.clip_end``, which raised AttributeError
    and (through run_analysis' outer except) failed the whole analysis whenever speech was on.
    """
    print("\n语音召回区间")
    from bms.analysis import pipeline as P
    from bms.analysis.rally import RallyInterval

    iv = RallyInterval(start=3.5, end=9.25)
    regions = P._speech_regions([iv])
    check("区间取自 start/end", regions == [(3.5, 9.25)], str(regions))
    check("RallyInterval 没有 clip_start", not hasattr(iv, "clip_start"))


def test_logging_setup() -> None:
    """setup_logging installs a file sink and is idempotent (repeated calls add no sinks)."""
    print("\nloguru 日志初始化")
    import tempfile

    from loguru import logger

    from bms.logging_setup import setup_logging

    tmp = Path(tempfile.mkdtemp(prefix="bms_log_"))
    d1 = setup_logging(level="INFO", log_dir=tmp, force=True)
    check("返回日志目录", Path(d1) == tmp, str(d1))

    logger.info("bms logging self-test {}", 123)
    logger.complete()  # flush the enqueued file sink
    files = list(tmp.glob("bms_*.log"))
    check("写入日志文件", bool(files), str(files))
    if files:
        text = files[0].read_text(encoding="utf-8")
        check("日志内容落盘", "logging self-test 123" in text, text[-200:])

    n1 = len(logger._core.handlers)  # noqa: SLF001 - loguru exposes no public handler count
    setup_logging(log_dir=tmp)
    check("重复调用不重复加 sink", len(logger._core.handlers) == n1,
          f"{n1} -> {len(logger._core.handlers)}")


# ------------------------------------------------------------------ API hardening regressions


def test_rally_patch_validation() -> None:
    """A rally patch must be whitelisted and type-checked: pydantic v2 does not validate setattr,
    so a bad value used to corrupt the sidecar and the next load dropped the whole analysis."""
    print("\n回合补丁校验")
    from fastapi import HTTPException

    from bms.core.models import Rally
    from bms.main import _patched_rally

    r = Rally(id="r_1", index=3, start=1.0, end=5.0, clip_start=0.5, clip_end=5.5)
    up = _patched_rally(r, {"keep": False, "starred": True, "note": "ok", "tags": ["many_shots"]})
    check("合法补丁生效",
          up.keep is False and up.starred is True and up.note == "ok" and up.tags == ["many_shots"])
    check("未改字段保留", up.id == "r_1" and up.start == 1.0 and up.index == 3)
    for bad in ({"clip_end": "not-a-number"}, {"tags": "many_shots"}, {"keep": "maybe"}):
        try:
            _patched_rally(r, bad)
            check(f"非法补丁被拒 {bad}", False)
        except HTTPException as e:
            check(f"非法补丁被拒 {bad}", e.status_code == 400)
    try:
        _patched_rally(r, {"bogus": 1})
        check("未知字段被拒", False)
    except HTTPException as e:
        check("未知字段被拒", e.status_code == 400)
    check("拒绝后原对象未被改动", r.clip_end == 5.5 and r.keep is True)


def test_invalid_weights_rejected() -> None:
    """An unknown scoring weight must be a 400 (never stored into stats["weights"])."""
    print("\n非法评分口径")
    from fastapi import HTTPException

    from bms.main import _validated_weights

    check("合法口径返回", _validated_weights("balanced") == "balanced")
    check("缺省为 balanced", _validated_weights(None) == "balanced")
    check("cross 缓存键回退 balanced", _validated_weights(None, fallback="cross:balanced") == "balanced")
    check("合法 fallback 保留", _validated_weights(None, fallback="balanced") == "balanced")
    for bad in ("bogus", "cross:balanced", 123):
        try:
            _validated_weights(bad)
            check(f"未知口径被拒 {bad!r}", False)
        except HTTPException as e:
            check(f"未知口径被拒 {bad!r}", e.status_code == 400)


def test_track_default_interpolation() -> None:
    """``timeline.track_default`` must be the parameterized catalog value and render ``n``."""
    print("\ntrack_default 插值")
    from bms import i18n
    from bms.locales import zh as ZH

    i18n.set_lang("zh")
    z = i18n.tr("timeline.track_default", n=1)
    i18n.set_lang("en")
    e = i18n.tr("timeline.track_default", n=1)
    i18n.set_lang("zh")
    check("中文带 1", z == "视频轨 1", z)
    check("英文带 1", e == "Video track 1", e)
    check("没有未替换的 {n}", "{n}" not in z and "{n}" not in e, f"{z}/{e}")
    check("目录里是可插值模板", ZH.MESSAGES["timeline.track_default"] == "视频轨 {n}",
          ZH.MESSAGES["timeline.track_default"])
    check("Accept-Language 尊重 q 值", i18n.parse_lang("en;q=0.5,zh;q=0.9") == "zh",
          i18n.parse_lang("en;q=0.5,zh;q=0.9"))


def test_malformed_body_validation() -> None:
    """Malformed client values must become 400s, not 500s."""
    print("\n畸形请求体校验")
    from fastapi import HTTPException

    from bms.core.models import AnalysisParams, Rally
    from bms.main import _match_filter, _param_with_overrides

    try:
        _param_with_overrides(AnalysisParams(), {"min_rally_seconds": "abc"})
        check("非法参数值被拒", False)
    except HTTPException as e:
        check("非法参数值被拒", e.status_code == 400)
    check("未知参数被忽略",
          _param_with_overrides(AnalysisParams(), {"bogus": 1}).min_rally_seconds == 2.0)
    check("合法覆盖生效",
          _param_with_overrides(AnalysisParams(), {"min_rally_seconds": 5}).min_rally_seconds == 5.0)
    try:
        _param_with_overrides(AnalysisParams(), "not-a-dict")
        check("非字典参数被拒", False)
    except HTTPException as e:
        check("非字典参数被拒", e.status_code == 400)

    r = Rally(duration=10.0)
    try:
        _match_filter(r, {"minScore": "abc"})
        check("非法筛选值被拒", False)
    except HTTPException as e:
        check("非法筛选值被拒", e.status_code == 400)


def test_corrupt_project_json_and_atomic_save() -> None:
    """A corrupt main file must degrade to a 404 (None), and saves must be atomic + round-trip."""
    print("\n损坏工程与原子写入")
    import tempfile

    from bms.core import store as ST

    tmp = Path(tempfile.mkdtemp(prefix="bms_store_"))
    old = ST.PROJECTS_DIR
    ST.PROJECTS_DIR = tmp
    try:
        (tmp / "p_bad.json").write_text("{ not valid json", encoding="utf-8")
        check("损坏工程返回 None（路由转 404）", ST.load_project("p_bad") is None)

        p = ST.create_project("t")
        pid = p.id
        check("创建后可加载", ST.load_project(pid) is not None)
        check("无遗留 .tmp", not list(tmp.glob("*.tmp")), str(list(tmp.glob("*.tmp"))))

        def apply(proj) -> None:
            proj.name = "changed"
            proj.ui["x"] = 1

        got = ST.update_project(pid, apply, write_analyses=False)
        check("update_project 返回工程", got is not None and got.name == "changed")
        reloaded = ST.load_project(pid)
        check("改动已落盘", reloaded is not None and reloaded.name == "changed")
        check("ui 改动保留", reloaded is not None and reloaded.ui.get("x") == 1)
        check("不存在的工程返回 None", ST.update_project("p_missing", apply) is None)
    finally:
        ST.PROJECTS_DIR = old


def test_cache_clear_unknown_target() -> None:
    """An unknown cache target must be a 400 instead of silently clearing everything."""
    print("\n清缓存目标校验")
    from fastapi import HTTPException

    from bms.main import cache_clear

    try:
        cache_clear({"target": "bogus"})
        check("未知目标被拒", False)
    except HTTPException as e:
        check("未知目标被拒", e.status_code == 400)


def test_export_merge_passes_cancel() -> None:
    """Every ffmpeg call in the segmented (merge) path, including the final merge, must receive cancel."""
    print("\n导出合并传递取消")
    import tempfile

    from bms.core import ffmpeg as FF
    from bms.core.models import Clip, ExportPreset, MediaInfo, Project, Timeline, Track
    from bms.render import exporter as EX

    tmp = Path(tempfile.mkdtemp(prefix="bms_merge_"))
    src = tmp / "a.mp4"
    src.write_bytes(b"x" * 2048)
    proj = Project(name="t", media=[MediaInfo(id="m_a", path=str(src), name="a", has_audio=False,
                                              width=640, height=360, duration=5.0)])
    tl = Timeline(tracks=[Track(kind="video", clips=[
        Clip(media_id="m_a", src_in=0.0, src_out=2.0, tl_start=0.0)])], fps=30.0)
    preset = ExportPreset(id="draft", name="D")
    out = tmp / "out.mp4"

    calls: list = []
    old_caps, old_rwp = EX.ff.caps, EX.ff.run_with_progress

    def fake_caps():
        return {}

    def fake_run(cmd, total, on_progress=None, cancel=None, log_tail=8000):
        calls.append(cancel)
        Path(str(cmd[-1])).write_bytes(b"z" * 4096)  # make the segment/merge look produced
        return FF.ProcResult(0, "", "")

    EX.ff.caps = fake_caps
    EX.ff.run_with_progress = fake_run
    try:
        ok, err = EX._segmented(proj, tl, preset, out, lambda p, m="": None, lambda: False)
    finally:
        EX.ff.caps, EX.ff.run_with_progress = old_caps, old_rwp
    check("分段导出走通", ok, err)
    check("所有 ffmpeg 调用（含合并）都传了 cancel",
          bool(calls) and all(c is not None for c in calls), f"n={len(calls)}")


def test_local_origin_guard() -> None:
    """Local-only CORS / DNS-rebinding guard: local host+origin pass, foreign ones get 403.

    A browser page on any website can otherwise drive this local API; a DNS-rebinding
    domain resolves to 127.0.0.1 but carries its own Host header and must be rejected too.
    Non-browser clients send no Origin and must keep working (CLI / scripts / e2e).
    """
    print("\n本地来源守卫")
    try:
        from fastapi.testclient import TestClient
    except Exception as e:  # pragma: no cover - httpx absent
        check("TestClient 可用（跳过）", True, str(e))
        return
    from bms.main import app

    client = TestClient(app)
    check(
        "本地 Host 无 Origin 放行",
        client.get("/api/health", headers={"host": "127.0.0.1"}).status_code == 200,
    )
    check(
        "本地 Origin 放行",
        client.get(
            "/api/health", headers={"host": "127.0.0.1", "origin": "http://localhost:5273"}
        ).status_code == 200,
    )
    check(
        "file://（Origin null）放行",
        client.get("/api/health", headers={"host": "127.0.0.1", "origin": "null"}).status_code == 200,
    )
    check(
        "外部 Origin 拒绝",
        client.get(
            "/api/health", headers={"host": "127.0.0.1", "origin": "http://evil.example"}
        ).status_code == 403,
    )
    check(
        "外部 Host 拒绝（防 DNS rebinding）",
        client.get("/api/health", headers={"host": "evil.example"}).status_code == 403,
    )


def test_samples_generation() -> None:
    """Sample synthesis must raise on a broken ffmpeg path, produce two non-empty
    files, and reuse them on the second call (idempotent)."""
    print("\n示例素材合成")
    import tempfile
    from bms import samples as SM
    with tempfile.TemporaryDirectory() as td:
        # bad ffmpeg path must raise, not silently return files
        try:
            SM.ensure_sample_files(Path(td) / "bad", "definitely-not-ffmpeg", small=True)
            check("bad ffmpeg raises", False)
        except Exception:
            check("bad ffmpeg raises", True)
        files = SM.ensure_sample_files(Path(td), FFMPEG, small=True)
        check("生成两个示例", len(files) == 2)
        check("示例非空", all(f.exists() and f.stat().st_size > 0 for f in files))
        again = SM.ensure_sample_files(Path(td), FFMPEG, small=True)
        check("二次调用直接复用", [f.name for f in again] == [f.name for f in files])


def test_samples_endpoint() -> None:
    """POST /api/samples/seed generates the two demo clips and returns their
    absolute paths; a generation failure surfaces as HTTP 500 with the
    underlying error message instead of being swallowed."""
    print("\n示例素材端点")
    try:
        from fastapi.testclient import TestClient
    except Exception as e:  # pragma: no cover - httpx absent
        check("TestClient 可用（跳过）", True, str(e))
        return
    import tempfile
    import bms.main as MAIN
    from bms.main import app
    with tempfile.TemporaryDirectory() as td:
        old_data_dir = MAIN.DATA_DIR
        MAIN.DATA_DIR = Path(td)  # keep test artifacts out of the real data dir
        try:
            with TestClient(app) as c:
                # Host header required: the local-origin guard rejects TestClient's default host.
                r = c.post("/api/samples/seed", json={"small": True},
                           headers={"host": "127.0.0.1"})
                check("seed 返回 200", r.status_code == 200, r.text[:200])
                files = r.json().get("files", [])
                check("返回两个文件路径", len(files) == 2)
                check("路径为绝对路径且文件存在",
                      all(Path(f).is_absolute() and Path(f).is_file() for f in files))
                # A failure must reach the client as HTTP 500 + original message.
                old_gen = MAIN.ensure_sample_files

                def _boom(*a, **k):
                    raise RuntimeError("boom")

                MAIN.ensure_sample_files = _boom
                try:
                    r2 = c.post("/api/samples/seed", json={}, headers={"host": "127.0.0.1"})
                    check("失败返回 500", r2.status_code == 500, r2.text[:200])
                    check("500 带原始错误信息", r2.json().get("detail") == "boom", r2.text[:200])
                finally:
                    MAIN.ensure_sample_files = old_gen
        finally:
            MAIN.DATA_DIR = old_data_dir


def main() -> int:
    test_player_pipeline_contract()
    test_shuttle_pipeline_contract()
    test_bytetrack_tracker()
    test_box_size_stats()
    test_size_filter()
    test_probe_frame_capture()
    test_size_filter_selects_end_to_end()
    test_polygon_geometry()
    test_polygon_calibration()
    test_select_active_players_signature_end_to_end()
    test_manual_quad_parsing()
    test_quiet_spans_and_segmentation()
    test_detection_coverage()
    test_match_format_stats()
    test_join_abutting()
    test_activity_segmentation_has_gaps()
    test_split_by_hit_gaps()
    test_refine_trims_tail()
    test_join_abutting_keeps_shots()
    test_finish_intervals_padding_once()
    test_resegment_matches_optimizer_finishing()
    test_boundary_evidence_direction()
    test_boundary_fit_weights()
    test_boundary_refine_guardrails()
    test_boundary_pack_roundtrip()
    test_fuse_weight_defaults()
    test_fuse_components_roundtrip()
    test_shuttle_in_flight_build()
    test_fuse_shuttle_in_flight_blend()
    test_shuttle_backend_selection()
    test_shuttle_gpu_morph_equivalence()
    test_shuttle_gpu_cpu_parity()
    test_shuttle_gpu_runtime_error_falls_back()
    test_shuttle_budget_seconds_frame_unit()
    test_rebuild_fused_weight_overrides()
    test_pose_swing_peaks()
    test_pose_hit_gating_one_to_one()
    test_pose_gate_degrades()
    test_pose_keypoint_crop_inverse_map()
    test_pose_boxes_signature_quantization_stable()
    test_boxes_cache_v2_roundtrip_and_v1_compat()
    test_pose_gate_force()
    test_resegment_regates_hits()
    test_resegment_gate_degrades_without_raw()
    test_resegment_respects_use_audio()
    test_segment_rallies_reuses_player_signal()
    test_merge_by_availability_windows()
    test_annotation_evidence_and_metrics()
    test_annotation_boundary_metrics()
    test_annotation_objective_and_grid()
    test_annotation_label_quality()
    test_annotation_optimizer_runs()
    test_annotation_weights_stage()
    test_annotation_eval_window_intersection()
    test_annotation_hit_gate_stage()
    test_preset_hit_params()
    test_api_filters_and_id_safety()
    test_job_media_binding()
    test_analyze_batch_scope()
    test_preset_helpers()
    test_bulk_media_purge()
    test_cancel_jobs_for_media()
    test_cancel_suppresses_error_status()
    test_cross_media_rescore()
    test_speech_phrase_sanitize()
    test_speech_contract_and_degrade()
    test_speech_near_match()
    test_speech_refine_windows_and_merge()
    test_speech_bonus_and_scoring()
    test_exports_registry()
    test_i18n_catalogs()
    test_proxy_source_fallback()
    test_clear_derived_paths_on_cache_clear()
    test_speech_regions_use_interval_bounds()
    test_logging_setup()
    test_rally_patch_validation()
    test_invalid_weights_rejected()
    test_track_default_interpolation()
    test_malformed_body_validation()
    test_corrupt_project_json_and_atomic_save()
    test_cache_clear_unknown_target()
    test_export_merge_passes_cancel()
    test_local_origin_guard()
    test_samples_generation()
    test_samples_endpoint()
    print()
    if FAILURES:
        print(f"失败 {len(FAILURES)} 项：{FAILURES}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
