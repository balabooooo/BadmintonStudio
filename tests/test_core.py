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

import inspect
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
    test_join_abutting()
    test_activity_segmentation_has_gaps()
    test_split_by_hit_gaps()
    test_refine_trims_tail()
    test_join_abutting_keeps_shots()
    test_pose_swing_peaks()
    test_pose_hit_gating_one_to_one()
    test_pose_gate_degrades()
    test_pose_gate_force()
    test_resegment_regates_hits()
    test_resegment_gate_degrades_without_raw()
    test_resegment_respects_use_audio()
    test_segment_rallies_reuses_player_signal()
    test_merge_by_availability_windows()
    test_annotation_evidence_and_metrics()
    test_annotation_optimizer_runs()
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
    print()
    if FAILURES:
        print(f"失败 {len(FAILURES)} 项：{FAILURES}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
