"""最基础的回归测试：覆盖「静默失败」最容易发生的那几处。

用 ``python -m pytest tests`` 或直接 ``python tests/test_core.py`` 运行。
刻意不依赖任何测试框架的 pytest 专属功能，直接跑也能跑。

为什么只测这些：语音/视觉算法的准确度没法用单元测试保证，但**接口错配**
可以——而这类错误恰恰是最危险的，因为 ``analyze_players`` 的失败会被
pipeline 的 try/except 吞掉、只写进 player_trace，界面上看起来一切正常，
只是结果变差。
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


# ------------------------------------------------------------------ 接口一致性


def test_player_pipeline_contract() -> None:
    """流水线调用球员模块时用的每个关键字参数都必须在签名里存在。

    这一条是拿真实 bug 换来的：``_select_active_players`` 加了 ``boxes`` 参数
    但签名没同步，球员检测直接抛 TypeError，被 pipeline 的 try/except 吃掉，
    结果「分析成功但一个球员也没检到」，回合边界全乱。
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


# ------------------------------------------------------------------ 尺寸过滤


def test_box_size_stats() -> None:
    """尺寸统计要在「有球员」时给出合理参考，在「没球员」时优雅退化。"""
    print("\n框尺寸统计")
    boxes = [[(0, 0.10, 0.45, 0.22, 0.75)] for _ in range(20)]
    st = PL._box_size_stats(boxes, 1.7778)
    check("有球员时参考尺度在 0.05~0.5 之间", 0.05 < st["ref"] < 0.5, str(st))
    check("小目标下限为正", st["min_abs"] > 0, str(st))
    check("空输入退化", PL._box_size_stats([], 1.7778)["ref"] == 0.0)
    # 流水线里 det_frames 是**四元组**（x1,y1,x2,y2），只有 frame_boxes 才带轨迹号。
    # 只认五元组的实现会让 ref 恒为 0、自适应门限静默失效 —— 这条用例专门钉住它。
    boxes4 = [[(0.10, 0.45, 0.22, 0.75)] for _ in range(20)]
    st4 = PL._box_size_stats(boxes4, 1.7778)
    check("四元组框也能算出参考尺度", st4["ref"] > 0.05, str(st4))
    check("两种表示给出同一个参考尺度", abs(st4["ref"] - st["ref"]) < 1e-9, f"{st4} vs {st}")
    check("框表示归一化：四元组", PL._box_xyxy((0.1, 0.2, 0.3, 0.4)) == (0.1, 0.2, 0.3, 0.4))
    check("框表示归一化：五元组（带轨迹号）",
          PL._box_xyxy((7, 0.1, 0.2, 0.3, 0.4)) == (0.1, 0.2, 0.3, 0.4))
    check("坏输入返回 None", PL._box_xyxy((0.1, 0.2)) is None)


# ------------------------------------------------------------------ 人物框尺寸筛选


def test_size_filter() -> None:
    """尺寸筛选的两种口径 + 面积上下限 + 统计直方图。

    这一块是「用户自己指定阈值」的入口，判错的后果是**静默地**把真球员筛掉
    （分析照样跑完、只是回合全乱），所以每个口径都要有用例钉住。
    """
    print("\n人物框尺寸筛选")
    small = (0.40, 0.30, 0.45, 0.34)      # 框高 0.04（观众 / 远处的人）
    big = (0.40, 0.40, 0.60, 0.62)        # 框高 0.22（球员）

    off = PL.size_filter_from({"mode": "off"})
    check("off 时不做筛选", off.keep(small, 0.22) and off.keep(big, 0.22) and not off.active)

    absf = PL.size_filter_from({"mode": "absolute", "min_height": 0.10})
    check("absolute：低于绝对下限的被筛掉", not absf.keep(small, 0.22))
    check("absolute：够大的保留", absf.keep(big, 0.22))

    absu = PL.size_filter_from({"mode": "absolute", "min_height": 0.0, "max_height": 0.15})
    check("absolute：超过上限的被筛掉", not absu.keep(big, 0.22))
    check("absolute：上限内保留", absu.keep(small, 0.22))

    rel = PL.size_filter_from({"mode": "relative", "min_height": 0.5})
    # 同帧最大框高 0.22 -> 实际门限 0.11
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

    # 统计：直方图必须覆盖**筛选前**的所有框（否则界面看不到被砍掉的部分）
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
    """试测取帧：指定时刻的取帧要能跳过 seek 后的黑帧，帧图路径要稳定且唯一。

    这一条守的是「界面里抓一帧看看框选对没有」这个功能：如果取到的是黑帧，
    用户看到的就是一片黑上画着框，完全没法核对（而且会以为是自己标错了）。
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
    # 合成一段视频：前 5 帧故意全黑（模拟 seek 之后返回的黑帧），之后是亮帧
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
        fr = PL._read_frame_at(cap, 10.0, 0.0)          # 第 0 帧是黑帧
        check("黑帧被跳过（拿到的是非纯色帧）",
              fr is not None and float(fr.std()) > 2.0,
              f"std={0.0 if fr is None else float(fr.std()):.2f}")
        fr2 = PL._read_frame_at(cap, 10.0, 1.5)         # 第 15 帧正常
        check("正常时刻能取到帧", fr2 is not None and float(fr2.std()) > 2.0)
        fr3 = PL._read_frame_at(cap, 10.0, 999.0)       # 超出片长
        check("超出片长不抛异常", fr3 is None or float(fr3.std()) >= 0.0)
    cap.release()


def test_size_filter_selects_end_to_end() -> None:
    """真的带 size_filter 调一次 _select_active_players：阈值必须真的起作用。

    接口错配 / 阈值没接进候选筛选，在 import 阶段是查不出来的，
    而后果是「用户拖了阈值但结果一点没变」。
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

    # 反过来卡上限：阈值真的被用上了 —— 而且它能覆盖「自适应尺寸门限」
    # （自适应那套只会砍小轨迹，永远砍不掉大轨迹）
    upper = PL.size_filter_from({"mode": "absolute", "min_height": 0.0, "max_height": 0.12})
    ids2 = [t.track_id for t in PL._select_active_players(
        tracks, 12.0, 200.0, viewpoint="rear", boxes=boxes, aspect=1.7778, size_filter=upper)]
    check("尺寸上限能把大轨迹筛掉（阈值真的生效）", 1 not in ids2, str(ids2))


# ------------------------------------------------------------------ 场地多边形


def test_polygon_geometry() -> None:
    """多边形几何：环序、面积、拟合四边形、场内判定。"""
    print("\n场地多边形几何")
    # 一个「近边向下凸、远边向上凸」的六边形：全景 / 鱼眼素材的典型形状
    hexa = np.array([[0.10, 0.96], [0.50, 1.00], [0.90, 0.96],
                     [0.80, 0.40], [0.50, 0.30], [0.20, 0.40]], dtype=np.float32)
    ring = CC.order_poly(hexa, aspect=1.6)
    check("环序点数不变", ring.shape == hexa.shape)
    check("环序起点是最靠近画面的点（y 最大且最靠左）",
          abs(ring[0][1] - 0.96) < 1e-6 and abs(ring[0][0] - 0.10) < 1e-6, str(ring[0]))
    same = {tuple(np.round(p, 5)) for p in ring} == {tuple(np.round(p, 5)) for p in hexa}
    check("环序不增删点", same)

    quad = CC.order_poly(np.array([[0.9, 0.98], [0.5, 0.3], [0.1, 0.98], [0.5, 0.4]],
                                  dtype=np.float32), aspect=1.6)  # 打乱的四边形
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
    # 外扩：紧贴边线外侧的点在 margin 下也算场内（球员站在边线上不能被漏掉）
    edge = CC.point_in_poly(np.array([[0.5, 1.004]], dtype=np.float32), square)
    check("精确判定：边线外侧算场外", edge.tolist() == [False], str(edge.tolist()))
    edge_m = CC.point_in_poly(np.array([[0.5, 1.004]], dtype=np.float32), square, margin=0.02)
    check("带外扩后边线外侧算场内", edge_m.tolist() == [True], str(edge_m.tolist()))
    # 弯边六边形：外接矩形里的两个角必须被判成「场外」——这正是多边形表示的价值
    corner = CC.point_in_poly(np.array([[0.02, 0.02], [0.98, 0.02], [0.5, 0.65]],
                                       dtype=np.float32), hexa)
    check("弯边以外（矩形内）的点被判成场外", corner[0] == False and corner[1] == False,  # noqa: E712
          str(corner.tolist()))


def test_polygon_calibration() -> None:
    """多边形标定：单应、畸变度量、ROI、payload 字段。"""
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

    # 退化的输入必须被挡住（否则 ROI 会缩成一个点，把所有人筛掉）
    tiny = np.array([[0.5, 0.5], [0.501, 0.5], [0.501, 0.501], [0.5, 0.501]], dtype=np.float32)
    bad = CC.build_calibration(tiny * np.array([fw, fh], dtype=np.float32), fw, fh)
    check("面积过小的多边形被拒", not bad.ok, str(bad.notes))
    tri = CC.build_calibration(np.array([[100, 900], [900, 900], [500, 200]], dtype=np.float32),
                               fw, fh)
    check("少于四点的多边形被拒", not tri.ok)

    # Auto 路径：弯边 mask 应该拟合出多于 4 个点（而不是被硬压成四边形）
    try:
        import cv2

        mask = np.zeros((540, 960), dtype=np.uint8)
        pts: list[list[float]] = []
        for i in range(41):                      # 近边：向下鼓
            t = i / 40.0
            pts.append([60 + t * 840, 480 + 40 * np.sin(np.pi * t)])
        for i in range(41):                      # 右边：向右鼓
            t = i / 40.0
            pts.append([900 + 40 * np.sin(np.pi * t), 480 - t * 380])
        for i in range(41):                      # 远边：向上鼓
            t = 1.0 - i / 40.0
            pts.append([900 - t * 840, 100 - 40 * np.sin(np.pi * t)])
        for i in range(41):                      # 左边：向左鼓
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

        # 反过来：直边的场地必须仍然只给 4 个点（否则下游拿到的标定会平白变复杂，
        # 「自动加点」就成了给所有素材都加噪声）
        straight = np.zeros((540, 960), dtype=np.uint8)
        cv2.fillPoly(straight, [np.array([[120, 520], [840, 520], [700, 120], [260, 120]],
                                         dtype=np.int32)], 255)
        poly2 = CC.find_court_poly(straight, min_area_ratio=0.02)
        check("直边场地仍然拟合成四边形", poly2 is not None and poly2.shape[0] == 4,
              f"点数 {0 if poly2 is None else poly2.shape[0]}")
    except ImportError:  # pragma: no cover - 没装 opencv 的环境
        check("opencv 可用（跳过弯边拟合用例）", False)


def test_select_active_players_signature_end_to_end() -> None:
    """真的调一次 _select_active_players：接口错配在 import 阶段是查不出来的。"""
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


# ------------------------------------------------------------------ 手动标定


def test_manual_quad_parsing() -> None:
    print("\n手动标定多边形")
    good = [[0.1, 0.99], [0.9, 0.99], [0.8, 0.2], [0.2, 0.2]]
    check("合法四边形可用", P._manual_quad(good) is not None)
    check("点数不对被拒", P._manual_quad(good[:3]) is None)
    check("像素坐标被拒", P._manual_quad([[100, 900], [900, 900], [800, 200], [200, 200]]) is None)
    check("四点重合被拒", P._manual_quad([[0.5, 0.5]] * 4) is None)
    check("None 被拒", P._manual_quad(None) is None)
    # 多点（全景 / 鱼眼）路径
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


# ------------------------------------------------------------------ 切分


def test_quiet_spans_and_segmentation() -> None:
    """合成一段「对拉 → 停顿 → 对拉」，切分必须切出两段且中间留间隔。"""
    print("\n球员运动切分（合成信号）")
    fps = 12.0
    dur = 120.0
    n = int(dur * fps)
    m = np.zeros(n, dtype=np.float32)
    for a, b in ((10, 40), (60, 95)):
        m[int(a * fps):int(b * fps)] = 1.0
    m += np.random.RandomState(0).normal(0, 0.02, n).astype(np.float32)
    spans = RV.find_quiet_spans(m, fps, min_quiet=1.0, prominence_ratio=0.3)
    # 这段合成信号在 40~60s 之间有一次停顿；开头和结尾都是连续运动，
    # 所以「静默段」恰好 1 个（首尾的回合由 segment_by_player_motion 补齐）。
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
        # 第一个回合应该在 10s 附近开始、40s 附近结束（含留白）
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
    """活跃度谷值切分不应该产出首尾相接的区间。"""
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
    """击球序列里的大空档必须把区间切开，小空档不能切。

    这一条对应实测的 bug：30 分钟素材切出一个 69.35 秒、72 拍的「回合」
    （0.0~69.35s），里面其实有两三个回合。击球间隔里有一个 6.93 秒的空档，
    切分却完全没用上这个信息。
    """
    print("\n击球空档切分")
    fps = 12.0
    dur = 80.0
    # 0~29s 密集击球（每 0.6s 一拍），29~35s 空档，35~50s 又密集
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

    # 只有小空档（都在 MAX_INTRA_HIT_GAP 以内）时不允许切
    small = np.asarray([0.5, 1.2, 1.9, 4.5, 5.2, 5.9, 6.6], dtype=np.float64)
    h2 = RA.HitDetection(
        times=small,
        strength=np.full(small.size, 0.6, dtype=np.float32),
        confidence=np.full(small.size, 0.8, dtype=np.float32),
        envelope=np.zeros(10, dtype=np.float32), env_fps=250.0,
    )
    out2 = RA.split_by_hit_gaps([RA.RallyInterval(start=0.0, end=7.0)], h2)
    check("小空档不切", len(out2) == 1, f"{len(out2)}")

    # 空档很大但两侧只有 1 拍：是漏检，不是两个回合
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
    """终点必须能被收紧到「最后一拍 + 尾巴」，而不是只能往后推。

    旧实现写的是 ``iv.end = max(iv.end, last + tail)``，于是无论切分给出的
    区间有多长都收不回来 —— 这正是「球落地之后还留很长一段」的直接原因。
    """
    print("\n终点锚定到最后一拍")
    times = np.asarray([10.0, 10.8, 11.6, 12.4, 13.2], dtype=np.float64)
    hits = RA.HitDetection(
        times=times,
        strength=np.full(times.size, 0.7, dtype=np.float32),
        confidence=np.full(times.size, 0.9, dtype=np.float32),
        envelope=np.zeros(10, dtype=np.float32), env_fps=250.0,
    )
    # 切分给出的终点远在最后一拍之后（13.2 -> 40.0），必须收到 13.2 + 0.9
    ivs = [RA.RallyInterval(start=9.0, end=40.0)]
    out = RA.refine_with_hits(ivs, hits, pre_roll=1.2, post_roll=0.5, tail_seconds=0.9)
    check("终点被收紧到最后一拍 + 尾巴",
          abs(out[0].end - (13.2 + 0.9)) < 0.01, f"{out[0].end:.2f}")
    check("起点对齐到第一次击球前 pre_roll",
          abs(out[0].start - (10.0 - 1.2)) < 0.01, f"{out[0].start:.2f}")

    # 反向：最后一拍之后紧接着（< gap_limit）还有击球，说明球还在飞，不能收
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

    # 一次击球只能属于一个回合：切点两侧不能把同一拍都算进来。
    # （起点允许往前伸 pre_roll 去吃发球准备，所以两段的**边界**本来就会重叠，
    #   真正的约束是「拍不重复」，重叠交给 dedupe_overlaps 去切。）
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
    """合并相邻区间时必须保留击球索引（旧代码清空，导致合并后显示「0 拍」）。"""
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
    """造一路合成的姿态信号：在给定时刻各放一个挥拍峰。"""
    from bms.analysis import pose as POSE

    n = int(dur * fps)
    sw = np.full(n, 0.4, dtype=np.float32)
    over = np.zeros(n, dtype=np.float32)
    ok = np.ones(n, dtype=np.float32)
    for t in peaks:
        i = int(t * fps)
        # 一个挥拍大约 0.25 秒：三帧内冲上去再回来
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
    """挥拍峰的检测要落在合成峰的位置上。"""
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
    """一个挥拍峰只能解释一个击球；没有挥拍的击球要被判为「别场地的」。"""
    print("\n姿态：击球归属门控")
    from bms.analysis import pose as POSE

    pose = _synthetic_pose(peaks=(3.0, 8.0, 15.0))
    # 3.0 和 3.08 两拍都贴着一个峰 → 只留更近的那个；5.0 附近没有任何挥拍
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
    """姿态不可信时必须原样放行，而不是把击球砍掉一半。"""
    print("\n姿态：不可用时降级")
    from bms.analysis import pose as POSE

    hits = _hits([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # 覆盖率太低
    low = _synthetic_pose(coverage=0.2)
    mask, trace = POSE.gate_hits(hits, low)
    check("覆盖率低时门控不生效", mask is None, str(trace))
    # 完全没有峰（信号全平）
    flat = _synthetic_pose(peaks=())
    mask2, trace2 = POSE.gate_hits(hits, flat)
    check("没有挥拍峰时候降级（保留比例过低）", mask2 is None, str(trace2))
    # pose 为 None
    mask3, _ = POSE.gate_hits(hits, None)
    check("pose 为 None 时返回 None", mask3 is None)
    check("filter_hits 收到 None 掩码时原样返回", POSE.filter_hits(hits, None) is hits)


def test_segment_rallies_reuses_player_signal() -> None:
    """给了存下来的球员运动曲线时，_segment_rallies 必须走球员切分。

    这一条是拿真实 bug 换来的：快速重切分（resegment）没有球员框，
    旧版本直接退化成「只看活跃度」，于是同一个工程「重切分」和「重跑分析」
    给出不一样的回合数 —— 而重切分正是调切分参数时最常用的操作。
    """
    print("\n重切分复用球员信号")
    fps = 12.0
    dur = 120.0
    n = int(dur * fps)
    act = np.full(n, 0.5, dtype=np.float32)          # 活跃度故意做成平的
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
    """「按时间段择优」的三种情况必须都正确。

    这一条是拿实测数据换来的。旧实现写的是「球员切分有候选就用它、没有就退回
    活跃度切分」，把用来做判断的 ``valid_ratio`` 算出来却没用到。后果是：
    球员切分**故意留空**的地方（那正是两个回合之间真实的停顿）会被活跃度候选
    补上，回合又被粘长 —— 实测最长回合比修好之后多出 6 秒。
    但反过来「没候选就一律留空」又会把球员切分的漏检当成停顿，实测召回率掉 6 个百分点。
    所以第三种情况（球员在动 → 兜住）是必需的。
    """
    print("\n按时间段择优（三种情况）")
    fps = 12.0
    dur = 120.0
    n = int(dur * fps)
    cov = np.ones(n, dtype=np.float32)
    pm = np.zeros(n, dtype=np.float32)                      # 默认：球员不动

    # 10~30s：球员在动，但球员切分没给候选（模拟漏检）→ 应该用活跃度兜住
    # 50~70s：球员不动，球员切分也没候选（真实停顿）→ 应该留空
    # 90~110s：球员切分给了候选 → 应该直接用它
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
    """标注评估的纯函数 + 击球密度证据。

    击球密度是这条素材上唯一有区分度的一路（实测量测 AUC≈0.78），
    它一旦被算错，整个「用标注校准参数」就失去意义。
    """
    print("\n标注评估与击球证据")
    fps = 12.0
    n = int(60 * fps)
    # 10~15s 连打 6 拍，其余时间没有击球
    hd = RA.hit_density_signal(_hits([10, 10.5, 11, 12, 13, 14]), fps, n)
    check("击球密度长度正确", hd.size == n)
    check("连打的时段密度高", float(hd[int(10 * fps):int(15 * fps)].mean()) > 0.4,
          str(float(hd[int(10 * fps):int(15 * fps)].mean())))
    check("没击球的时段密度低", float(hd[int(40 * fps):int(55 * fps)].mean()) < 0.1,
          str(float(hd[int(40 * fps):int(55 * fps)].mean())))

    # 球员不动时证据为 0；球员在动 + 在击球时证据高
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
    """「用标注搜参」必须能跑通并返回可用的最优参数（离线、不重跑 AI）。"""
    print("\n标注搜参")
    fps = 12.0
    dur = 120.0
    n = int(dur * fps)
    act = np.full(n, 0.4, dtype=np.float32)
    pm = np.zeros(n, dtype=np.float32)
    hits = np.zeros(n, dtype=np.float32)
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


def main() -> int:
    test_player_pipeline_contract()
    test_shuttle_pipeline_contract()
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
    test_segment_rallies_reuses_player_signal()
    test_merge_by_availability_windows()
    test_annotation_evidence_and_metrics()
    test_annotation_optimizer_runs()
    print()
    if FAILURES:
        print(f"失败 {len(FAILURES)} 项：{FAILURES}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
