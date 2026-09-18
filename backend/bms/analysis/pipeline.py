"""分析流水线：把各模块串起来，产出可编辑的回合列表。

流程
----
1. 准备派生资源（代理视频 / 音轨）
2. 音频击球检测       -> 击球时刻
3. 画面运动分析       -> 运动能量 / 抖动 / 清晰度 / 场地热区
4. 球员检测与跟踪      -> 场上活跃度（可选，缺失自动降级）
5. 羽毛球轨迹跟踪      -> 球的出现与速度（可选，缺失自动降级）
6. 多模态融合 + 状态机 -> 回合区间
7. 音频回贴边界        -> 发球 / 接发球时刻
8. 特征提取 + 评分     -> 回合分数与标签
"""

from __future__ import annotations

import traceback
from typing import Any, Callable

import numpy as np

from ..config import ensure_dirs
from ..core.models import (
    AnalysisParams,
    AnalysisResult,
    MediaInfo,
    Rally,
    RallyFeatures,
    RallyScores,
    ShotEvent,
)
from ..core import media as M
from . import audio_hits as AH
from . import court_calib as CC
from . import motion as MO
from . import rally as RA
from . import rally_vision as RV
from . import scoring as SC

Progress = Callable[[float, str, str], None]  # (progress, stage, message)


def _noop(p: float, s: str, m: str = "") -> None:
    pass


def _try_import(name: str):
    try:
        mod = __import__(f"bms.analysis.{name}", fromlist=[name])
        return mod
    except Exception:
        return None


def attribute_sides(rallies: list[Rally], player_sig, fps: float,
                    viewpoint: str = "unknown") -> None:
    """把「发球 / 接发球」落到具体球员身上。

    做法：比赛球员里**框更大的一定更靠近相机**，据此把两名球员分成 near / far；
    再看发球时刻前后谁在动——刚挥拍那个人就是发球方。这个判断只用到了
    检测框，不依赖姿态模型，所以在球员很小的时候也基本能用。

    **「框大 = 近」只在后方/侧方低机位成立。** 高机位与俯拍下两个人离相机的
    距离差不多，框面积也就差不多，继续按面积区分只会把 near/far 分配到
    噪声上。这两种机位下本函数直接不写（保持 ``unknown``）。

    拿不准时保持 ``unknown``，宁可不写也不要写错。
    """
    if viewpoint in ("overhead", "elevated"):
        return
    if player_sig is None:
        return
    tracks = getattr(player_sig, "tracks", None) or []
    ids = list(getattr(player_sig, "active_player_ids", None) or [])
    boxes = getattr(player_sig, "frame_boxes", None) or []
    pfps = float(getattr(player_sig, "fps", fps) or fps)
    if not tracks or not ids or not boxes or pfps <= 0:
        return

    by_id = {t.track_id: t for t in tracks}
    actives = [by_id[i] for i in ids if i in by_id]
    if len(actives) < 2:
        return

    # 画面里框越大 = 离相机越近
    biggest = max(actives, key=lambda t: t.mean_area)
    others = [t for t in actives if t.track_id != biggest.track_id]
    if not others:
        return
    second = max(others, key=lambda t: t.mean_area)
    if biggest.mean_area <= second.mean_area * 1.05:
        return  # 两人离相机差不多远，分不出前后，放弃
    near_id, far_id = biggest.track_id, second.track_id

    def center_at(track_id: int, t: float) -> tuple[float, float] | None:
        idx = int(round(t * pfps))
        if idx < 0 or idx >= len(boxes):
            return None
        for item in boxes[idx]:
            if item[0] == track_id:
                _, x1, y1, x2, y2 = item
                return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        return None

    def movement(track_id: int, t0: float, t1: float) -> float:
        """时间段内该球员的累计位移（按画面高度归一）。"""
        a = max(0, int(round(t0 * pfps)))
        b = min(len(boxes), max(a + 1, int(round(t1 * pfps))))
        total = 0.0
        prev: tuple[float, float] | None = None
        for i in range(a, b):
            cur = None
            for item in boxes[i]:
                if item[0] == track_id:
                    _, x1, y1, x2, y2 = item
                    cur = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    break
            if cur is not None and prev is not None:
                total += float(np.hypot(cur[0] - prev[0], cur[1] - prev[1]))
            if cur is not None:
                prev = cur
        return total

    for r in rallies:
        t = r.serve_time
        if t is None:
            continue
        # 发球动作发生在发球时刻之前约 0.7 秒
        mv_near = movement(near_id, t - 0.7, t + 0.25)
        mv_far = movement(far_id, t - 0.7, t + 0.25)
        if max(mv_near, mv_far) < 0.006:
            continue  # 两个人都没动，判断不出来
        server_is_near = mv_near >= mv_far
        r.serve_player = "near" if server_is_near else "far"  # type: ignore[assignment]
        r.receive_player = "far" if server_is_near else "near"  # type: ignore[assignment]
        # 逐拍按「发球方交替」重排，比固定 near 起始更贴近真实
        first_side = r.serve_player
        for k, s in enumerate(r.shots):
            s.player = first_side if k % 2 == 0 else ("far" if first_side == "near" else "near")  # type: ignore[assignment]


def run_analysis(
    media: MediaInfo,
    params: AnalysisParams | None = None,
    on_progress: Progress = _noop,
    cancel: Callable[[], bool] | None = None,
    weights_key: str = "balanced",
    roi: tuple[float, float, float, float] | None = None,
) -> AnalysisResult:
    ensure_dirs()
    params = params or AnalysisParams()
    res = AnalysisResult(media_id=media.id, params=params, status="running")
    res.started_at = int(__import__("time").time() * 1000)

    def emit(p: float, stage: str, msg: str = "") -> None:
        res.progress = float(max(0.0, min(1.0, p)))
        res.stage = stage
        res.message = msg
        on_progress(res.progress, stage, msg)

    def cancelled() -> bool:
        return bool(cancel and cancel())

    try:
        # ---------------------------------------------------------- 1. 派生资源
        emit(0.01, "prepare", "检查派生资源")
        M.ensure_proxy(media, on=lambda p, m: emit(0.01 + 0.14 * p, "proxy", m), cancel=cancel)
        M.ensure_audio(media, on=lambda p, m: emit(0.15 + 0.05 * p, "audio", m), cancel=cancel)
        M.ensure_poster(media)
        if cancelled():
            res.status = "cancelled"
            return res

        duration = media.duration
        fw = int(media.proxy_width or media.width or 0)
        fh = int(media.proxy_height or media.height or 0)
        if fw <= 0 or fh <= 0:
            fw, fh = 1920, 1080

        # ---------------------------------------------------------- 1.5 场地标定
        # 先标定再检测球员：有了「场地在画面的哪一块」就能把球员检测限制在
        # 这块场地里，从而不再依赖「框最大的两个人就是球员」这种跟机位强耦合的
        # 猜测；同时机位类型会决定后续用哪套先验参数。
        calib: CC.CourtCalibration | None = None
        hint = params.viewpoint if params.viewpoint != "auto" else None
        # 手动标定的场地边界：优先用新的多点多边形字段，旧工程的 court_quad
        # （只有四个角）继续生效 —— 它是 court_poly 只给四个点的特例。
        manual = _manual_poly(params.court_poly)
        manual_source = "court_poly"
        if manual is None and params.court_quad:
            manual = _manual_poly(params.court_quad)
            manual_source = "court_quad"
        if manual is not None:
            # 用户手动标过场地：直接用，不再猜测。
            # 手动标定比任何自动方法都可靠，而且标一次就能一直复用。
            calib = CC.build_calibration(manual * np.array([fw, fh], dtype=np.float32),
                                         fw, fh, source="manual")
            n_pts = int(manual.shape[0])
            calib.note(f"使用手动标定的场地边界（{n_pts} 个点，来自 params.{manual_source}）")
            if not calib.ok:
                calib.note("手动标定退化（点太少、面积为零或共线），回退到自动标定")
                manual = None
        if manual is None and params.auto_calibrate:
            emit(0.205, "calibrate", "标定场地与机位")
            try:
                calib = CC.calibrate(
                    str(media.proxy_path or media.path),
                    frame_size=(fw, fh),
                    hint=hint,      # type: ignore[arg-type]
                )
            except Exception as e:      # 标定失败不能中断分析
                calib = CC.CourtCalibration()
                calib.note(f"标定异常：{type(e).__name__}: {e}")
        elif manual is None:
            calib = CC.CourtCalibration()
            calib.note("已关闭自动标定，且没有手动标定（使用全画幅与通用参数）")
        if cancelled():
            res.status = "cancelled"
            return res

        # 用户显式给了 ROI 就用它，否则用标定出来的场地范围。
        # 注意这里是**外接矩形**（只用来粗筛和解码范围），真正的场内判定
        # 由下面的 eff_poly 交给球员模块按多边形做。
        eff_roi = roi
        eff_poly: list[list[float]] | None = None
        if calib is not None and calib.ok:
            if eff_roi is None:
                eff_roi = calib.court_mask_roi(margin=0.04)
            eff_poly = calib.poly_norm()

        # ---------------------------------------------------------- 2. 音频击球
        hits: AH.HitDetection | None = None
        hit_trace: dict[str, Any] = {}
        if params.use_audio and media.audio_path:
            emit(0.22, "hits", "检测击球声")
            hits = AH.detect_hits(media.audio_path, sensitivity=params.hit_sensitivity)
            hit_trace = {
                "count": int(hits.times.size),
                "env_fps": float(hits.env_fps),
                "noise_floor_db": round(float(hits.noise_floor_db), 1),
            }
        if cancelled():
            res.status = "cancelled"
            return res

        # ---------------------------------------------------------- 3. 画面运动
        motion_sig: MO.MotionSignal | None = None
        motion_dict: dict[str, np.ndarray] | None = None
        if params.use_motion:
            emit(0.26, "motion", "分析画面运动")
            motion_sig = MO.analyze_motion(
                media,
                sample_fps=min(params.sample_fps, 15.0),
                work_width=320,
                on=lambda p, m: emit(0.26 + 0.14 * p, "motion", m),
                cancel=cancel,
            )
            motion_dict = {"fps": np.array([motion_sig.fps], dtype=np.float32), "motion": motion_sig.motion,
                           "court_motion": motion_sig.court_motion, "sharpness": motion_sig.sharpness,
                           "shake": motion_sig.shake}
            motion_dict["fps"] = float(motion_sig.fps)
        if cancelled():
            res.status = "cancelled"
            return res

        # ---------------------------------------------------------- 4. 球员
        player_sig = None
        players_dict: dict[str, np.ndarray] | None = None
        player_trace: dict[str, Any] = {}
        # 人物框尺寸筛选：用户在参数里指定的阈值（off 时等价于旧行为）。
        # 它和「场地 ROI」是两道互补的闸门：ROI 管「在哪找」，尺寸管「找多大的」。
        size_filter = {
            "mode": params.player_size_mode,
            "min_height": params.player_min_height,
            "max_height": params.player_max_height,
            "min_area": params.player_min_area,
            "max_area": params.player_max_area,
        }
        if params.use_players:
            mod = _try_import("players")
            if mod is not None and hasattr(mod, "analyze_players"):
                try:
                    emit(0.40, "players", "检测与跟踪球员")
                    player_sig = mod.analyze_players(
                        str(media.proxy_path or media.path),
                        sample_fps=min(params.sample_fps, 12.0),
                        roi=eff_roi,
                        roi_poly=eff_poly,
                        size_filter=size_filter,
                        max_seconds=0.0,
                        viewpoint=(calib.viewpoint if calib is not None else "unknown"),
                        on_progress=lambda p, s: emit(0.40 + 0.24 * p, "players", s),
                        cancel=cancel,
                    )
                    players_dict = {
                        "fps": float(getattr(player_sig, "fps", params.sample_fps)),
                        "active_count": getattr(player_sig, "active_count", np.zeros(0)),
                        "active_speed": getattr(player_sig, "active_speed", np.zeros(0)),
                        "max_speed": getattr(player_sig, "max_speed", np.zeros(0)),
                    }
                    player_trace = {
                        "tracks": len(getattr(player_sig, "tracks", []) or []),
                        "active_ids": list(getattr(player_sig, "active_player_ids", []) or []),
                        # 尺寸筛选的实测分布：界面用它画出直方图并实时预览阈值效果
                        "size_filter": dict(getattr(player_sig, "size_stats", {}) or {}),
                        "roi_poly_points": len(eff_poly or []),
                    }
                except Exception as e:  # 球员模块失败不应中断整体分析
                    player_trace = {"error": f"{type(e).__name__}: {e}"}
        if cancelled():
            res.status = "cancelled"
            return res

        # ---------------------------------------------------------- 4.5 姿态
        # 姿态**不改切分用哪一路信号**，只回答一个问题：这个音频击球是不是
        # 我们这场比赛打的。多球场球馆里这件事音频自己答不了
        # （实测 audio_reliability = 0.04），而没有它就只能靠「击球间隔」
        # 去猜回合边界 —— 隔壁场地的声音会把间隔填满，于是回合被粘长。
        # 失败一律降级：pose_sig 为 None 时下游行为与没开这个开关时一致。
        pose_sig = None
        pose_trace: dict[str, Any] = {}
        gate_mask: np.ndarray | None = None
        if params.use_pose and hits is not None and player_sig is not None:
            boxes = getattr(player_sig, "frame_boxes", None)
            if boxes:
                mod = _try_import("pose")
                if mod is not None and hasattr(mod, "analyze_pose"):
                    try:
                        emit(0.64, "pose", "分析球员姿态以判定击球归属")
                        pose_sig = mod.analyze_pose(
                            str(media.proxy_path or media.path),
                            boxes,
                            float(getattr(player_sig, "fps", 12.0) or 12.0),
                            on_progress=lambda p, s: emit(0.64 + 0.01 * p, "pose", s),
                            cancel=cancel,
                        )
                        if pose_sig is not None:
                            pose_trace = dict(pose_sig.trace)
                            gate_mask, g_trace = mod.gate_hits(hits, pose_sig)
                            pose_trace.update(g_trace)
                            if gate_mask is not None:
                                hits = mod.filter_hits(hits, gate_mask)
                    except Exception as e:  # 姿态失败只能降级，绝不能中断整体分析
                        pose_trace = {"error": f"{type(e).__name__}: {e}"}
                else:
                    pose_trace = {"disabled": "姿态模块不可用（缺少 ultralytics？）"}
            else:
                pose_trace = {"skipped": "没有球员框，姿态无从下手"}
        elif not params.use_pose:
            pose_trace = {"disabled": "params.use_pose = false"}
        else:
            pose_trace = {"skipped": "没有音频击球或球员信号，姿态没有用武之地"}
        if cancelled():
            res.status = "cancelled"
            return res

        # ---------------------------------------------------------- 5. 羽毛球
        shuttle_sig = None
        shuttle_dict: dict[str, np.ndarray] | None = None
        shuttle_trace: dict[str, Any] = {}
        if params.use_shuttle:
            mod = _try_import("shuttle")
            if mod is not None and hasattr(mod, "analyze_shuttle"):
                try:
                    budget = float(params.shuttle_budget_seconds or 0.0)
                    limit = min(duration, budget) if budget > 0 else 0.0
                    coverage = (limit / duration) if (limit > 0 and duration > 0) else 1.0
                    emit(0.66, "shuttle", "跟踪羽毛球轨迹")
                    shuttle_sig = mod.analyze_shuttle(
                        str(media.proxy_path or media.path),
                        sample_fps=min(params.shuttle_fps, params.sample_fps),
                        roi=eff_roi,
                        max_seconds=limit,
                        on_progress=lambda p, s: emit(0.66 + 0.18 * p * max(coverage, 0.05), "shuttle", s),
                        cancel=cancel,
                    )
                    shuttle_trace = {"tracks": len(getattr(shuttle_sig, "tracks", []) or [])}
                    if coverage < 0.6:
                        # 只覆盖了一部分时间轴，保留会严重带偏融合权重，直接丢弃
                        shuttle_trace["skipped"] = (
                            f"仅覆盖 {coverage * 100:.0f}% 的时间轴（预算 {budget:.0f}s），"
                            "为避免融合被带偏已忽略该信号；如需全片跟踪请把时间预算设为 0"
                        )
                        shuttle_dict = None
                    else:
                        shuttle_dict = {
                            "fps": float(getattr(shuttle_sig, "fps", params.sample_fps)),
                            "presence": getattr(shuttle_sig, "presence", np.zeros(0)),
                            "max_candidate_speed": getattr(shuttle_sig, "max_candidate_speed", np.zeros(0)),
                        }
                        shuttle_trace["coverage"] = round(coverage, 3)
                except Exception as e:
                    shuttle_trace = {"error": f"{type(e).__name__}: {e}"}
        else:
            shuttle_trace = {"disabled": "羽毛球跟踪默认关闭（长视频耗时很长）"}
        if cancelled():
            res.status = "cancelled"
            return res

        # ---------------------------------------------------------- 6. 融合 + 切分
        emit(0.85, "segment", "融合信号并切分回合")
        ana_fps = float(min(params.sample_fps, 15.0)) or 15.0
        fused = RA.fuse(
            fps=ana_fps,
            duration=duration,
            hits=hits,
            motion=motion_dict,
            players=players_dict,
            shuttle=shuttle_dict,
            roi_activity=motion_sig.motion if motion_sig is not None else None,
        )

        # 球员检测跑完了，用真实球员框再判一次机位（这次有数据支撑）
        if calib is not None and player_sig is not None:
            try:
                CC.refine_viewpoint(
                    calib,
                    getattr(player_sig, "frame_boxes", None),
                    float(getattr(player_sig, "fps", 0.0) or 0.0),
                    (fw, fh),
                    hint=hint,      # type: ignore[arg-type]
                )
            except Exception as e:
                calib.note(f"机位二次判定失败：{type(e).__name__}: {e}")

        intervals, seg_trace = _segment_rallies(
            fused=fused,
            params=params,
            duration=duration,
            player_sig=player_sig,
            shuttle_sig=shuttle_sig,
            hits=hits,
        )

        # ---------------------------------------------------------- 7. 边界锚定
        # 不管用哪种切分方式，都用**击球序列**把边界钉一次：
        #   1. 在击球序列的大空档处切开 —— 消掉「一个回合里混进下一个回合」；
        #   2. 终点锚到最后一拍 + 球落地所需的飞行时间 —— 消掉「球落地之后
        #      还留很长一段」。旧代码这里写的是 max()，终点只能往后推、
        #      永远收不回来，所以哪怕切出 69 秒、72 拍的区间也修不掉。
        # 球员运动切分给出的**起点**是「球员开始动」，本身已含发球准备，
        # 所以对它只收终点、不推迟起点。
        player_path = seg_trace.get("method") == "player_motion"
        if hits is not None:
            intervals = RA.refine_with_hits(
                intervals, hits,
                pre_roll=params.pre_roll,
                post_roll=params.post_roll,
                tail_seconds=params.hit_tail_seconds,
                trim_start=not player_path,
            )
        else:
            for iv in intervals:
                iv.start = max(0.0, iv.start - params.pre_roll)
                iv.end = min(duration, iv.end + params.post_roll)

        if not intervals and hits is not None and hits.times.size:
            # 视觉完全没信号时退回纯音频
            intervals = RA.fallback_from_audio(hits, params)
            seg_trace["method"] = "audio_fallback"

        # 相邻回合的前后留白会互相覆盖，拼成片时会让同一段画面出现两次
        intervals = RA.dedupe_overlaps(intervals, hits=hits, fps=fused.fps, activity=fused.activity)

        # 再次过滤过短回合
        intervals = [iv for iv in intervals if (iv.end - iv.start) >= params.min_rally_seconds]
        intervals.sort(key=lambda v: v.start)

        # 收尾：把仍然「挨在一起」的区间合成一个。真实比赛里得分之后必然有停顿，
        # 相邻到 0.3 秒以内的两个「回合」一定是同一次对拉被切成了两半，
        # 留一条假边界只会让人误以为「这里换了一球」。
        intervals = _join_abutting(intervals)

        # ---------------------------------------------------------- 8. 特征 + 评分
        emit(0.92, "score", "提取回合特征并评分")
        RA.attach_features(intervals, fused, hits=hits, motion=motion_dict,
                           players=players_dict, shuttle=shuttle_dict)

        quality = _quality_per_rally(intervals, motion_sig, player_sig, fused.fps)
        weights = SC.PRESETS.get(weights_key, SC.PRESETS["balanced"])
        feats = [iv.features for iv in intervals]
        scores = SC.score_rallies(feats, weights, quality)

        rallies: list[Rally] = []
        for i, iv in enumerate(intervals):
            sc = scores[i] if i < len(scores) else {}
            q = quality[i] if i < len(quality) else None
            rallies.append(_build_rally(i, iv, hits, media, sc, params, q))

        # 把发球/接发球落到具体球员（依赖球员跟踪结果）
        attribute_sides(rallies, player_sig, fused.fps,
                        viewpoint=(calib.viewpoint if calib is not None else "unknown"))

        # 用「最高分回合的中间时刻」当封面，比固定取片长 25% 靠谱得多
        if rallies:
            best = max(rallies, key=lambda r: r.scores.total)
            try:
                M.ensure_poster(media, at=(best.start + best.end) / 2.0)
            except Exception:
                pass

        res.rallies = rallies
        res.hits = _hits_to_events(hits, intervals) if hits is not None else []
        res.signals = _pack_signals(fused, motion_sig, player_sig, shuttle_sig, hits, duration)
        # 姿态信号与「击球归属」掩码一并存下来：快速重切分（resegment）不重跑 AI，
        # 但它必须能复现同一套门控结果，否则「重切分」和「重跑」给出的回合数
        # 会不一样 —— 这正是用户最容易困惑的地方。
        if pose_sig is not None:
            res.signals["pose_swing"] = _downsample(pose_sig.swing)
            res.signals["pose_ok"] = _downsample(pose_sig.ok)
            res.signals["pose_overhead"] = _downsample(pose_sig.overhead)
            res.signals["pose_fps"] = [round(float(pose_sig.fps), 3)]
            res.signals["pose_coverage"] = [round(float(pose_sig.coverage), 4)]
        # 注意：这里存的是**门控之后**的击球序列，所以不需要再单独存掩码。
        # 快速重切分（resegment）不重跑 AI，它读到的就是同一套击球，
        # 门控结果自然一致。存掩码反而有风险：掩码长度和存下来的击球数
        # 一旦对不上（比如换了参数重跑过），再套一次就会把击球过滤两遍。
        res.signal_fps = fused.fps
        res.stats = SC.derive_stats(feats, scores)
        res.stats["hit_trace"] = hit_trace
        res.stats["player_trace"] = player_trace
        res.stats["shuttle_trace"] = shuttle_trace
        res.stats["pose_trace"] = pose_trace
        res.stats["weights"] = weights_key
        res.stats["audio_reliability"] = round(float(fused.audio_reliability), 3)
        res.stats["component_weights"] = {k: round(v, 3) for k, v in fused.weights.items()}
        res.stats["roi"] = list(roi) if roi else None
        res.stats["effective_roi"] = list(eff_roi) if eff_roi else None
        # 实际用于「场内判定」的多边形（标定成功时就是场地边界；不然是 None）
        res.stats["effective_poly"] = [list(p) for p in eff_poly] if eff_poly else None
        res.stats["player_size_filter"] = dict(size_filter)
        res.stats["duration"] = duration
        res.stats["segmentation"] = seg_trace
        if motion_sig is not None:
            res.stats["auto_roi"] = list(motion_sig.roi)
        res.calibration = calib.as_payload() if calib is not None else None
        res.court = _court_payload(motion_sig, player_sig, calib)
        res.status = "done"
        emit(1.0, "done", f"完成：切出 {len(rallies)} 个回合")
        return res

    except Exception as e:  # noqa: BLE001
        res.status = "error"
        res.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}"
        emit(res.progress, "error", str(e))
        return res
    finally:
        import time as _t

        res.finished_at = int(_t.time() * 1000)


# ------------------------------------------------------------------ 辅助


def _manual_poly(raw: object) -> "np.ndarray | None":
    """把手动标定的场地边界（归一化 0~1，4~24 点）转成数组；不合法返回 None。

    接受的点数上限来自 :data:`CC.MAX_POLY_POINTS`：单应拟合要在多边形顶点里
    穷举四边形，点数再多只会更慢、不会更准（真实场地的边界拐点就那么几个）。
    面积下限 0.02 是「四个点几乎重合」这类废标定 —— 它会让 ROI 退化成一个小点,
    把所有人筛掉，比不标定更糟，所以宁可当作没标定。
    """
    if not raw:
        return None
    try:
        arr = np.asarray(raw, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return None
    n = arr.shape[0]
    if n < CC.MIN_POLY_POINTS or n > CC.MAX_POLY_POINTS:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    if float(np.abs(arr).max()) > 1.5:      # 显然不是归一化坐标（比如传了像素）
        return None
    area = CC.poly_area_norm(arr)
    if area < 0.02:
        return None
    return arr


#: 旧名字（只有四个角的年代）。保留是为了让既有调用方 / 测试不用改。
_manual_quad = _manual_poly


def _build_rally(i: int, iv: RA.RallyInterval, hits, media: MediaInfo,
                 sc: dict[str, float], params: AnalysisParams,
                 quality: dict[str, float] | None = None) -> Rally:
    f = iv.features
    r = Rally(
        index=i + 1,
        start=round(iv.start, 3),
        end=round(iv.end, 3),
        duration=round(max(0.0, iv.end - iv.start), 3),
        clip_start=round(max(0.0, iv.start - params.pre_roll), 3),
        clip_end=round(min(media.duration, iv.end + params.post_roll), 3),
        serve_time=round(iv.serve_time, 3) if iv.serve_time is not None else None,
        receive_time=round(iv.receive_time, 3) if iv.receive_time is not None else None,
        scores=RallyScores(
            total=float(sc.get("total", 0.0)),
            length=float(sc.get("length", 0.0)),
            intensity=float(sc.get("intensity", 0.0)),
            technique=float(sc.get("technique", 0.0)),
            excitement=float(sc.get("excitement", 0.0)),
            production=float(sc.get("production", 0.0)),
        ),
        tags=list(sc.get("tags", [])),  # type: ignore[arg-type]
    )
    r.features = RallyFeatures(
        duration=float(f.get("duration", 0.0)),
        shot_count=int(f.get("shot_count", 0)),
        tempo=float(f.get("tempo", 0.0)),
        motion_energy=float(f.get("motion_mean", 0.0)),
        motion_peak=float(f.get("motion_peak", 0.0)),
        travel_near=0.0,
        travel_far=0.0,
        smash_count=int(f.get("smash_count", 0)),
        longest_exchange=int(f.get("shot_count", 0)),
        finish_intensity=float(f.get("finish_tempo", 0.0)),
        shuttle_speed_p50=float(f.get("shuttle_speed_p50", 0.0)),
        shuttle_speed_p95=float(f.get("shuttle_speed_p90", 0.0)),
        confidence=float(f.get("confidence", 0.5)),
        # 评分要用的中间量也存下来，换评分口径时才能用同样的原始值重算
        hit_strength_p90=float(f.get("hit_strength_p90", 0.0)),
        player_speed_mean=float(f.get("player_speed_mean", 0.0)),
        player_speed_max=float(f.get("player_speed_max", 0.0)),
        shuttle_presence=float(f.get("shuttle_presence", 0.0)),
        quality_sharpness=float((quality or {}).get("sharpness", 0.6)),
        quality_shake=float((quality or {}).get("shake", 0.3)),
        quality_subject_size=float((quality or {}).get("subject_size", 0.25)),
    )
    # 逐拍事件。**这里先不写 near/far**：能不能分清近端远端要靠
    # `attribute_sides()` 从球员跟踪结果里推断，推不出来时保持 unknown。
    # 旧版本在这里无条件写 "near"/"far" 交替，于是「推断失败」和
    # 「推断成功」在数据上完全一样 —— 也就是把猜的结果当成了测的结果。
    if hits is not None and iv.hit_indices:
        idx = iv.hit_indices
        ts = hits.times[idx]
        cs = hits.confidence[idx]
        for k in range(len(idx)):
            kind = "serve" if k == 0 else ("receive" if k == 1 else "unknown")
            r.shots.append(ShotEvent(
                time=round(float(ts[k]), 3),
                player="unknown",     # type: ignore[arg-type]
                kind=kind,            # type: ignore[arg-type]
                confidence=round(float(cs[k]), 3),
                speed=None,
            ))
    return r


def _hits_to_events(hits: AH.HitDetection, intervals: list[RA.RallyInterval]) -> list[ShotEvent]:
    keep: set[int] = set()
    for iv in intervals:
        keep.update(iv.hit_indices)
    out = []
    for i in sorted(keep):
        out.append(ShotEvent(time=round(float(hits.times[i]), 3),
                             confidence=round(float(hits.confidence[i]), 3)))
    return out


def _quality_per_rally(intervals, motion_sig, player_sig, fps: float) -> list[dict[str, float]]:
    """每个回合的画面质量（清晰度 / 抖动 / 主体大小）。"""
    out = []
    for iv in intervals:
        q = {"sharpness": 0.6, "shake": 0.3, "subject_size": 0.25}
        if motion_sig is not None:
            q["sharpness"] = _mean_range(motion_sig.sharpness, motion_sig.fps, iv.start, iv.end, 0.6)
            q["shake"] = _mean_range(motion_sig.shake, motion_sig.fps, iv.start, iv.end, 0.3)
        if player_sig is not None:
            boxes = getattr(player_sig, "frame_boxes", None)
            pfps = float(getattr(player_sig, "fps", fps))
            if boxes:
                a = max(0, int(iv.start * pfps))
                b = min(len(boxes), max(a + 1, int(iv.end * pfps)))
                heights = []
                for fr in boxes[a:b]:
                    for item in fr:
                        try:
                            _, x1, y1, x2, y2 = item
                            heights.append(abs(y2 - y1))
                        except Exception:
                            continue
                if heights:
                    q["subject_size"] = float(np.median(heights))
        out.append(q)
    return out


def _mean_range(arr, fps, t0, t1, default: float) -> float:
    if arr is None or len(arr) == 0 or fps <= 0:
        return default
    a = max(0, int(t0 * fps))
    b = min(len(arr), max(a + 1, int(t1 * fps)))
    if b <= a:
        return default
    return float(np.asarray(arr[a:b], dtype=np.float32).mean())


def _downsample(a: np.ndarray, target: int = 2400) -> list[float]:
    if a is None or len(a) == 0:
        return []
    a = np.asarray(a, dtype=np.float32)
    if a.size <= target:
        return [round(float(v), 4) for v in a]
    idx = np.linspace(0, a.size - 1, target).astype(np.int64)
    return [round(float(v), 4) for v in a[idx]]


def _pack_signals(fused, motion_sig, player_sig, shuttle_sig, hits, duration: float) -> dict[str, list[float]]:
    """打包给前端画图的下采样信号。

    另外把**满帧率的融合活跃度**也存下来（``activity_full``）：这样用户调
    「切分粒度 / 间隔判定」时可以只重跑切分这一步（毫秒级），不用再等几分钟
    的球员检测和运动分析。
    """
    out: dict[str, list[float]] = {
        "activity": _downsample(fused.activity),
        "activity_full": [round(float(v), 4) for v in fused.activity],
        "duration": [round(duration, 3)],
        "fps": [round(fused.fps, 3)],
        "threshold_hi": [round(fused.threshold_hi, 4)],
        "threshold_lo": [round(fused.threshold_lo, 4)],
    }
    for k, v in fused.components.items():
        out[f"component_{k}"] = _downsample(v)
    out["weights"] = [round(fused.weights.get(k, 0.0), 3) for k in
                      ("players", "motion", "audio_hits", "shuttle", "roi")]
    if motion_sig is not None:
        out["motion"] = _downsample(motion_sig.motion)
        out["motion_fps"] = [round(motion_sig.fps, 3)]
    if player_sig is not None:
        out["active_count"] = _downsample(getattr(player_sig, "active_count", np.zeros(0)))
        out["active_speed"] = _downsample(getattr(player_sig, "active_speed", np.zeros(0)))
        out["player_fps"] = [round(float(getattr(player_sig, "fps", 0.0)), 3)]
        # 把「球员运动曲线」以**满帧率**存下来：调切分参数时它是最有用的那一路
        # 信号（静默段就是回合边界），存满帧率才能让快速重切分忠实复现。
        boxes = getattr(player_sig, "frame_boxes", None)
        pfps = float(getattr(player_sig, "fps", 0.0) or 0.0)
        if boxes and pfps > 0:
            try:
                pm = RV.box_motion(list(boxes), pfps, window=1.0)
                pm = RV._robust_norm(RV._smooth(pm, max(1, int(pfps * 0.5))))
                out["player_motion_full"] = [round(float(v), 4) for v in pm]
                out["player_motion"] = _downsample(pm)
                # 逐帧「球员检测是否有效」也存满帧率：快速重切分要靠它决定
                # 「这一段时间该信球员还是信活跃度」，并且要把边界限制在
                # 检测有效的地方。存下来的话 resegment 就能完全复现
                # 「球员运动切分」，不必退化成只看活跃度。
                cov = RV.detection_coverage(list(boxes), pfps)
                out["player_coverage_full"] = [1.0 if v > 0.5 else 0.0 for v in cov]
            except Exception:
                pass
        # 逐帧「主体横向中心」+ 主体宽度，供竖屏自动跟随裁切使用
        if boxes:
            xs = np.full(len(boxes), np.nan, dtype=np.float32)
            ws = np.zeros(len(boxes), dtype=np.float32)
            for i, fr in enumerate(boxes):
                if not fr:
                    continue
                cx = [(float(b[1]) + float(b[3])) / 2.0 for b in fr]
                wd = [abs(float(b[3]) - float(b[1])) for b in fr]
                xs[i] = float(np.mean(cx))
                ws[i] = float(np.max(wd)) if wd else 0.0
            # 用前向填充补上缺失帧，再做滑动平均，得到平滑的跟随路径
            idx = np.arange(xs.size)
            good = ~np.isnan(xs)
            if good.sum() > 2:
                xs = np.interp(idx, idx[good], xs[good]).astype(np.float32)
                out["subject_x"] = _downsample(xs)
                out["subject_w"] = _downsample(ws)
                out["subject_fps"] = [round(float(getattr(player_sig, "fps", 0.0)), 3)]
    if player_sig is not None:
        # 两名球员在画面里的**左右跨度**：竖屏裁切要的是「两个人都进画面」，
        # 用中心均值会把镜头对准两人中间（侧方机位下等于谁都没对准）。
        left, right = _subject_span(getattr(player_sig, "frame_boxes", None) or [])
        if left is not None and right is not None:
            out["subject_left"] = _downsample(left)
            out["subject_right"] = _downsample(right)
            out["subject_fps"] = [round(float(getattr(player_sig, "fps", 0.0)), 3)]
    if shuttle_sig is not None:
        out["shuttle_presence"] = _downsample(getattr(shuttle_sig, "presence", np.zeros(0)))
        out["shuttle_fps"] = [round(float(getattr(shuttle_sig, "fps", 0.0)), 3)]
    if hits is not None and hits.times.size:
        out["hit_times"] = [round(float(v), 3) for v in hits.times]
        out["hit_strength"] = [round(float(v), 3) for v in hits.strength]
        out["hit_confidence"] = [round(float(v), 3) for v in hits.confidence]
        out["envelope"] = _downsample(hits.envelope)
        out["envelope_fps"] = [round(float(hits.env_fps), 3)]
    return out


def _subject_span(frame_boxes: list) -> tuple[np.ndarray | None, np.ndarray | None]:
    """逐帧「所有在场球员的最左 / 最右边界」，用于竖屏自动裁切。

    旧实现取的是所有框中心的**均值**：侧方机位下两名球员在画面两端，
    均值正好落在他们中间，裁出来谁都看不清。取左右包络后，
    裁切框可以按「把 [left, right] 都装进去」来定，双打也能一起进画面。
    """
    n = len(frame_boxes)
    if n == 0:
        return None, None
    left = np.full(n, np.nan, dtype=np.float32)
    right = np.full(n, np.nan, dtype=np.float32)
    for i, fr in enumerate(frame_boxes):
        if not fr:
            continue
        xs0 = [float(b[1]) for b in fr if len(b) >= 5]
        xs1 = [float(b[3]) for b in fr if len(b) >= 5]
        if not xs0:
            continue
        left[i] = min(xs0)
        right[i] = max(xs1)
    idx = np.arange(n)
    for arr in (left, right):
        good = ~np.isnan(arr)
        if good.sum() > 2:
            arr[:] = np.interp(idx, idx[good], arr[good])
    # 平滑，避免裁切框抖动
    win = max(1, int(round(0.8 * 12)))
    if left.size > win:
        left = RV._smooth(left, win)
        right = RV._smooth(right, win)
    return left, right


def _court_payload(motion_sig, player_sig, calib: CC.CourtCalibration | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"source": "auto"}
    if motion_sig is not None:
        payload["roi"] = list(motion_sig.roi)
    if player_sig is not None and getattr(player_sig, "frame_boxes", None):
        payload["active_player_ids"] = list(getattr(player_sig, "active_player_ids", []) or [])
    if calib is not None:
        payload["calibration"] = calib.as_payload()
        if calib.ok:
            payload["source"] = "color+homography"
    return payload


# ------------------------------------------------------------------ 切分调度


def _join_abutting(intervals: list[RA.RallyInterval],
                   gap: float = 0.3) -> list[RA.RallyInterval]:
    """把间隔小于 ``gap`` 秒的相邻区间合并成一个。

    真实比赛里得分之后必然有捡球 / 换发球的停顿，两个回合之间不会没有间隔。
    出现「前一个的 end 正好等于后一个的 start」说明这其实是同一次对拉被切成了
    两半（球员被短暂遮挡、或者活跃度抖了一下）。合并掉比留一条假边界好：
    假边界会让用户以为这里换了一球，还会在时间线上切出多余的片段。
    """
    if len(intervals) < 2:
        return intervals
    out: list[RA.RallyInterval] = []
    for iv in sorted(intervals, key=lambda v: v.start):
        if out and iv.start - out[-1].end < gap:
            prev = out[-1]
            prev.end = max(prev.end, iv.end)
            prev.confidence = max(prev.confidence, iv.confidence)
            # 击球索引合并后不再连续，但**仍然有效**（它们是全局下标的并集）。
            # 旧代码在这里直接清空，于是合并出来的回合一律显示「0 拍」——
            # 而 shot_count 是参与评分的，等于让「被粘起来的回合」拿不到力度分。
            prev.hit_indices = sorted(set(prev.hit_indices) | set(iv.hit_indices))
            if prev.features.get("hit_anchored") and iv.features.get("hit_anchored"):
                g = max(float(prev.features.get("tail_gap", 0.0)),
                        float(iv.features.get("tail_gap", 0.0)))
                prev.features["tail_gap"] = g
            if prev.serve_time is None:
                prev.serve_time = iv.serve_time
            continue
        out.append(iv)
    return out


def _segments_to_intervals(segs: list[RV.RawSegment]) -> list[RA.RallyInterval]:
    """把切分结果转成回合区间（相邻候选交给 :func:`_join_abutting` 合并）。"""
    if not segs:
        return []
    segs = sorted(segs, key=lambda s: (s.start, -s.end))
    groups: list[list[RV.RawSegment]] = [[segs[0]]]
    for s in segs[1:]:
        if s.start - groups[-1][-1].end < 0.3:
            groups[-1].append(s)
        else:
            groups.append([s])
    out: list[RA.RallyInterval] = []
    for g in groups:
        start = min(x.start for x in g)
        end = max(x.end for x in g)
        score = max(x.score for x in g)
        iv = RA.RallyInterval(start=start, end=end,
                              confidence=float(np.clip(0.25 + 0.75 * score, 0.05, 1.0)))
        for x in g:
            for k, v in (x.meta or {}).items():
                if isinstance(v, (int, float)):
                    iv.features[k] = float(v)
        out.append(iv)
    return out


def _seg_quality(segs: list[RV.RawSegment], fps: float = 15.0,
                 hit_density: np.ndarray | None = None) -> float:
    """给一套切分结果打一个「有多可信」的分数，用于 hybrid 模式下择优。

    奖励「回合之间真的有停顿」和「时长分布合理」，惩罚两件事：

    * **首尾相接**：前一个回合的结束正好等于后一个的开始。真实比赛里
      得分之后必然有捡球/换发球，出现首尾相接说明「回合之间」根本没被识别出来。
    * **时长过于整齐**：一堆几乎一样长的片段说明是机械等分出来的。

    另外（``hit_density`` 可用时）奖励「回合内击球密集、回合间稀疏」的
    对比度。没有它，择优会在两条**都不准**的路径里挑一条看着整齐的：
    实测把松弛过的参数同时给了球员路径和活跃度路径之后，活跃度路径因为
    段数少、间隔大而拿到更高分，结果真正的「球员运动 × 击球密度」证据路径
    被丢掉（HANDOVER 第 10 节提醒过的「整体漂移」）。
    """
    if not segs:
        return 0.0
    segs = sorted(segs, key=lambda s: s.start)
    n = len(segs)
    durs = np.asarray([s.end - s.start for s in segs], dtype=np.float64)
    gaps = np.asarray([max(0.0, segs[i + 1].start - segs[i].end) for i in range(n - 1)],
                      dtype=np.float64) if n > 1 else np.zeros(0)
    rest_ratio = float(np.mean(gaps > 1.0)) if gaps.size else 0.0
    # 时长离散度：太大（混了长休息）太小（机械等分）都不好
    cv = float(durs.std() / max(durs.mean(), 1e-6))
    spread_score = float(np.clip(cv / 0.45, 0.0, 1.0)) if cv < 0.45 else float(np.clip((1.2 - cv) / 0.6, 0.0, 1.0))
    dur_ok = float(np.clip(np.mean((durs >= 4.0) & (durs <= 90.0)), 0.0, 1.0))
    mean_score = float(np.mean([s.score for s in segs]))
    q = float(np.clip(0.45 * rest_ratio + 0.25 * spread_score + 0.15 * dur_ok + 0.15 * mean_score,
                      0.0, 1.0))
    if hit_density is not None and hit_density.size and fps > 0:
        # 击球证据的「F 值代理」：p = 段内平均击球密度，r = 被段覆盖的
        # 击球密度总量占比。它直接度量「选择的这些段有没有把在连续打球的
        # 时刻包住」，比段数 / 间隔这些形状指标更接近真正要的东西。
        # 证据可用时以它为主：只靠形状指标会挑出「段数少、间隔大」的机械等分
        # 结果（实测活跃度路径 5 段，形状分 0.86，但只覆盖了 19% 的击球证据）。
        m = hit_density.size
        mask = np.zeros(m, dtype=bool)
        for s in segs:
            a = max(0, min(m - 1, int(s.start * fps)))
            b = max(a + 1, min(m, int(s.end * fps)))
            mask[a:b] = True
        total = float(hit_density.sum())
        if mask.any() and total > 0.0:
            p = float(hit_density[mask].mean())
            r = float(hit_density[mask].sum() / total)
            hit_f = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
            q = 0.3 * q + 0.7 * float(np.clip(hit_f, 0.0, 1.0))
    return float(np.clip(q, 0.0, 1.0))


def _segment_rallies(
    fused: RA.FusedSignal,
    params: AnalysisParams,
    duration: float,
    player_sig=None,
    shuttle_sig=None,
    player_motion: np.ndarray | None = None,
    player_coverage: np.ndarray | None = None,
    hits: Any = None,
    opt_override: RV.SegmentOptions | None = None,
) -> tuple[list[RA.RallyInterval], dict[str, Any]]:
    """按 ``params.segment_mode`` 选切分方式，返回 (区间, 诊断信息)。

    优先级：

    1. **球员运动切分**（``player_motion``）：球员是唯一贯穿「发球→对拉→死球」
       全过程、而且不会被隔壁场地影响的观测对象。它的「静默段」就是回合边界，
       不需要手工阈值，也不会出现首尾相接的片段。
    2. **活跃度谷值切分**（``activity_valleys``）：没有球员信号时用融合活跃度，
       但只承认显著度足够的谷，不再做机械等分。
    3. **旧的迟滞状态机**（``activity_state_machine``）：前两条都拿不出东西时的
       最后兜底，行为与旧版本一致。

    ``player_motion`` / ``player_coverage`` 是**已经算好的**球员运动曲线与检测
    覆盖率（都在 ``fused.fps`` 上）。快速重切分（:func:`resegment`）没有球员框，
    但分析时把这两路曲线存下来了，于是它也能走「球员运动切分」这条路，
    而不是被迫退化成只看活跃度 —— 否则用户会看到「重切分之后的回合数
    和完整重跑对不上」。
    """
    opt = RV.SegmentOptions(
        min_rally=float(params.min_rally_seconds),
        max_rally=float(params.max_rally_seconds),
        pre_roll=float(params.pre_roll),
        post_roll=float(params.post_roll),
        # 静默段尺度：人工标注校准出来的值优先，旧参数里没有时退回 SegmentOptions
        # 的默认值（用 getattr 兜底，保证旧工程反序列化后仍能工作）。
        min_quiet=float(getattr(params, "seg_min_quiet", 0.7)),
        prominence_ratio=float(getattr(params, "seg_prominence", 0.18)),
        min_rest=float(getattr(params, "seg_min_rest", 0.8)),
        min_core=float(getattr(params, "seg_min_core", 1.0)),
    )
    #: 允许调用方（标注校准脚本 / 测试）覆盖静默段检测的参数。
    #: 这些参数不在 ``AnalysisParams`` 里（它们是切分算法的内部尺度），
    #: 但用人工标注校准它们恰恰是最有价值的一件事。
    if opt_override is not None:
        opt = opt_override
    shuttle_presence = getattr(shuttle_sig, "presence", None)
    shuttle_fps = float(getattr(shuttle_sig, "fps", 0.0) or 0.0) if shuttle_sig is not None else 0.0
    boxes = getattr(player_sig, "frame_boxes", None) if player_sig is not None else None
    player_fps = float(getattr(player_sig, "fps", 0.0) or 0.0) if player_sig is not None else 0.0

    # 击球密度：这条素材上区分度最高的一路证据。它同时用在两条路径里 ——
    # 乘进球员运动（`audio_visual_evidence`）以及传给 `segment_visual`。
    # 快速重切分（resegment）通过 `hits` 把它复原出来，不依赖重跑 AI。
    hit_density = (RA.hit_density_signal(hits, fused.fps, fused.activity.size)
                   if hits is not None and getattr(hits, "times", np.zeros(0)).size else None)

    mode = params.segment_mode
    trace: dict[str, Any] = {"mode": mode}

    vis: list[RV.RawSegment] = []
    coverage = 0.0
    #: 逐帧覆盖率数组 —— 注意别和上面的 ``coverage``（有效时间占比）混了：
    #: 一个喂给「按时间段择优」，一个是给用户看的诊断标量。
    cov_arr: np.ndarray | None = None
    #: 球员运动曲线（``fused.fps`` 上）——「按时间段择优」判断「球员切分没给候选
    #: 到底是漏检还是真停顿」时要用它。
    pm_arr: np.ndarray | None = None
    want_players = mode in ("auto", "hybrid")
    if want_players and player_motion is not None and player_motion.size:
        # 快速重切分路径：球员运动曲线与覆盖率是分析时存下来的，直接复用。
        pm = RV._resample_to(player_motion, player_fps or fused.fps, fused.fps,
                             fused.activity.size)
        cov_arr = (RV._resample_to(player_coverage, player_fps or fused.fps, fused.fps,
                                   fused.activity.size)
                   if player_coverage is not None and player_coverage.size else None)
        coverage = float(np.mean(cov_arr > 0.5)) if cov_arr is not None else 0.0
        pm_arr = pm
        # 切分用的是「球员运动 × 击球密度」；`pm_arr` 保留原始球员运动给
        # `_merge_by_availability` 做「球员到底动没动」的判断（那里不该被
        # 击球密度影响，否则漏检的击球会让「真停顿」被误判成「在打球」）。
        ev = RV.audio_visual_evidence(pm, hit_density)
        vis = RV.segment_by_player_motion(RV.SegmentSignals(
            fps=fused.fps, duration=duration, player_motion=ev,
            activity=fused.activity,
            shuttle=(RV._resample_to(shuttle_presence, shuttle_fps, fused.fps,
                                     fused.activity.size)
                     if shuttle_presence is not None and shuttle_fps > 0 else None),
            has_players=True, coverage=cov_arr, coverage_ratio=coverage,
        ), opt)
        if vis:
            vis = RV.verify_with_shuttle(vis, RV.SegmentSignals(
                fps=fused.fps, duration=duration, player_motion=ev, has_players=True,
                shuttle=(RV._resample_to(shuttle_presence, shuttle_fps, fused.fps,
                                         fused.activity.size)
                         if shuttle_presence is not None and shuttle_fps > 0 else None),
            ), opt)
    elif want_players:
        vis, coverage = RV.segment_visual(
            fps=fused.fps, duration=duration,
            player_boxes=boxes, player_fps=player_fps,
            activity=fused.activity,
            shuttle_presence=shuttle_presence, shuttle_fps=shuttle_fps,
            hit_density=hit_density, opt=opt,
        )
        if boxes and player_fps > 0:
            # 只为了「择优」时判断球员动没动，不需要归一化（门限是相对的）
            try:
                pm_arr = RV._resample_to(RV.box_motion(list(boxes), player_fps, window=1.0),
                                         player_fps, fused.fps, fused.activity.size)
            except Exception:
                pm_arr = None
    trace["player_coverage"] = round(float(coverage), 3)
    if player_motion is not None and player_motion.size:
        trace["player_signal"] = "reused"

    # 活跃度切分很便宜（就是一次谷值检测），所以 auto 模式下总是算一遍。
    # 只有算出来才谈得上「按时间段择优」；用「覆盖率低于某个数才算」这种
    # 门槛反而会在临界值附近（实测 0.49 / 0.53 两种都出现过）漏掉整条路径。
    need_activity = mode in ("activity", "hybrid", "auto")

    act: list[RV.RawSegment] = []
    if need_activity:
        act = RV.segment_activity(
            fused.activity, fused.fps, duration, opt,
            shuttle_presence=shuttle_presence, shuttle_fps=shuttle_fps,
        )
    trace["activity_count"] = len(act)

    # auto 模式：按**时间段**决定信谁。球员检测好的时间用球员切分（边界是
    # 球员真的停下来，最可靠）；球员检测失效的时间用活跃度切分补齐。
    # 这个判断必须是局部的：同一段素材里球员可能前半段检得到、后半段检不到，
    # 用一个全片平均覆盖率去做「全局二选一」会导致某一段整体漏掉。
    #
    # 注意这里**不覆盖** ``vis``：合并结果作为一个**并列候选**参与后面的择优。
    # 旧实现直接 `vis = merged`，于是「合并」这个动作不可能被否掉；实测把
    # 击球密度证据路径交给合并器之后，它会在窗口里把一个长候选拼回来
    # （实测拼出一个 19s 的区间，把两个真实回合粘在一起），而 `_seg_quality`
    # 因为只看段数/间隔反而觉得它「更整齐」。留成并列候选才能按证据择优。
    merged_vis: list[RV.RawSegment] | None = None
    if (mode == "auto" and vis and act and (boxes or cov_arr is not None)
            and hit_density is None):
        merged_vis = _merge_by_availability(
            vis, act, boxes, player_fps, fused.fps, duration,
            coverage=cov_arr, player_motion=pm_arr,
            merge_trace=trace.setdefault("windows", {}))
        trace["merged_from"] = {"player_motion": len(vis), "activity_valleys": len(act)}

    legacy: list[RA.RallyInterval] = []
    if not vis and not merged_vis and not act:
        legacy = RA.segment(
            fused,
            gap_seconds=params.gap_seconds,
            min_seconds=params.min_rally_seconds,
            max_seconds=params.max_rally_seconds,
            target_seconds=params.target_rally_seconds,
            split_sensitivity=params.split_sensitivity,
        )
        trace.update(method="activity_state_machine", count=len(legacy))
        return legacy, trace

    if mode == "hybrid" and vis and act:
        qv, qa = (_seg_quality(vis, fused.fps, hit_density),
                  _seg_quality(act, fused.fps, hit_density))
        trace["quality"] = {"player_motion": round(qv, 3), "activity_valleys": round(qa, 3)}
        if qa > qv + 0.05:
            trace.update(method="activity_valleys", count=len(act))
            return _segments_to_intervals(act), trace
        trace.update(method="player_motion", count=len(vis))
        return _segments_to_intervals(vis), trace

    # 三路候选并列择优：未合并的证据路径 / 按时间段合并后的路径 / 纯活跃度。
    # 评判见 :func:`_seg_quality`（其中「击球证据对齐度」是「哪条路径真的对上了
    # 『在连续打球』这件事」的客观依据）。
    cands: list[tuple[list[RV.RawSegment], str]] = []
    if vis:
        cands.append((vis, "player_motion"))
    if merged_vis:
        cands.append((merged_vis, "player_motion+activity"))
    if act:
        cands.append((act, "activity_valleys"))
    scored = [(_seg_quality(c, fused.fps, hit_density), c, name) for c, name in cands]
    trace["quality"] = {name: round(q, 3) for q, _, name in scored}
    best = max(scored, key=lambda t: t[0]) if scored else (0.0, [], "empty")
    trace.update(method=best[2], count=len(best[1]))
    return _segments_to_intervals(best[1]), trace


def _merge_by_availability(
    vis: list[RV.RawSegment],
    act: list[RV.RawSegment],
    player_boxes: list,
    player_fps: float,
    fps: float,
    duration: float,
    coverage: np.ndarray | None = None,
    player_motion: np.ndarray | None = None,
    merge_trace: dict[str, Any] | None = None,
) -> list[RV.RawSegment]:
    """按时间段在「球员切分」和「活跃度切分」之间择优。

    做法是把两边的候选按时间切成互不重叠的时间窗，每个窗口内：

    * 算这一窗里**球员检测的有效比例**（``detection_coverage``）；
    * 有效比例高 → 用球员切分给的回合（边界就是球员真的停下来，最可靠）；
    * 有效比例低 → 用活跃度切分给的回合（球员切分在这里不可信，
      那里的「静默」可能只是检不到人）。

    为什么必须做成局部判断：同一段素材里比赛球员完全可能前半段检得到、
    后半段检不到。用一个全片平均覆盖率做「全局二选一」，要么在球员信号差的
    时间段整体漏掉回合，要么在球员信号好的时间段被活跃度切分的粗边界带偏。
    实测 30 分钟素材上，球员检测只有 30%~54% 的时间是有效的，两种做法
    差出 2~3 倍的回合数。

    **每一窗只信一边**，并且分三种情况：

    * 球员检测有效 → 球员切分说了算；
    * 球员检测有效、但球员切分没给候选 → 要看**球员到底动没动**：
      球员在动说明是球员切分的漏检（它对「移动占空比 / 核心片段长度」有门限，
      混合片段容易被整个丢掉），用活跃度切分兜住；球员没动才是真正的停顿，
      留空；
    * 球员检测无效 → 用活跃度切分。

    中间那一条是必需的：只写「球员切分没候选就留空」会把整段回合丢掉
    （实测召回率掉 6 个百分点），而只写「没候选就退回活跃度」又会把
    球员切分好不容易切开的回合重新粘回去（实测最长回合从 22.7 秒涨回 28.7 秒）。
    """
    if not vis:
        return act
    if not act:
        return vis
    n = max(1, int(round(duration * fps)))
    if coverage is not None and len(coverage) == n:
        # 快速重切分路径：覆盖率是分析时存下来的，直接用
        cov = np.asarray(coverage, dtype=np.float32)
    elif player_boxes:
        cov = RV._resample_to(RV.detection_coverage(player_boxes, player_fps),
                              player_fps, fps, n)
    else:
        # 既没有框也没有存下来的覆盖率：不能凭空断言「检测失效」，
        # 那样会把所有候选都判给活跃度切分。按「全程有效」处理。
        cov = np.ones(n, dtype=np.float32)
    thr = 0.5

    def valid_ratio(a: float, b: float) -> float:
        i0 = max(0, min(n - 1, int(a * fps)))
        i1 = max(i0 + 1, min(n, int(b * fps)))
        return float(np.mean(cov[i0:i1] > thr))

    # 球员运动曲线：用来判断「球员切分没给候选」到底是漏检还是真停顿。
    # 门限与 rally_vision.segment_by_player_motion 保持一致（静息地板 + 25% 动态范围）。
    if player_motion is not None and player_motion.size:
        if player_motion.size == n:
            pm = np.asarray(player_motion, dtype=np.float32)
        else:
            # 调用方一般都已经重采样到 fps 上了；这里兜个底，
            # 按「数组长度 / 总时长」推出它自己的帧率再重采样。
            src_fps = player_motion.size / max(duration, 1e-6)
            pm = RV._resample_to(player_motion, src_fps, fps, n)
        base = float(np.percentile(pm, 20))
        span = max(float(np.percentile(pm, 95)) - base, 1e-9)
        move_thr = base + 0.25 * span
        moving = pm >= move_thr
    else:
        moving = None

    def moving_ratio(a: float, b: float) -> float:
        if moving is None:
            return 0.0
        i0 = max(0, min(n - 1, int(a * fps)))
        i1 = max(i0 + 1, min(n, int(b * fps)))
        return float(np.mean(moving[i0:i1]))

    def has_motion_burst(a: float, b: float, min_run: float = 1.2,
                         min_duty: float = 0.35) -> bool:
        """这一段里有没有「球员连着动了一阵」的片段。

        为什么不看整段的平均移动比例：一个 20 秒的窗口里可能只有 8 秒在打
        （其余在捡球），平均下来比例很低，但**那 8 秒确实是回合** ——
        实测正是这个原因把 29~51 秒整段判成了「没人动」，一口气丢掉 20 秒的比赛。

        门限比 ``rally_vision`` 里的 ``min_core``（2.5 秒）**故意放宽到 1.2 秒**：
        这个函数是**召回的安全网**，宁可疑心一点也别把整段回合丢掉；
        而「真的没人动」这个判断本身有足够强的证据 —— 球
        """
        if moving is None:
            return False
        i0 = max(0, min(n - 1, int(a * fps)))
        i1 = max(i0 + 1, min(n, int(b * fps)))
        idx = np.nonzero(moving[i0:i1])[0]
        if idx.size == 0:
            return False
        gap = max(1, int(round(0.35 * fps)))
        splits = np.nonzero(np.diff(idx) > gap)[0]
        bounds = np.concatenate([[0], splits + 1, [idx.size]])
        for k in range(len(bounds) - 1):
            run = int(idx[bounds[k + 1] - 1]) - int(idx[bounds[k]])
            if run < min_run * fps:
                continue
            seg = moving[i0 + int(idx[bounds[k]]): i0 + int(idx[bounds[k + 1] - 1]) + 1]
            if float(np.mean(seg)) >= min_duty:
                return True
        return False

    # 时间窗只由前两路的边界决定。**击球序列（``extra``）刻意不参与划分** ——
    # 它只用来「补空」。理由很实际：窗口是靠边界点排序切出来的，多一路边界
    # 会把所有窗口重新划分一遍，于是「补空」这个动作会顺带改变已经定好的
    # 那些窗口，结果整体漂移（实测第一段回合直接消失）。补空就该只补空。
    marks = sorted({0.0, duration}
                   | {s.start for s in vis} | {s.end for s in vis}
                   | {s.start for s in act} | {s.end for s in act})
    out: list[RV.RawSegment] = []
    used = {"player_motion": 0, "activity_valleys": 0, "rescued": 0, "empty": 0}
    for k in range(len(marks) - 1):
        w0, w1 = marks[k], marks[k + 1]
        if w1 - w0 < 0.2:
            continue
        mid = (w0 + w1) / 2.0
        # 关键：**按这一窗里球员检测到底有没有效来选信谁**，而不是「谁有候选就用谁」。
        # 旧实现写的是「vis 有候选就用 vis，没有就退回 act」，把 valid_ratio 算出来
        # 却没用到 —— 于是凡是球员切分**故意留空**的地方（那正是两个回合之间真实的
        # 停顿），都会被活跃度切分的候选补上，球员切分好不容易切开的回合又被粘回去。
        if valid_ratio(w0, w1) >= thr:
            candidates = [s for s in vis if s.start <= mid < s.end]
            src = "player_motion"
            if not candidates and has_motion_burst(w0, w1):
                # 球员确实连着动过 → 球员切分这一窗是漏检，用活跃度候选兜住，
                # 别把一整个回合丢掉。两路都沉默才算「这一窗真的没有回合」。
                candidates = [s for s in act if s.start <= mid < s.end]
                src = "rescued"
        else:
            candidates = [s for s in act if s.start <= mid < s.end]
            src = "activity_valleys"
        if not candidates:
            used["empty"] += 1
            continue
        used[src] += 1
        s = max(candidates, key=lambda x: x.end - x.start)
        out.append(RV.RawSegment(start=max(w0, s.start), end=min(w1, s.end),
                                 score=s.score, meta=dict(s.meta or {})))
    if merge_trace is not None:
        merge_trace.update(used)

    # 相邻窗口可能来自同一个候选，直接按时间合并
    merged: list[RV.RawSegment] = []
    for s in sorted(out, key=lambda x: x.start):
        if merged and s.start - merged[-1].end < 0.3:
            prev = merged[-1]
            prev.end = max(prev.end, s.end)
            prev.score = max(prev.score, s.score)
            continue
        merged.append(s)
    return [s for s in merged if s.end - s.start >= 1.0]


__all__ = ["run_analysis", "resegment"]


# ------------------------------------------------------------------ 快速重切分


def resegment(
    res: AnalysisResult,
    params: AnalysisParams | None = None,
    weights_key: str = "balanced",
) -> AnalysisResult:
    """只重跑「切分 + 特征 + 评分」，复用已经算好的活跃度曲线。

    调「切分粒度 / 间隔判定 / 留白 / 评分口径」时用它，毫秒级出结果，
    不必再等几分钟的球员检测与运动分析。前提是分析结果里存了
    ``signals["activity_full"]``（新版本分析都会存）。

    逐拍信息用存下来的音频击球时刻重建，所以「发球/接发球」这类边界
    依然能对齐到真实的击球声。
    """
    params = params or res.params
    sig = res.signals or {}
    full = sig.get("activity_full") or []
    if not full:
        raise RuntimeError("这次分析没有保存满帧率活跃度曲线，无法快速重切分，请重新运行一次 AI 分析")

    fps = float((sig.get("fps") or [12.0])[0]) or 12.0
    duration = float((sig.get("duration") or [0.0])[0]) or 0.0
    if duration <= 0:
        duration = len(full) / fps

    act = np.asarray(full, dtype=np.float32)
    med = float(np.median(act))
    fused = RA.FusedSignal(
        fps=fps,
        duration=duration,
        activity=act,
        components={},
        weights={},
        threshold_hi=max(float(np.percentile(act, 78)), med * 1.25),
        threshold_lo=max(float(np.percentile(act, 55)) * 0.92, med * 1.05),
        audio_reliability=float(res.stats.get("audio_reliability", 1.0) or 1.0),
    )

    # 重切分时没有球员框，但球员**运动曲线**和检测覆盖率是分析时存下来的
    # （``player_motion_full`` / ``player_coverage_full``），所以这里可以走
    # 和完整分析完全一样的 :func:`_segment_rallies`，而不是退化成只看活跃度。
    # 旧版本正是因为这里退化，用户才会看到「快速重切分之后的回合数和
    # 完整重跑对不上」——而重切分恰恰是调「留白 / 最短最长回合」时最常用的操作。
    hits = _rebuild_hits(sig)
    # 这里**不需要**再套一次「击球归属」门控：存进 signals 的 hit_times 本身
    # 就是门控之后的序列（见 run_analysis 里的 _pack_signals 调用）。
    # 再套一次会把击球过滤两遍。
    shuttle_presence = None
    shuttle_fps = 0.0
    if sig.get("shuttle_presence"):
        shuttle_presence = np.asarray(sig["shuttle_presence"], dtype=np.float32)
        shuttle_fps = float((sig.get("shuttle_fps") or [10.0])[0])

    pm_full = sig.get("player_motion_full") or []
    cov_full = sig.get("player_coverage_full") or []
    player_fps = float((sig.get("player_fps") or [0.0])[0]) or 0.0
    player_motion = (np.asarray(pm_full, dtype=np.float32)
                     if pm_full and player_fps > 0 else None)
    player_coverage = (np.asarray(cov_full, dtype=np.float32)
                       if cov_full and len(cov_full) == len(pm_full) else None)
    if player_motion is None:
        player_fps = 0.0

    intervals, seg_trace = _segment_rallies(
        fused=fused, params=params, duration=duration,
        player_sig=None, shuttle_sig=None,
        player_motion=player_motion, player_coverage=player_coverage,
        hits=hits,
    )
    seg_method = str(seg_trace.get("method") or "")

    # 边界锚定：快慢两条路径都要做，否则「终点收不回来」这个毛病会从
    # 重新切分这条路上又冒出来（用户会看到「重切分之后又变长了」）。
    if hits is not None:
        intervals = RA.refine_with_hits(
            intervals, hits,
            pre_roll=params.pre_roll,
            post_roll=params.post_roll,
            tail_seconds=params.hit_tail_seconds,
            # 与 run_analysis 保持一致：球员切分的起点是「球员开始动」，
            # 本身已含发球准备，不该再被推迟到第一拍。旧版 resegment 漏了这个
            # 开关（默认 True），于是同一工程「快速重切分」会把起点整体后移，
            # 和完整重跑给出的边界对不上。
            trim_start=seg_method != "player_motion",
        )
    else:
        for iv in intervals:
            iv.start = max(0.0, iv.start - params.pre_roll)
            iv.end = min(duration, iv.end + params.post_roll)

    intervals = RA.dedupe_overlaps(intervals, hits=hits, fps=fps, activity=act)
    intervals = [iv for iv in intervals if (iv.end - iv.start) >= params.min_rally_seconds]
    intervals.sort(key=lambda v: v.start)
    # 与 run_analysis 保持一致：紧挨在一起的两段是同一次对拉被切开的两半，
    # 留一条假边界只会让人误以为「这里换了一球」。
    # （旧版 resegment 漏了这一步，于是「快速重切分」和「完整重跑」给出
    #   不一样的结果 —— 同一个工程重切分之后回合数会变。）
    intervals = _join_abutting(intervals)

    motion_dict = _dict_from(sig, "motion", "motion_fps")
    players_dict = _dict_from_multi(sig, ("active_count", "active_speed", "max_speed"), "player_fps")
    shuttle_dict = _dict_from_multi(sig, ("presence", "max_candidate_speed"), "shuttle_fps")
    if shuttle_dict and "presence" in shuttle_dict:
        shuttle_dict["presence"] = shuttle_dict.pop("presence")

    RA.attach_features(intervals, fused, hits=hits, motion=motion_dict,
                       players=players_dict, shuttle=shuttle_dict)

    quality = [_quality_default() for _ in intervals]
    weights = SC.PRESETS.get(weights_key, SC.PRESETS["balanced"])
    feats = [iv.features for iv in intervals]
    scores = SC.score_rallies(feats, weights, quality)

    media_stub = _MediaStub(duration)
    kept: dict[str, Rally] = {}
    for r in res.rallies:
        kept[r.id] = r

    rallies: list[Rally] = []
    for i, iv in enumerate(intervals):
        sc = scores[i] if i < len(scores) else {}
        r = _build_rally(i, iv, hits, media_stub, sc, params)
        # 保留用户在旧回合上的编辑：按时间重叠把 starred / note / keep 迁移过来
        old = _best_overlap(r, res.rallies)
        if old is not None:
            r.starred = old.starred
            r.note = old.note
            r.keep = old.keep
        rallies.append(r)

    res.rallies = rallies
    res.hits = _hits_to_events(hits, intervals) if hits is not None else []
    res.params = params
    # 先把「重切分不会改变」的那部分统计留出来，再整体替换
    keep_keys = ("hit_trace", "player_trace", "shuttle_trace", "pose_trace",
                 "audio_reliability", "component_weights", "duration", "auto_roi",
                 "roi", "effective_roi", "effective_poly", "player_size_filter",
                 "segmentation")
    carried = {k: v for k, v in (res.stats or {}).items() if k in keep_keys}
    carried.setdefault("duration", duration)
    res.stats = SC.derive_stats(feats, scores)
    res.stats.update(carried)
    # 保留切分诊断：以前这里只写 method/count，把「按时间段择优」每一窗
    # 到底信了谁、哪些窗被判成「没有回合」全丢了 —— 而那正是排查
    # 「为什么这一段回合不见了」时唯一有用的信息。
    seg_trace = dict(seg_trace or {})
    seg_trace.update({"mode": params.segment_mode, "method": seg_method,
                      "count": len(intervals), "resegmented": True})
    res.stats["segmentation"] = seg_trace
    res.stats["weights"] = weights_key
    res.stats["resegmented"] = True
    res.signal_fps = fps
    return res


def _quality_default() -> dict[str, float]:
    return {"sharpness": 0.6, "shake": 0.3, "subject_size": 0.25}


class _MediaStub:
    """``_build_rally`` 只用到 duration，这里给个最小替身。"""

    def __init__(self, duration: float) -> None:
        self.duration = duration


def _rebuild_hits(sig: dict) -> AH.HitDetection | None:
    t = sig.get("hit_times") or []
    if not t:
        return None
    s = sig.get("hit_strength") or [1.0] * len(t)
    c = sig.get("hit_confidence") or [0.5] * len(t)
    return AH.HitDetection(
        times=np.asarray(t, dtype=np.float64),
        strength=np.asarray(s, dtype=np.float32),
        confidence=np.asarray(c, dtype=np.float32),
        envelope=np.zeros(0),
        env_fps=float((sig.get("envelope_fps") or [250.0])[0]),
    )


def _dict_from(sig: dict, key: str, fps_key: str) -> dict | None:
    v = sig.get(key)
    if not v:
        return None
    return {"fps": float((sig.get(fps_key) or [12.0])[0]), key: np.asarray(v, dtype=np.float32)}


def _dict_from_multi(sig: dict, keys: tuple[str, ...], fps_key: str) -> dict | None:
    out: dict = {}
    for k in keys:
        v = sig.get(k)
        if v:
            out[k] = np.asarray(v, dtype=np.float32)
    if not out:
        return None
    out["fps"] = float((sig.get(fps_key) or [12.0])[0])
    return out


def _best_overlap(r: Rally, olds: list[Rally]) -> Rally | None:
    best: Rally | None = None
    best_ov = 0.0
    for o in olds:
        ov = max(0.0, min(r.end, o.end) - max(r.start, o.start))
        if ov > best_ov:
            best_ov, best = ov, o
    # 至少一半以上重叠才认为是「同一个回合的延续」
    return best if best is not None and best_ov >= 0.5 * max(0.1, r.duration) else None
