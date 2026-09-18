"""人工标注的读写，以及「用标注自动优化切分参数」。

为什么需要它
------------
切分参数（静默段尺度、最短回合等）过去只能靠肉眼看结果、凭感觉调 ——
HANDOVER 第 8.6 节明确说「没有 ground truth 就无法判断调参对不对」。
有了人工标注之后，这件事变成一个可量化的问题：

1. 把标注当作 ground truth；
2. 在参数网格上重跑切分（复用已存好的信号，毫秒级，不重跑 AI）；
3. 用 IoU 匹配算 Precision / Recall / F1，挑 F1 最高的一组；
4. 把最优参数写回 ``AnalysisParams``，再走一次 resegment 就应用到工程上。

本模块只做「读标注 / 存标注 / 评估 / 搜参」，不碰 HTTP；路由见
:mod:`bms.api.annotations`。
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..config import ANNOTATIONS_DIR
from . import pipeline as P
from . import rally as RA
from . import rally_vision as RV
from ..core.models import AnalysisParams, AnalysisResult, MediaInfo

#: 参数搜索网格。每一项是 (字段名, 候选值)。
#: 只搜真正影响切分的量：静默段四个尺度 + 最短回合。``pre_roll`` / ``post_roll``
#: / ``hit_tail_seconds`` 影响的是边界回贴，对 IoU 也有影响，但维度一多组合爆炸，
#: 先固定为工程当前值。
SEARCH_GRID: list[tuple[str, list[float]]] = [
    ("seg_prominence", [0.10, 0.15, 0.22, 0.30]),
    ("seg_min_core", [0.8, 1.2, 1.8, 2.5]),
    ("seg_min_rest", [0.6, 0.8, 1.2]),
    ("seg_min_quiet", [0.6, 0.8, 1.0]),
    ("min_rally_seconds", [2.0, 2.5, 3.0]),
]


# ------------------------------------------------------------------ 标注读写


def annotation_path(media: MediaInfo) -> Path:
    """标注文件按「代理视频名」命名（与旧的独立标注工具保持一致）。

    用代理名而不是原名：分析实际处理的是代理，标注也是对着代理画面标的；
    而且旧工具已经按代理名存过一批标注，沿用名字能直接读到。
    """
    for raw in (media.proxy_path, media.path):
        if raw:
            return ANNOTATIONS_DIR / f"{Path(raw).stem}.anno.json"
    return ANNOTATIONS_DIR / f"{media.id}.anno.json"


def load_annotation(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"rallies": [], "focus": None, "note": ""}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"rallies": [], "focus": None, "note": ""}
    if not isinstance(data, dict):
        return {"rallies": [], "focus": None, "note": ""}
    data.setdefault("rallies", [])
    return data


def normalize_rallies(raw: Iterable[dict]) -> list[dict[str, Any]]:
    """清洗标注回合：丢掉起止非法 / 倒置的项，按起点排序。"""
    out: list[dict[str, Any]] = []
    for r in raw or []:
        try:
            start = max(0.0, float(r["start"]))
            end = float(r["end"])
        except Exception:
            continue
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            continue
        out.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "note": str(r.get("note") or ""),
            "source": str(r.get("source") or "manual"),
        })
    out.sort(key=lambda x: x["start"])
    return out


def save_annotation(path: Path, media: MediaInfo, payload: dict, duration: float,
                    fps: float) -> dict[str, Any]:
    rallies = normalize_rallies(payload.get("rallies") or [])
    focus = payload.get("focus")
    doc = {
        "video": str(media.proxy_path or media.path),
        "video_name": Path(media.proxy_path or media.path).name,
        "duration": round(float(duration), 3),
        "fps": round(float(fps), 4),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(rallies),
        "focus": focus,
        "note": str(payload.get("note") or ""),
        "rallies": rallies,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return doc


def annotation_to_csv(doc: dict, fps: float) -> str:
    lines = ["index,start,end,duration,start_frame,end_frame,source,note"]
    for i, r in enumerate(normalize_rallies(doc.get("rallies") or []), 1):
        s, e = float(r["start"]), float(r["end"])
        sf = int(round(s * fps)) if fps else ""
        ef = int(round(e * fps)) if fps else ""
        note = str(r.get("note") or "").replace(",", " ")
        lines.append(f"{i},{s:.3f},{e:.3f},{e - s:.3f},{sf},{ef},{r.get('source','')},{note}")
    return "\n".join(lines)


# ------------------------------------------------------------------ 评估


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def match_pairs(preds: list[tuple[float, float]], gt: list[tuple[float, float]],
                thr: float = 0.5) -> list[tuple[int, int, float]]:
    """贪心 IoU 匹配（按 IoU 从高到低配对，一对一）。"""
    cand: list[tuple[float, int, int]] = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gt):
            v = iou(p, g)
            if v >= thr:
                cand.append((v, i, j))
    cand.sort(reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for v, i, j in cand:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        pairs.append((i, j, v))
    return pairs


def metrics(preds: list[tuple[float, float]], gt: list[tuple[float, float]],
            thr: float = 0.5) -> dict[str, float]:
    pairs = match_pairs(preds, gt, thr)
    tp = len(pairs)
    fp = len(preds) - tp
    fn = len(gt) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"iou": thr, "n": len(preds), "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


# ------------------------------------------------------------------ 搜参


@dataclass
class _Context:
    fused: RA.FusedSignal
    hits: Any
    player_motion: np.ndarray | None
    player_coverage: np.ndarray | None
    fps: float
    duration: float
    lo: float
    hi: float


def _build_context(res: AnalysisResult, lo: float, hi: float) -> _Context:
    sig = res.signals or {}
    act = np.asarray(sig.get("activity_full") or [], dtype=np.float32)
    if act.size == 0:
        raise ValueError("分析结果里没有 activity_full；请先用新版跑一次 AI 分析")
    fps = float((sig.get("fps") or [12.0])[0]) or 12.0
    duration = float((sig.get("duration") or [0.0])[0]) or (act.size / fps)
    med = float(np.median(act))
    fused = RA.FusedSignal(
        fps=fps, duration=duration, activity=act,
        threshold_hi=max(float(np.percentile(act, 78)), med * 1.25),
        threshold_lo=max(float(np.percentile(act, 55)) * 0.92, med * 1.05),
        audio_reliability=float(res.stats.get("audio_reliability", 1.0) or 1.0),
    )
    hits = P._rebuild_hits(sig)
    pfps = float((sig.get("player_fps") or [0.0])[0]) or 0.0
    pm_raw = sig.get("player_motion_full") or []
    cov_raw = sig.get("player_coverage_full") or []
    player_motion = np.asarray(pm_raw, dtype=np.float32) if pm_raw and pfps > 0 else None
    player_coverage = (np.asarray(cov_raw, dtype=np.float32)
                       if cov_raw and len(cov_raw) == len(pm_raw) else None)
    return _Context(fused=fused, hits=hits, player_motion=player_motion,
                    player_coverage=player_coverage, fps=fps, duration=duration,
                    lo=lo, hi=hi)


def _predict(ctx: _Context, params: AnalysisParams) -> list[tuple[float, float]]:
    """用一组参数跑一次切分 + 收尾，返回落在标注窗口内的回合。"""
    opt = RV.SegmentOptions(
        min_rally=float(params.min_rally_seconds),
        max_rally=float(params.max_rally_seconds),
        pre_roll=float(params.pre_roll), post_roll=float(params.post_roll),
        min_quiet=float(params.seg_min_quiet),
        prominence_ratio=float(params.seg_prominence),
        min_rest=float(params.seg_min_rest),
        min_core=float(params.seg_min_core),
    )
    intervals, trace = P._segment_rallies(
        ctx.fused, params, ctx.duration,
        player_motion=ctx.player_motion, player_coverage=ctx.player_coverage,
        hits=ctx.hits, opt_override=opt)
    method = str(trace.get("method") or "")
    if ctx.hits is not None and getattr(ctx.hits, "times", np.zeros(0)).size:
        intervals = RA.refine_with_hits(
            intervals, ctx.hits, pre_roll=params.pre_roll, post_roll=params.post_roll,
            tail_seconds=params.hit_tail_seconds,
            trim_start=method != "player_motion")
    intervals = RA.dedupe_overlaps(intervals, hits=ctx.hits, fps=ctx.fps,
                                   activity=ctx.fused.activity)
    intervals = [iv for iv in intervals if iv.end - iv.start >= params.min_rally_seconds]
    intervals.sort(key=lambda v: v.start)
    intervals = P._join_abutting(intervals)
    return [(iv.start, iv.end) for iv in intervals
            if iv.end > ctx.lo and iv.start < ctx.hi]


def optimize(res: AnalysisResult, gt: list[tuple[float, float]],
             focus: tuple[float, float] | None = None,
             grid: list[tuple[str, list[float]]] | None = None,
             iou_thr: float = 0.5) -> dict[str, Any]:
    """在参数网格上搜 F1 最高的一组切分参数。

    Returns 一个字典：``baseline``（当前参数的指标）、``best``（最优参数）与
    ``results``（每个组合的指标，按 F1 降序）。注意这**只是搜索**；要应用到
    工程还需要把 ``best`` 写回 params 并 resegment。
    """
    gt = [(float(a), float(b)) for a, b in gt if b > a]
    if not gt:
        raise ValueError("标注为空，无法优化")
    lo = float(focus[0]) if focus else min(a for a, _ in gt)
    hi = float(focus[1]) if focus else max(b for _, b in gt)
    ctx = _build_context(res, lo, hi)
    base = res.params
    grid = grid or SEARCH_GRID

    names = [n for n, _ in grid]
    out: list[dict[str, Any]] = []
    for combo in itertools.product(*[vals for _, vals in grid]):
        patch = dict(zip(names, combo))
        params = base.model_copy(update=patch)
        try:
            preds = _predict(ctx, params)
        except Exception as e:  # noqa: BLE001
            continue
        m = metrics(preds, gt, iou_thr)
        m["params"] = {k: float(v) for k, v in patch.items()}
        out.append(m)

    baseline_preds = _predict(ctx, base)
    baseline = metrics(baseline_preds, gt, iou_thr)
    baseline["params"] = {n: float(getattr(base, n)) for n in names}

    # F1 相同再比召回 / 精度，避免选出一组「只是切得少」的参数
    out.sort(key=lambda m: (m["f1"], m["recall"], m["precision"]), reverse=True)
    return {
        "gt_count": len(gt),
        "focus": [lo, hi],
        "iou_threshold": iou_thr,
        "baseline": baseline,
        "best": out[0] if out else None,
        "results": out[:40],
        "tried": len(out),
        "search_fields": names,
    }


# ------------------------------------------------------------------ 建议（默认值）

#: 从标注里能算出来的、不依赖搜参的客观建议，用来解释「为什么这些参数更合适」。
def suggest(gt: list[tuple[float, float]]) -> dict[str, float]:
    gt = [(float(a), float(b)) for a, b in gt if b > a]
    if not gt:
        return {}
    durs = np.asarray([b - a for a, b in gt], dtype=np.float64)
    gaps = np.asarray([gt[i + 1][0] - gt[i][1] for i in range(len(gt) - 1)], dtype=np.float64)
    gaps = gaps[gaps > 0]
    out = {
        "count": float(len(gt)),
        "duration_min": round(float(durs.min()), 2),
        "duration_median": round(float(np.median(durs)), 2),
        "duration_max": round(float(durs.max()), 2),
    }
    if gaps.size:
        out["gap_median"] = round(float(np.median(gaps)), 2)
        out["gap_min"] = round(float(gaps.min()), 2)
    return out
