"""回合质量评分。

评分是「把客观特征映射到 0~100 的可解释分数」，不是黑箱。
每个分项都由可读的公式给出，权重可通过界面上的预设切换。

分项
----
- **length 长度**：多拍、时长。
- **intensity 强度**：球员移动速度、画面运动峰值、末段节奏。
- **technique 技术含量**：球速、击球力度、是否有起跳。
- **excitement 精彩度**：长度 + 强度 + 末段提速 + 杀球数量的综合。
- **production 画面质量**：清晰度、抖动、主体是否够大（够清楚才值得用）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

EPS = 1e-9


@dataclass
class ScoreWeights:
    name: str = "balanced"
    label: str = "均衡"
    length: float = 0.22
    intensity: float = 0.30
    technique: float = 0.20
    excitement: float = 0.20
    production: float = 0.08
    #: 长度分饱和阈值（拍数）
    shot_saturate: float = 24.0
    #: 时长饱和阈值（秒）
    dur_saturate: float = 22.0


PRESETS: dict[str, ScoreWeights] = {
    "balanced": ScoreWeights("balanced", "均衡"),
    "highlight": ScoreWeights(
        "highlight", "精彩集锦",
        length=0.16, intensity=0.34, technique=0.26, excitement=0.20, production=0.04,
        shot_saturate=18.0, dur_saturate=18.0,
    ),
    "long_rally": ScoreWeights(
        "long_rally", "多拍回合",
        length=0.42, intensity=0.20, technique=0.14, excitement=0.18, production=0.06,
        shot_saturate=40.0, dur_saturate=35.0,
    ),
    "technique": ScoreWeights(
        "technique", "技术动作",
        length=0.14, intensity=0.18, technique=0.42, excitement=0.18, production=0.08,
        shot_saturate=20.0, dur_saturate=20.0,
    ),
    "training": ScoreWeights(
        "training", "训练复盘",
        length=0.30, intensity=0.26, technique=0.18, excitement=0.10, production=0.16,
        shot_saturate=30.0, dur_saturate=30.0,
    ),
}

#: 各特征在「全体回合」中的相对位置被换算成分位数分数，
#: 这样同一批素材里的回合能互相比较（相对评分），而不是被绝对阈值卡死。
_RELATIVE_FEATURES = {
    "duration": 1.0,
    "shot_count": 1.0,
    "tempo": 1.0,
    "finish_tempo": 1.0,
    "activity_mean": 1.0,
    "activity_peak": 1.0,
    "player_speed_mean": 1.0,
    "player_speed_max": 1.0,
    "motion_mean": 1.0,
    "motion_peak": 1.0,
    "hit_strength_p90": 1.0,
    "shuttle_speed_p90": 1.0,
    "shuttle_presence": 1.0,
    "rally_span": 1.0,
}


def _pct_rank(values: np.ndarray) -> np.ndarray:
    """把一组数值映射到 0~1 的百分位（并列取平均）。"""
    n = values.size
    if n == 0:
        return values
    if n == 1:
        return np.full(1, 0.5, dtype=np.float32)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.arange(n, dtype=np.float32)
    # 并列平均
    s = values[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks / max(1, n - 1)


def score_rallies(
    features: list[dict[str, float]],
    weights: ScoreWeights | None = None,
    quality: list[dict[str, float]] | None = None,
) -> list[dict[str, float]]:
    """对一批回合打分。

    参数
    ----
    features : 每个回合的特征字典（来自 :func:`rally.attach_features`）
    weights  : 评分权重预设
    quality  : 每个回合的画面质量特征（sharpness / shake / subject_size），可选

    返回
    ----
    与输入等长的评分字典列表，键为 ``total/length/intensity/technique/excitement/production``
    以及 ``tags`` 所需的中间量。
    """
    w = weights or PRESETS["balanced"]
    n = len(features)
    if n == 0:
        return []

    def col(key: str, default: float = 0.0) -> np.ndarray:
        return np.array([float(f.get(key, default) or 0.0) for f in features], dtype=np.float32)

    dur = col("duration")
    shots = col("shot_count")
    tempo = col("tempo")
    finish_tempo = col("finish_tempo")
    act_mean = col("activity_mean")
    act_peak = col("activity_peak")
    pspd_mean = col("player_speed_mean")
    pspd_max = col("player_speed_max")
    mmean = col("motion_mean")
    mpeak = col("motion_peak")
    hstr = col("hit_strength_p90")
    sspd = col("shuttle_speed_p90")
    spres = col("shuttle_presence")
    conf = col("confidence", 0.5)

    # --- 长度分：拍数与时长，用饱和曲线（不是线性，避免超长回合独占榜首）
    len_shots = 1.0 - np.exp(-shots / max(w.shot_saturate * 0.55, 1.0))
    len_dur = 1.0 - np.exp(-dur / max(w.dur_saturate * 0.55, 1.0))
    length = 100.0 * np.clip(0.62 * len_shots + 0.38 * len_dur, 0, 1)

    # --- 强度分：球员速度为主，画面运动与末段节奏为辅
    spd_ref = np.percentile(pspd_mean, 85) + EPS
    spd_score = np.clip(pspd_mean / spd_ref, 0, 1.2) / 1.2
    mpeak_score = np.clip(mpeak / (np.percentile(mpeak, 85) + EPS), 0, 1.2) / 1.2
    tem_ref = np.percentile(tempo, 85) + EPS
    tem_score = np.clip(tempo / tem_ref, 0, 1.2) / 1.2
    ft_ref = np.percentile(finish_tempo, 85) + EPS
    ft_score = np.clip(finish_tempo / ft_ref, 0, 1.3) / 1.3
    intensity = 100.0 * np.clip(
        0.38 * spd_score + 0.22 * mpeak_score + 0.18 * tem_score + 0.22 * ft_score, 0, 1
    )

    # --- 技术分：球速、击球力度、出现羽毛球的持续性
    s_ref = np.percentile(sspd, 85) + EPS
    s_score = np.clip(sspd / s_ref, 0, 1.2) / 1.2
    h_ref = np.percentile(hstr, 85) + EPS
    h_score = np.clip(hstr / h_ref, 0, 1.2) / 1.2
    p_ref = np.percentile(spres, 85) + EPS
    p_score = np.clip(spres / p_ref, 0, 1.2) / 1.2
    has_shuttle = float(np.any(sspd > 0))
    if has_shuttle:
        technique = 100.0 * np.clip(0.42 * s_score + 0.34 * h_score + 0.24 * p_score, 0, 1)
    else:
        technique = 100.0 * np.clip(0.55 * h_score + 0.45 * p_score, 0, 1)

    # --- 精彩度：长 + 猛 + 末段提速 + 杀球
    length_surge = np.clip((shots - 6.0) / 14.0, 0, 1)
    finish_surge = np.clip((finish_tempo - np.percentile(finish_tempo, 50)) /
                           (np.percentile(finish_tempo, 90) - np.percentile(finish_tempo, 50) + EPS), 0, 1)
    excitement = 100.0 * np.clip(
        0.30 * (length / 100.0) + 0.30 * (intensity / 100.0) +
        0.20 * finish_surge + 0.20 * length_surge, 0, 1
    )

    # --- 画面质量
    if quality and len(quality) == n:
        sharp = np.array([float(q.get("sharpness", 0.5)) for q in quality], dtype=np.float32)
        shake = np.array([float(q.get("shake", 0.5)) for q in quality], dtype=np.float32)
        size = np.array([float(q.get("subject_size", 0.3)) for q in quality], dtype=np.float32)
        production = 100.0 * np.clip(
            0.45 * np.clip(sharp, 0, 1) + 0.25 * (1.0 - np.clip(shake, 0, 1)) +
            0.30 * np.clip(size / 0.35, 0, 1), 0, 1
        )
    else:
        production = np.full(n, 70.0, dtype=np.float32)

    # --- 分析置信度作为总分的置信折扣
    conf_adj = 0.75 + 0.25 * np.clip(conf, 0, 1)

    total = (
        w.length * length + w.intensity * intensity + w.technique * technique +
        w.excitement * excitement + w.production * production
    ) * conf_adj

    out: list[dict[str, float]] = []
    for i in range(n):
        tags = _tags(features[i], dur[i], shots[i], tempo[i], finish_tempo[i],
                     pspd_max[i], mpeak[i], total[i])
        out.append({
            "total": round(float(np.clip(total[i], 0, 100)), 1),
            "length": round(float(np.clip(length[i], 0, 100)), 1),
            "intensity": round(float(np.clip(intensity[i], 0, 100)), 1),
            "technique": round(float(np.clip(technique[i], 0, 100)), 1),
            "excitement": round(float(np.clip(excitement[i], 0, 100)), 1),
            "production": round(float(np.clip(production[i], 0, 100)), 1),
        })
        out[-1]["tags"] = tags  # type: ignore[assignment]
    return out


def _tags(f: dict[str, float], dur: float, shots: float, tempo: float, finish_tempo: float,
          pspd_max: float, mpeak: float, total: float) -> list[str]:
    tags: list[str] = []
    if shots >= 20:
        tags.append("超长多拍")
    elif shots >= 12:
        tags.append("多拍")
    if tempo >= 1.5:
        tags.append("快节奏")
    if finish_tempo >= 1.8 and shots >= 6:
        tags.append("末段提速")
    if mpeak >= 0.72:
        tags.append("高强度跑动")
    if f.get("shuttle_speed_p90", 0) > 0 and f.get("shuttle_speed_p90", 0) >= 0.55:
        tags.append("高速球")
    if dur >= 25:
        tags.append("长回合")
    elif dur <= 5.5:
        tags.append("短回合")
    if total >= 75:
        tags.append("高分")
    if f.get("confidence", 1.0) < 0.35:
        tags.append("低置信")
    return tags


def derive_stats(features: list[dict[str, float]], scores: list[dict[str, float]]) -> dict[str, Any]:
    """给整个分析结果生成汇总统计。"""
    if not features:
        return {"count": 0}
    dur = np.array([f.get("duration", 0.0) for f in features], dtype=np.float32)
    shots = np.array([f.get("shot_count", 0.0) for f in features], dtype=np.float32)
    tot = np.array([s["total"] for s in scores], dtype=np.float32)
    active = float(dur.sum())
    return {
        "count": len(features),
        "total_duration": float(dur.sum()),
        "active_duration": active,
        "avg_duration": float(dur.mean()),
        "median_duration": float(np.median(dur)),
        "max_duration": float(dur.max()),
        "avg_shots": float(shots.mean()) if shots.size else 0.0,
        "max_shots": int(shots.max()) if shots.size else 0,
        "total_shots": int(shots.sum()) if shots.size else 0,
        "avg_score": float(tot.mean()) if tot.size else 0.0,
        "score_p90": float(np.percentile(tot, 90)) if tot.size else 0.0,
        "high_score_count": int((tot >= 70).sum()) if tot.size else 0,
        "hist_duration": np.histogram(dur, bins=10, range=(0, max(30.0, float(dur.max()) if dur.size else 30)))[0].tolist(),
        "hist_score": np.histogram(tot, bins=10, range=(0, 100))[0].tolist(),
    }


def auto_threshold(scores: list[dict[str, float]], keep_ratio: float = 0.4) -> float:
    """按「保留比例」反推一个总分阈值，供「只保留前 40% 精彩回合」这类操作。"""
    if not scores:
        return 0.0
    tot = np.array([s["total"] for s in scores], dtype=np.float32)
    if keep_ratio >= 1.0:
        return float(tot.min())
    q = 100.0 * (1.0 - keep_ratio)
    return float(np.percentile(tot, q))
