"""Rally quality scoring.

Scoring means "mapping objective features to an interpretable 0~100 score", not a black box.
Each sub-score is given by a readable formula, and the weights can be switched via presets in the UI.

Sub-scores
----------
- **length**: number of shots, duration.
- **intensity**: player movement speed, peak frame motion, late-rally tempo.
- **technique**: shuttle speed, hit force, whether there is a jump.
- **excitement**: combination of length + intensity + late-rally surge + number of smashes.
- **production**: sharpness, shake, whether the subject is large enough (only clear enough footage is worth using).
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
    #: Saturation threshold for the length score (shot count)
    shot_saturate: float = 24.0
    #: Saturation threshold for duration (seconds)
    dur_saturate: float = 22.0
    #: Weight of the new "highlight" sub-score (smash / confrontation / late-rally variance).
    #: 0 disables it (old behavior); positive values blend it into the total.
    highlight: float = 0.0


PRESETS: dict[str, ScoreWeights] = {
    "balanced": ScoreWeights("balanced", "balanced"),
    "highlight": ScoreWeights(
        "highlight", "highlight",
        length=0.16, intensity=0.34, technique=0.26, excitement=0.20, production=0.04,
        shot_saturate=18.0, dur_saturate=18.0,
    ),
    "highlight_pro": ScoreWeights(
        "highlight_pro", "highlight_pro",
        length=0.10, intensity=0.22, technique=0.22, excitement=0.24, production=0.02,
        highlight=0.20,
        shot_saturate=14.0, dur_saturate=14.0,
    ),
    "long_rally": ScoreWeights(
        "long_rally", "long_rally",
        length=0.42, intensity=0.20, technique=0.14, excitement=0.18, production=0.06,
        shot_saturate=40.0, dur_saturate=35.0,
    ),
    "technique": ScoreWeights(
        "technique", "technique",
        length=0.14, intensity=0.18, technique=0.42, excitement=0.18, production=0.08,
        shot_saturate=20.0, dur_saturate=20.0,
    ),
    "training": ScoreWeights(
        "training", "training",
        length=0.30, intensity=0.26, technique=0.18, excitement=0.10, production=0.16,
        shot_saturate=30.0, dur_saturate=30.0,
    ),
}

#: Canonical rally tag codes. Stored in analysis JSON and used by the frontend for
#: filtering; display names are translated in the UI (``tag.<code>``).
TAG_ULTRA_LONG = "ultra_long_rally"
TAG_MANY_SHOTS = "many_shots"
TAG_FAST_TEMPO = "fast_tempo"
TAG_LATE_ACCEL = "late_acceleration"
TAG_HIGH_MOBILITY = "high_mobility"
TAG_FAST_SHUTTLE = "fast_shuttle"
TAG_LONG_RALLY = "long_rally"
TAG_SHORT_RALLY = "short_rally"
TAG_HIGH_SCORE = "high_score"
TAG_LOW_CONFIDENCE = "low_confidence"
TAG_HIGHLIGHT = "highlight"
TAG_SMASH = "smash"
TAG_CONFRONTATION = "confrontation"

#: Legacy Chinese tag values (from older project files) -> canonical codes.
TAG_LEGACY_MAP: dict[str, str] = {
    "超长多拍": TAG_ULTRA_LONG,
    "多拍": TAG_MANY_SHOTS,
    "快节奏": TAG_FAST_TEMPO,
    "末段提速": TAG_LATE_ACCEL,
    "高强度跑动": TAG_HIGH_MOBILITY,
    "高速球": TAG_FAST_SHUTTLE,
    "长回合": TAG_LONG_RALLY,
    "短回合": TAG_SHORT_RALLY,
    "高分": TAG_HIGH_SCORE,
    "低置信": TAG_LOW_CONFIDENCE,
}


def migrate_tags(tags: list[str] | None) -> list[str]:
    """Map legacy Chinese tags to canonical codes, preserving order and uniqueness."""
    out: list[str] = []
    for t in tags or []:
        code = TAG_LEGACY_MAP.get(t, t)
        if code and code not in out:
            out.append(code)
    return out

#: The relative position of each feature within "all rallies" is converted into a percentile score,
#: so rallies from the same batch of footage can be compared with each other (relative scoring) instead
#: of being pinned by absolute thresholds.
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
    """Map a set of values to 0~1 percentiles (ties are averaged)."""
    n = values.size
    if n == 0:
        return values
    if n == 1:
        return np.full(1, 0.5, dtype=np.float32)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.arange(n, dtype=np.float32)
    # Average over ties
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
    """Score a batch of rallies.

    Parameters
    ----------
    features : feature dict for each rally (from :func:`rally.attach_features`)
    weights  : scoring weight preset
    quality  : optional frame-quality features for each rally (sharpness / shake / subject_size)

    Returns
    -------
    A list of score dicts as long as the input, with keys ``total/length/intensity/technique/excitement/production``
    plus the intermediate quantities needed for ``tags``.
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

    # --- Length score: shot count and duration, using a saturation curve (not linear, so over-long rallies do not dominate the top)
    len_shots = 1.0 - np.exp(-shots / max(w.shot_saturate * 0.55, 1.0))
    len_dur = 1.0 - np.exp(-dur / max(w.dur_saturate * 0.55, 1.0))
    length = 100.0 * np.clip(0.62 * len_shots + 0.38 * len_dur, 0, 1)

    # --- Intensity score: player speed is primary, frame motion and late-rally tempo are secondary
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

    # --- Technique score: shuttle speed, hit force, continuity of shuttle presence
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

    # --- Excitement: long + fierce + late-rally surge + smashes
    length_surge = np.clip((shots - 6.0) / 14.0, 0, 1)
    finish_surge = np.clip((finish_tempo - np.percentile(finish_tempo, 50)) /
                           (np.percentile(finish_tempo, 90) - np.percentile(finish_tempo, 50) + EPS), 0, 1)
    excitement = 100.0 * np.clip(
        0.30 * (length / 100.0) + 0.30 * (intensity / 100.0) +
        0.20 * finish_surge + 0.20 * length_surge, 0, 1
    )

    # --- Highlight sub-score: smash count, confrontation streak, late-rally variance
    smash = col("smash_proxy")
    streak = col("confrontation_streak", 1.0)
    tvar = col("tempo_variance")
    smash_score = np.clip(smash / max(np.percentile(smash, 90), 1.0), 0, 1)
    streak_score = np.clip(streak / max(np.percentile(streak, 90), 1.0), 0, 1)
    tvar_score = np.clip(tvar / max(np.percentile(tvar, 90), 1.0), 0, 1)
    highlight = 100.0 * np.clip(
        0.40 * smash_score + 0.35 * streak_score + 0.25 * tvar_score, 0, 1
    )

    # --- Frame quality
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

    # --- Analysis confidence used as a confidence discount on the total score
    conf_adj = 0.75 + 0.25 * np.clip(conf, 0, 1)

    total = (
        w.length * length + w.intensity * intensity + w.technique * technique +
        w.excitement * excitement + w.production * production + w.highlight * highlight
    ) * conf_adj

    # --- Voice command bonus: rallies that hit shouts like "good shot" get a fixed extra score.
    # This is an **absolute bonus** (not sub-score weighting), added after the confidence discount, with the total capped at 100.
    # When it is 0 the result is exactly the same as with the feature disabled.
    total = total + col("speech_bonus")

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
            "highlight": round(float(np.clip(highlight[i], 0, 100)), 1),
        })
        out[-1]["tags"] = tags  # type: ignore[assignment]
    return out


def _tags(f: dict[str, float], dur: float, shots: float, tempo: float, finish_tempo: float,
          pspd_max: float, mpeak: float, total: float) -> list[str]:
    tags: list[str] = []
    if shots >= 20:
        tags.append(TAG_ULTRA_LONG)
    elif shots >= 12:
        tags.append(TAG_MANY_SHOTS)
    if tempo >= 1.5:
        tags.append(TAG_FAST_TEMPO)
    if finish_tempo >= 1.8 and shots >= 6:
        tags.append(TAG_LATE_ACCEL)
    if mpeak >= 0.72:
        tags.append(TAG_HIGH_MOBILITY)
    if f.get("shuttle_speed_p90", 0) > 0 and f.get("shuttle_speed_p90", 0) >= 0.55:
        tags.append(TAG_FAST_SHUTTLE)
    if dur >= 25:
        tags.append(TAG_LONG_RALLY)
    elif dur <= 5.5:
        tags.append(TAG_SHORT_RALLY)
    if total >= 75:
        tags.append(TAG_HIGH_SCORE)
    if f.get("confidence", 1.0) < 0.35:
        tags.append(TAG_LOW_CONFIDENCE)
    if f.get("smash_proxy", 0) >= 2:
        tags.append(TAG_SMASH)
    if f.get("confrontation_streak", 1) >= 6:
        tags.append(TAG_CONFRONTATION)
    if total >= 70 and (f.get("smash_proxy", 0) >= 1 or f.get("confrontation_streak", 1) >= 4):
        tags.append(TAG_HIGHLIGHT)
    # Matched phrases are used directly as tags (e.g. "good shot"), so users can see at a glance where the bonus comes from
    for p in f.get("speech_phrases") or []:
        if isinstance(p, str) and p and p not in tags:
            tags.append(p)
    return tags


def derive_stats(features: list[dict[str, float]], scores: list[dict[str, float]]) -> dict[str, Any]:
    """Generate summary statistics for the whole analysis result."""
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
    """Derive a total-score threshold from a "keep ratio", for operations like "keep only the top 40% rallies"."""
    if not scores:
        return 0.0
    tot = np.array([s["total"] for s in scores], dtype=np.float32)
    if keep_ratio >= 1.0:
        return float(tot.min())
    q = 100.0 * (1.0 - keep_ratio)
    return float(np.percentile(tot, q))
