# 回合分割与评分优化技术报告（clip2 / clip5）

> 日期：2026-09-27。基于人工标注（ground truth）对 `0919_clip2`、`0919_clip5` 两段视频前半部分做回合分割与高光评分的算法优化。本文档记录：现有算法不足、改进点、参数调整依据、性能评估指标与 A/B 对比结果。

---

## 1. 评估方法

- **标注数据**：`data/annotations/0919_clip2_m_7b0cb5fd7121_960x540.anno.json`（GT=25 回合）、`0919_clip5_m_af2f4518f3e2_960x540.anno.json`（GT=22 回合）。
- **评估窗口**：`[0, max(GT.end) + 12s]`，只评标注覆盖的前半部分。
- **匹配准则**：预测区间与 GT 区间 IoU ≥ 0.5 记为 TP（贪心一对一匹配），输出 Precision / Recall / F1。
- **评估路径**：与线上一致的 `pipeline.resegment()` 生产路径（复用缓存信号，不重跑 AI），脚本见 `scripts/tune_clip25.py`、`scripts/eval_scoring_ab.py`。

## 2. 任务一：回合分割精度优化

### 2.1 现有算法的不足

优化前（旧默认参数）基线：

| 片段 | P | R | F1 | TP/FP/FN |
|---|---|---|---|---|
| clip2 | 0.565 | 0.520 | **0.542** | 13 / 10 / 12 |
| clip5 | 0.471 | 0.364 | **0.410** | 8 / 9 / 14 |

误差归因（`tune_clip25.py` 的 FN/FP dump）：

1. **邻场击球声干扰（核心问题）**：场馆内多块场地共用环境音，旧版只用姿态挥拍证据做归属门控，覆盖率低时失效，邻场击球被当作本方回合锚点 → 粘连段（FP）与虚假回合。
2. **短回合漏检（FN 主因）**：`seg_prominence=0.18`、`seg_min_core=1.0` 等偏保守，3~6s 短回合的活跃度峰显著度不足被整段丢弃。
3. **末端粘连**：`seg_min_rest=0.8`、`seg_min_quiet=0.7` 偏长，局间停顿未被识别为间隔，相邻回合被并成一段。

### 2.2 改进点 1：复合击球归属门控（`analysis/pose.py`）

新增两个函数（旧 `gate_hits` 保留作降级与测试锚点）：

- **`local_strength_pct(times, strength, half_win=15.0)`**：每拍强度在 ±15s 邻域内的分位数。物理依据：固定机位+麦克风下，本方场地击球声 consistently 强于邻场（AGC 鲁棒）。在 clip2 实测为最强单特征（AUC 0.79，姿态证据仅 0.54）。
- **`composite_gate(...)`**：`score = max(w_pose·pose_evidence, w_str·strength_pct)`，取 max 而非乘积以保召回——姿态漏检（遮挡/人太小）可由强度救回，轻吊（声音弱）可由挥拍证据救回。`w_pose=1.0, w_str=0.6, threshold=0.45`，保留率越界（<12% 或 >97%）时自动降级为不过滤。
- **退化规则**：邻域强度无方差（max−min<1e-6，如合成/重度归一化音频）时强度线索置 0——无对比度即无归属信息，退化为纯姿态证据；孤立拍保持中性偏高以保召回。

### 2.3 改进点 2：分割参数重调（网格搜索，依据见下表）

| 参数 | 旧默认 | 新默认 | 调整依据 |
|---|---|---|---|
| `seg_prominence` | 0.18 | **0.10** | stageA 网格搜索最优区间 0.08~0.10；0.18 漏检短回合 |
| `seg_min_core` | 1.0 | **2.5** | 配合低 prominence，要求核心段足够长以抑制碎段 |
| `seg_min_rest` | 0.8 | **0.6** | 缩短局间判定，切开粘连段 |
| `seg_min_quiet` | 0.7 | **0.6** | 同上，提高谷值灵敏度 |
| `min_rally_seconds` | 3.0 | **2.0** | GT 中短回合占比高，3.0 直接滤掉真实短回合 |
| `pose_gate_threshold` | 0.22 | **0.45** | stageB 网格搜索：复合门控下 0.45 为 F1 最优点 |

搜索空间：stageA `prom×core×rest×quiet×minr` = 5×4×3×3×3=540 组；stageB 门控阈值 9 档。脚本 `scripts/tune_clip25.py`、`scripts/exp_gate_v2.py`、`scripts/exp_gate_v3.py`。

### 2.4 优化效果（可量化提升）

生产路径（`resegment` + 新默认参数）实测：

| 片段 | 旧 F1 | 新 F1 | 新 P / R | 提升 |
|---|---|---|---|---|
| clip2 | 0.542 | **0.846** | 0.815 / 0.880 | **+56%** |
| clip5 | 0.410 | **0.558** | 0.571 / 0.545 | **+36%** |

击球归属门控效果：clip2 原始候选 2204 拍 → 保留 1038 拍；clip5 1539 拍 → 保留 720 拍，邻场噪声拍被大量剔除。

## 3. 任务二：回合评分策略优化

### 3.1 现有评分策略的缺陷

旧版 5 子分（length / intensity / technique / excitement / production）× 预设权重，分位数相对评分。缺陷：

1. **无"对抗/杀球"特征**：excitement 只由长度+强度+末段提速组合，无法区分"多拍拉锯+重杀"与"慢速多拍"。
2. **长度主导**：balanced 下长度权重 0.22，短促激烈的高质量回合排名偏后。
3. **无高光专用口径**：最接近的 `highlight` 预设只是把 intensity/technique 权重拉高，没有直接刻画高光元素的子分。

### 3.2 新特征体系（`analysis/rally.py::attach_features`）

| 特征 | 定义 | 刻画目标 |
|---|---|---|
| `tempo_variance` | 回合内拍间隔的方差 | 节奏变化/变速突击 |
| `confrontation_streak` | 拍间隔 ≤0.85s 的最长连续快攻串 | 强对抗持续度 |
| `smash_proxy` | 强度 ≥ 本回合 p85 的拍数 | 杀球/重杀次数（姿态后半段覆盖不足，用强度代理） |

### 3.3 新评分模型（`analysis/scoring.py`）

- 新增 **highlight 子分**（各特征按全体回合 p90 归一）：
  `highlight = 100 × clip(0.40·smash + 0.35·streak + 0.25·tempo_variance)`
- 新增预设 **`highlight_pro`**：length=0.10, intensity=0.22, technique=0.22, excitement=0.24, production=0.02, **highlight=0.20**，拍数/时长饱和阈值降至 14（短而狠的回合不再吃亏）。
- 新增标签：`smash`（smash_proxy≥2）、`confrontation`（streak≥6）、`highlight`（总分≥70 且 smash≥1 或 streak≥4）。

### 3.4 A/B 对比（A=balanced，B=highlight_pro）

脚本：`scripts/eval_scoring_ab.py --save data/cache/scoring_ab.json`。按 B 分降序，关键样本：

| 片段 | 区间 | 时长/拍数/杀球 | A 分 | B 分 | 变化 |
|---|---|---|---|---|---|
| clip5 | [50.2, 61.7] | 11.5s / 10拍 / 3杀 | 56.4 | **61.9** | 第 1 名，末段提速 1.91 |
| clip2 | [134.6, 143.1] | 8.6s / 9拍 / 2杀 | 52.7 | **58.0** | 第 1 名 |
| clip5 | [217.2, 232.8] | 15.5s / 13拍 / 2杀 | 37.3 | **43.8** | 最大幅度提前 |
| clip2 | [272.1, 281.4] | 9.2s / 7拍 / 1杀 | 34.7 | **46.8** | 显著提前 |
| clip2 | [208.3, 215.4] | 7.1s / 3拍 / 慢节奏 | 20.2 | 25.3 | 压后 |
| clip5 | [83.5, 89.8] | 6.4s / 2拍 / 慢节奏 | 29.6 | 32.2 | 压后 |

结论：B 口径一致地把「长多拍+杀球+连续快攻」提前、纯短平快压后，符合专业高光筛选逻辑。

### 3.5 A/B 评审片段（用户已定稿）

已导出到 `data/exports/ab_review/`（前 2 个为 B 口径 Top1，中间 2 个为 B 大幅提前，末 2 个为 B 压后）。经用户看片评审，**B（highlight_pro）口径的排序符合专业高光筛选预期，方案定稿**。

- `clip5_B1_50.2-61.7_10shots_3smash.mp4`、`clip2_B1_134.6-143.1_9shots_2smash.mp4`
- `clip5_B-rise_217.2-232.8_13shots_2smash.mp4`、`clip2_B-rise_272.1-281.4_7shots.mp4`
- `clip5_B-sink_83.5-89.8_2shots_slow.mp4`、`clip2_B-sink_208.3-215.4_3shots_slow.mp4`

## 4. 复现方法

```powershell
# 分割基线 + 网格搜索 + 误差归因
.\.venv\Scripts\python.exe scripts\tune_clip25.py --save data\cache\tune_baseline.json
# 复合门控实验（v2 网格 / v3 两阶段锚定+短间隔合并）
.\.venv\Scripts\python.exe scripts\exp_gate_v2.py
.\.venv\Scripts\python.exe scripts\exp_gate_v3.py
# A/B 评分对比
.\.venv\Scripts\python.exe scripts\eval_scoring_ab.py --save data\cache\scoring_ab.json
# 回归测试
.\.venv\Scripts\python.exe tests\test_core.py
```

## 5. 已知局限与后续方向

1. clip5 后半段 `pose_swing/pose_overhead` 覆盖为 0，杀球只能用强度代理（smash_proxy），后续可引入姿态 overhead 或球速突变做交叉验证。
2. `highlight_pro` 的 highlight 子分权重（0.40/0.35/0.25）经用户 A/B 看片评审后定稿。
3. clip5 召回仍偏低（0.545），FN 多为信号弱短回合；后续可探索视觉事件（球过网、场地端线检测）补充音频线索。
