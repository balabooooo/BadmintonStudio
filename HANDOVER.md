# 交接文档 · BadmintonStudio 回合终点优化

> 交接日期：2026-09-14
> 交接原因：**本次会话在一次 PowerShell 清理命令中损坏了 `backend/bms/analysis/pipeline.py`**，
> 需要另一位 agent 接手完成恢复，并继续功能优化。
> 请先读完第 1、2、3 节再动手。

---

## ✅ 恢复完成（2026-09-18 接手 agent）

`pipeline.py` **已于本次会话重建完成并验证通过**：

* `backend/bms/analysis/pipeline.py` 现为 1452 行，`ast.parse` / `import` 均通过；
* 用「损坏前逐字快照 + `.pyc` 字符串常量 + 损坏文件 ASCII 骨架」重建；
  未受最终回退影响的全部函数，编译出的**字节码与损坏前 `.pyc` 逐字节一致**
  （`co_code` / `co_names` / `co_varnames` / `co_consts`；仅行号表因重排不同）；
* 移除了实验性 `hit_runs`/`extra` 的残留引用（损坏文件里 `_merge_by_availability`
  仍引用未定义的 `extra`，会 `NameError`）；
* `tests/test_core.py` 全部通过（158 项断言，0 失败）；
* `rerun_analysis.py --reuse` 精确复现第 7.1 节第 ④ 列：**51 回合 / 均值 8.76s /
  中位 7.19s / 最长 20.48s / >30s 为 0 / 球员覆盖率 0.998 / 质量 0.914 vs 0.887**；
* 完整 GPU 端到端重跑成功（396.9s，57 回合，最长 22.75s，无 >30s，姿态层命中缓存）；
* 已 `git init` 并完成首次提交（commit `ddf6aa1`）。

> 第 3 节的「事故详情」保留作历史记录；其中对**恢复方法的说明已被本节的实测结果取代**。
> 特别注意：`.pyc` 编译于最终回退**之前**（仍含 `extra`/`hit_times`/`hit_runs`），
> 因此它并非最终版源码的等价物，只用于校验未改动的函数。
> 仍然遗留（需要人工标注才能推进）的是第 8.4 / 8.5 / 8.6 节的精度问题。

---

## 1. 一分钟摘要

| 项 | 状态 |
|---|---|
| **原任务** | 修复「每一回合的结束时刻找不准」——回合末尾常包含球落地后的长尾，有时还含下一个回合 |
| **功能改动** | ✅ 已完成并实测有效：4 项逻辑修复 + 新增姿态检测层。最长回合 **69.35s → 20.48s**，>30s 的回合 **2 → 0** |
| **代码状态** | ✅ `pipeline.py` 已重建并验证（见顶部「恢复完成」节）；其余 57 个文件完好 |
| **头号任务** | ✅ 已完成：恢复 `pipeline.py`（字节码校验 + 测试 + 离线/完整重跑） |
| **次要任务** | 一部分区段（15~28s、51~54s）仍有漏切；覆盖率偏低，需要用标注数据校准 |
| **项目已有 git** | ✅ 已 `git init` 并完成首次提交 `ddf6aa1`（`data/`、`.venv/`、`models/*.pt` 等已忽略） |

---

## 2. 环境与工具

```powershell
# Python 解释器（不要用系统的 python，缺依赖）
D:\Projects\BadmintonStudio\.venv\Scripts\python.exe
#   Python 3.14.6 | torch 2.13.0+cu126 | CUDA 可用 | NVIDIA GeForce RTX 3080
#   ultralytics 8.4.150 | scipy 1.18.1 | opencv 5.0.0 | numpy 2.5.2

# 回归测试（44 项，不需要 GPU）
& D:\Projects\BadmintonStudio\.venv\Scripts\python.exe D:\Projects\BadmintonStudio\tests\test_core.py

# 前端
cd D:\Projects\BadmintonStudio\frontend ; npm run build
```

**项目里没有 `.git`。** 任何破坏性操作都不可回滚 —— 动手前先自己复制一份。

### 测试素材

| 素材 | 路径 | 说明 |
|---|---|---|
| 30 分钟 4K 原片 | `C:\Users\13127\Videos\20260913_羽毛球\20260913_羽毛球_clip9.mp4` | 主测试素材，21 GB |
| 960×540 代理 | `data\cache\proxies\20260913_羽毛球_clip9_m_aa6b31f616ee_960x540.mp4` | **分析实际用的就是它**，30 fps |
| 16 kHz 音轨 | `data\cache\audio\20260913_羽毛球_clip9_m_aa6b31f616ee_16000.wav` | 音频击球检测用 |
| 5 分钟小代理 | `data\cache\proxies\clip9_5min_m_ba93524aa74d_640x360.mp4` | 15 fps，快速试验用 |
| 上次完整分析结果 | `data\cache\frames\analysis_new.json` | 1.4 MB，含 signals/stats/rallies，**可以离线重跑切分**（见第 8 节） |

素材特点（**决定了所有算法选择的物理事实**，务必先理解）：

* 机位是**超广角/鱼眼侧方**，画面里同时能看到**多块场地**，球员只有约 100 像素高；
* 因此**音频击球声被严重污染**：`audio_reliability` 实测只有 **0.04**；
* 场地边界是弯的（7 点多边形手动标定），单应变换只能近似；
* 场景是**双打**（4 名球员），比赛几乎连续进行，回合之间停顿很短。

---

## 3. ⚠️ 事故详情：`pipeline.py` 损坏

### 3.1 症状

```
SyntaxError: invalid character '\u20ac' (U+20AC)   # pipeline.py 第 508 行
```

后端 `import bms.analysis.pipeline` 直接失败 → **AI 分析功能不可用**（`tests/test_core.py` 全部因导入失败而不能跑）。

### 3.2 根因（我犯的错）

我为了删掉两行 `hit_times=...` 参数，执行了：

```powershell
(Get-Content backend\bms\analysis\pipeline.py -Raw) -replace "(?m)^\s*hit_times=\(...\),\r?\n","" |
    Set-Content backend\bms\analysis\pipeline.py -NoNewline -Encoding UTF8
```

**这台机器上的 `Get-Content` 把 UTF-8 文件按 ANSI（GBK ≈ CP936）代码页解码**，随后
`Set-Content -Encoding UTF8` 又按 UTF-8 写回 —— 典型的「双重编码」事故。它同时删掉了
**约 175 个换行**（所以文件从 1477 行变成 1302 行）。

> 顺带说明一个**排查时踩了半天的坑**：`Get-Content -Raw` 出来的乱码文本里，
> 有些字符在「用 Python 读回来」时看起来是对的。别被迷惑 —— 判据永远以
> **Python 能不能 `ast.parse` 为准**，不要相信控制台显示。

### 3.3 损伤范围（已用脚本逐文件核实）

我写了 `scripts/integrity_check.py`，扫描 `backend/` + `scripts/` + `tests/` 下
**全部 58 个 `.py` 文件**，检查「能否 UTF-8 解码」「能否 `ast.parse`」「CJK 乱码比例」：

| 文件 | 结果 |
|---|---|
| `backend/bms/analysis/pipeline.py` | **唯一受损**：1477 → 1302 行，全部中文变乱码，语法错误 |
| 其余 57 个文件 | **全部完好**（语法通过、中文正常） |

> 报告存在 `data/cache/frames/integrity.txt`。
> 几个 `__init__.py` 报 `E1` 是因为文件带 BOM 且只有 5 字节，**这是原本就有的，无害**。

**重要：功能改动一行都没丢。** `rally.py`、`pose.py`、`players.py`、`rally_vision.py`、
`models.py`、测试、前端改动全都完好。受损的只有**编排层** `pipeline.py`。

### 3.4 已保留的恢复材料（都在，且已校验）

| 材料 | 路径 | 校验 |
|---|---|---|
| **受损文件的精确字节副本** | `data/cache/frames/pipeline_broken.py` | 与当前 `pipeline.py` **逐字节相同**，`sha256 = d80c356259e91686…`，79190 字节 |
| **22 份损坏前的逐字快照** | `data/cache/frames/snap/snap_*.txt` | 从会话记录解出，覆盖全文约 **71%**，中文与换行都正确 |
| 快照索引 | `data/cache/frames/snapshot_index.txt` | 每份快照的原始行号范围与当时文件总行数 |
| **编译产物** | `backend/bms/analysis/__pycache__/pipeline.cpython-314.pyc` | 编译于损坏前 39 秒。含全部字符串常量（中文 docstring 原文）、行号表、字节码 |
| **行号偏移表** | `data/cache/frames/anchor_table.txt` | 见下文，**这是恢复换行的关键线索** |
| 会话原始记录 | `data/cache/frames/session.jsonl`（8 MB，来自 `~/.dsh/sessions/…`） | 快照就是从它解出来的 |
| 本次会话的编辑记录 | 会话本身 | 精确说明我改过什么（见第 6 节） |

**注意**：`snap/snap_05_L0504-0515_of1302.txt` 是**损坏之后**读的，内容是乱码，
**不能当恢复源**。其余 22 份都是损坏前的。`snapshot_index.txt` 里凡是 `of 1302` 的都排除。

### 3.5 已尝试但没走通的自动恢复

| 方法 | 结果 |
|---|---|
| 用 .NET `Encoding.GetEncoding(936/950/54936)` 反解 | 产生的字节不是合法 UTF-8（在 offset 75-76 断开） |
| 暴力枚举全部 140 种 .NET 编码，找「反解后是严格 UTF-8」的 | 只有 ASCII 类编码「通过」（实为把中文替换成 `?`），**没有任何编码能真正反解** |
| 用 Python `gbk` 反查表（含 PUA 处理） | 83 个字符（U+E0xx–U+E7xx）不在表内，且 235 条映射冲突 |
| 用 `.pyc` 字符串常量当罗塞塔石碑推字符映射 | 匹配上 12 个字面量，但非 ASCII 段数对不上 → 0 条映射 |

**结论：纯自动反解走不通，必须靠「快照拼接 + 结构化补全」。**

### 3.6 关键线索：行号偏移表

`data/cache/frames/anchor_table.txt` 给出了每个顶层定义在「原文件」与「损坏文件」里的行号，
偏移量**单调递减**，因此**每一段丢了多少换行是已知的**：

| 函数 | 原(.pyc)行 | 损坏文件行 | 偏移 | 该段内丢的换行 |
|---|---:|---:|---:|---:|
| `_noop` | 43 | 37 | −6 | 6（模块文档串里） |
| `_try_import` | 47 | 41 | −6 | 0 |
| `attribute_sides` | 55 | 49 | −6 | 0 |
| `run_analysis` | 142 | 125 | −17 | **11** |
| `_manual_poly` | 557 | 507 | −50 | **33** |
| `_build_rally` | 588 | 532 | −56 | 6 |
| `_hits_to_events` | 654 | 595 | −59 | 3 |
| `_quality_per_rally` | 665 | 606 | −59 | 0 |
| `_mean_range` | 693 | 634 | −59 | 0 |
| `_downsample` | 703 | 644 | −59 | 0 |
| `_pack_signals` | 713 | 654 | −59 | 0 |
| `_subject_span` | 796 | 727 | −69 | **10** |
| `_court_payload` | 830 | 759 | −71 | 2 |
| `_join_abutting` | 846 | 775 | −71 | 0 |
| `_segments_to_intervals` | 878 | 800 | −78 | 7 |
| `_seg_quality` | 904 | 826 | −78 | 0 |
| `_segment_rallies` | 930 | 849 | −81 | 3 |
| `_merge_by_availability` | 1094 | 976 | −118 | **37** |
| `resegment` | 1274 | 1117 | −157 | **39** |
| `_quality_default` | 1424 | 1250 | −174 | 17 |
| `_MediaStub` | 1428 | 1254 | −174 | 0 |
| `_rebuild_hits` | 1435 | 1261 | −174 | 0 |
| `_dict_from` | 1450 | 1276 | −174 | 0 |
| `_dict_from_multi` | 1457 | 1283 | −174 | 0 |
| `_best_overlap` | 1469 | 1295 | −174 | 0 |

**原文件最大行号 1477，损坏文件 1302 行，共丢 175 个换行。**
（注意：丢失集中在 `run_analysis` 内部、`_manual_poly` 之前、`_subject_span`/`_segments_to_intervals`
附近、`_merge_by_availability` 与 `resegment` 内部 —— 大部分是**注释和 docstring 的换行**，
所以文件里出现了「多行注释被拼成一行」的现象。这也是为什么 `ast.parse` 只在少数几处报错。）

### 3.7 推荐的恢复路线

**目标**：得到一个 `ast.parse` 通过、并且**编译出的字节码与 `.pyc` 等价**的 `pipeline.py`。

**路线 A（推荐，保真度最高）：快照拼接 + 结构化补全**

1. 以 `data/cache/frames/pipeline_broken.py` 为骨架（它的 **ASCII 一字未丢**，
   代码结构、标识符、缩进、标点全都完好）；
2. 用第 3.6 节的偏移表把「原文件行号 ↔ 损坏文件行号」对齐；
3. 对快照覆盖的区域（约 71%），**逐字替换**成快照内容 —— 这一步同时修复中文和换行；
4. 对未覆盖的区域（模块文档串 1~141、`_hits_to_events` 附近、`resegment` 尾部那批辅助函数），
   用「损坏文本的 ASCII + `.pyc` 的行号表 + `.pyc` 的字符串常量」重建：
   * 中文注释/docstring → 从 `.pyc` 的 `co_consts` 里取（**docstring 有原文**），
     注释文字需要按上下文重写（这部分不可能逐字还原，请在文档里标注）；
   * 换行 → 按第 3.6 节的偏移量，配合 `.pyc` 每行对应的语句信息插入。

**验证（必须做，且是强验证）**：

```python
# 重建后的文件编译成字节码，与 .pyc 逐 code object 比较 co_code / co_consts / 行号表
import marshal, pathlib
old = marshal.loads(pathlib.Path(PYC).read_bytes()[16:])
new = compile(open(recovered, encoding='utf-8').read(), PYC_STEM, 'exec')
# 递归比较 co_code、co_names、co_consts（跳过 docstring 差异）、co_varnames
```

字节码一致即可证明**代码语义完全等价**；注释文字不同不影响这个验证（请在文档里说明）。

**路线 B（最快恢复可用）**：只修 `ast.parse` 报错的那几处换行。文件立刻可用，
但中文仍是乱码、仍有被拼行的注释。**只适合应急，不要作为最终交付。**

**路线 C**：如果用户手上有 `pipeline.py` 的备份（另一台机器 / 云盘 / 编辑器历史），
直接用备份覆盖，然后按第 6 节把本次的 9 处改动重新贴上。**这是最省事也最可靠的路径，先问用户。**

### 3.8 🚫 绝对不要做的事

* **不要**用 PowerShell 的 `Get-Content` / `Set-Content` / `-replace` 处理任何含中文的源文件。
  要用 Python（`pathlib.Path.read_text(encoding='utf-8')` / `write_text(..., encoding='utf-8')`）。
* **不要**用 `& $py -c @"…"@` 这种控制台 heredoc 传中文字面量 —— 控制台编码会把中文弄坏，
  你会得到假象。**要写脚本就先用编辑工具落成 `.py` 文件再执行。**
* **不要**在没备份的前提下对源文件做「就地重写」。
* 恢复过程中**不要**直接覆盖 `pipeline.py`，先写到临时文件、验证通过再替换。

---

## 4. 本项目本次到底改了什么（全部清单）

### 4.1 完好的功能改动

| 文件 | 改动摘要 |
|---|---|
| `backend/bms/analysis/rally.py` | 新增 `hit_gap_limit()` / `split_by_hit_gaps()` / `_hit_idx_in_window()`；**重写 `refine_with_hits()`** 让终点能被收紧；`attach_features()` 不再丢弃已有 features；新增 `MAX_INTRA_HIT_GAP=3.0` |
| `backend/bms/analysis/rally_vision.py` | 无功能性新增（`segment_by_hit_runs` 试过又撤掉了，`__all__` 已清理） |
| `backend/bms/analysis/pose.py` | **新增（641 行）**：姿态层。`analyze_pose()` / `swing_peaks()` / `hit_swing_evidence()` / `gate_hits()` / `filter_hits()` |
| `backend/bms/analysis/players.py` | 新增 `_select_active_players_windowed()`，`analyze_players()` 改用它 |
| `backend/bms/core/models.py` | 新增 `hit_tail_seconds=0.9`、`use_pose=True`；`post_roll` 默认 1.6 → 0.5 |
| `frontend/src/lib/types.ts` | 新增 `hit_tail_seconds`、`use_pose` 字段 |
| `frontend/src/store/useStore.ts` | `post_roll` 1.4 → 0.6；新增 `hit_tail_seconds: 0.9`、`use_pose: true` |
| `frontend/src/components/AnalysisDialog.tsx` | 新增「死球余量」滑块、「姿态辅助（击球归属）」开关 |
| `tests/test_core.py` | 新增 5 组测试（空档切分 / 终点锚定 / 合并保留拍数 / 姿态三组 / 择优三情况 / resegment 复用球员信号），共 44 项 |
| `README.md` | 更新「回合终点为什么锚到最后一拍」「姿态辅助」「为什么以前不准」「即时重切分」等章节 |
| `models/yolo11n-pose.pt` | 新下载（6.2 MB） |

### 4.2 新增的验证脚本（都在 `scripts/`）

| 脚本 | 用途 |
|---|---|
| `eval_rally_end.py` | **离线**对比旧/新切分（读已有分析 JSON，毫秒级，不用 GPU） |
| `rerun_analysis.py` | 端到端重跑完整分析；`--reuse` 可直接读上次结果只出报告 |
| `check_player_coverage.py` | 核对球员信号覆盖（本次最重要的一个指标） |
| `test_pose.py` | 姿态模块端到端：覆盖率、挥拍信号、门控保留比例、聚类对比 |
| `contact_sheet.py` | 把若干时刻的画面拼成带时间戳的接触印相，肉眼核对边界 |
| `probe_pose_gate.py` | 姿态可行性的原始调查脚本（保留作证据） |

### 4.3 恢复用的脚本（`scripts/`，都是 ASCII 输出，可放心读）

`integrity_check.py`、`anchor_table.py`、`extract_snapshots.py`、`dump_transcript.py`、
`inspect_pyc.py`、`side_by_side.py`、`derive_mapping.py`、`derive_decode.py`、
`recover_encoding.ps1`、`recover_bruteforce.ps1`、`pick_recovered.py`、`diagnose_corrupt.py`、
`shape_of_session.py`、`shape2.py`

---

## 5. 算法改动详解（交付给下一位的核心知识）

### 5.1 问题诊断（用真实数据说话）

从 30 分钟素材的分析结果里量出来的事实：

| 现象 | 数据 | 后果 |
|---|---|---|
| 球员检测覆盖率低 | `segmentation.player_coverage = 0.48` | 球员运动切分只出 11 段，最终退回 `activity_valleys` |
| **`frame_boxes` 有 58% 的时间是空的** | `active_count` 非零仅 0.475；**0~300s 与 1200~1800s 完全为空** | `player_motion_full` 前 **372 秒恒为 0** |
| 音频不可信 | `audio_reliability = 0.04`，权重 0.023 | 等于这一路没用 |
| 活跃度曲线不塌 | 0~69s 一直平在 0.49~0.55，而 `threshold_hi = 0.541` | 迟滞状态机永不退出 → 切出 **69.35 秒、72 拍**的「一回合」 |
| **终点只能往后推** | `refine_with_hits` 里写的是 `iv.end = max(iv.end, last + tail)` | 即使击球正确也**无法**把球落地后那段砍掉 |

### 5.2 修复 1：终点锚定到最后一拍（`rally.py::refine_with_hits`）

```
终点 = 最后一拍 + hit_tail_seconds      （默认 0.9s，覆盖球落地所需飞行时间）
起点 = 第一拍   - pre_roll               （默认 1.2s，发球准备）
```

关键点：
* 旧代码用 `max()`，**只能延长不能收紧** → 改成「有证据时才收紧」；
* 收紧的判据：**全局下一次击球离最后一拍超过 `hit_gap_limit`**（连隔壁场地都没声音了，
  说明这一段确实没人打球）；若紧接着还有击球，说明球还在飞，保持更晚的终点；
* 取击球窗口时**用邻居边界夹逼**（`_HIT_TOL_BEFORE=1.0` / `_HIT_TOL_AFTER=1.2`），
  否则切点两侧会把同一拍都算进来 → 左段往后留 0.9s、右段往前留 1.2s → 重叠 2.1s
  → `dedupe_overlaps` 在中间切一刀 → 又首尾相接 → 被 `_join_abutting` 合并回去，**白切**。

### 5.3 修复 2：按击球空档切分（`rally.py::split_by_hit_gaps`）

**这是「一回合里混进下一个回合」的正解。** 依据是物理而不是调参：

* 一回合内的拍间隔由球的飞行时间决定 —— 全片实测 p50 = 0.48s、p80 = 1.19s、p99 = 3.25s；
* 两个回合之间隔着捡球/换发球，**没有任何人击球**的时间普遍 ≥4s。

所以「击球序列里的大空档」几乎就是回合边界 —— 这比「球员有没有停下来」可靠得多，
因为球员捡球时也一直在走动，活跃度根本不塌。护栏：两侧各至少 2 拍、各自撑得起
`min_side_seconds`，用来区分「真的是两个回合」和「漏检了几拍」。

`hit_gap_limit()` 从**典型**拍间隔（p70）乘 1.8 再取 floor=3.0 / cap=4.5。
⚠️ **这里有个教训**：原来我用 p80 乘 1.8 —— 门控之后序列变稀疏，p80 自己被
「回合之间的大空档」抬高到 2.42s，算出 4.36s 的门限，于是 3.7s / 3.1s 的真实停顿
全都切不开，20 多秒的回合留在那里。改成 p70 后门限 3.38s，那一回合正确裂成 7.76s + 9.86s。

### 5.4 修复 3：逐时间窗挑选比赛球员（`players.py::_select_active_players_windowed`）

**这是本次收益最大的单项修复。**

* 30 分钟素材有 **285~480 条轨迹** —— 球员被遮挡 / 走出画面就会换新 id；
* 而 `_select_active_players` 按**全片统计量**排名取前 4 条 → 4 条很可能**全部落在视频中段**；
* 于是 `frame_boxes` 在其余时段为空 → `player_motion` 整段恒为 0 → 球员切分整段作废。

做法：按 180s 窗口 / 90s 步长切窗，每窗只拿与之有时间重叠的轨迹去排名，
再把各窗结果取并集；全片那一遍仍跑，用它的 `size_ref` 做最后一道「够大」的闸门。

**实测：`active_count` 非零比例 0.475 → 0.981**（缺失仅 35s，集中在 222~244s 一个 21.6s 的空洞）。

### 5.5 修复 4：`_merge_by_availability` 里 `valid_ratio` 算了却没用

**这是一个真正的 bug，藏得很深。** 旧代码是「`vis` 有候选就用 `vis`，没有就退回 `act`」——
把用来做判断的 `valid_ratio` 算出来却**从未调用**。后果：凡是球员切分**故意留空**的地方
（那正是两个回合之间真实的停顿），都会被活跃度候选补上，好不容易切开的回合又被粘回去
（实测最长回合因此多出 6 秒）。

修好之后要**分三种情况**，因为只做「没候选就留空」会掉召回：

| 窗内球员检测 | 球员切分有候选 | 处理 |
|---|---|---|
| 有效 | 有 | 用球员切分（边界就是球员真的停下来） |
| 有效 | 无，但 `has_motion_burst()` 为真 | 球员确实连着动过 → 是漏检 → 用活跃度候选兜住 |
| 有效 | 无，且没有运动爆发 | **留空**（真正的停顿） |
| 无效 | — | 用活跃度切分 |

`has_motion_burst()` 的门限**故意比 `segment_by_player_motion` 的 `min_core`(2.5s) 放宽到 1.2s**：
它是召回安全网，宁可疑心一点也别把整段回合丢掉。

⚠️ **不要**把「击球序列」也用来填这些空窗。我试过：它会把停顿全填上，
最长回合从 22.7s 反弹到 36.4s、>30s 的回合回到 7 个。**试过，回退了，别再走这条路。**

### 5.6 新增：姿态层（`analysis/pose.py`）

**它不参与「用哪一路信号切分」，只回答一个问题：这个音频击球是不是我们打的。**

思路：我们的球员只在真击球时挥拍；隔壁场地的击球声在我们画面上**没有任何对应动作**。
所以「这一拍是否属于我们」第一次有了直接观测，而不再靠音频统计量去猜。

工程要点：

1. **不重新检测、不重新跟踪**，直接复用 `PlayerSignal.frame_boxes`（省一半算力，
   更避免「两套跟踪结果对不上」这种最难查的 bug）；
2. **把球员框裁出来放大到 192×192 再送 `yolo11n-pose`**。球员只有约 100px 高，
   整帧直接跑关键点会飘（腕、肘先丢）。裁窗要**往上偏并留余量**（`up_shift`、`margin=0.35`），
   否则头顶击球时手腕跑出框外；
   ⚠️ **`frame_boxes` 是归一化坐标（0~1）**，必须先乘画面尺寸还原成像素 ——
   我第一次忘了乘，得到 4×4 的窗口，姿态一个都检不出来；
3. **挥拍信号 = 手腕相对双肩中点的位移速度 ÷ 身体高度**。减肩中点是去掉整体位移
   （跑动时手腕也动，那不是挥拍），除身体高度是为了和远近无关；
4. **门控用「挥拍峰 ↔ 击球一对一匹配」，不是「窗口内取最大手腕速度」。**
   后者实测证据分中位数 **0.723**（几乎没区分度，因为对拉时 1~1.5s 就有一拍，
   ±0.35s 窗口已覆盖半个拍间隔）；改成峰匹配后中位数 **0.00**，保留比例 **41.6%**，
   和「我们这场大约占所有击球声一半」的直觉吻合；
5. **一定会降级，而不是变成新的错误来源**：覆盖率 < 0.35、保留比例落在 [12%, 97%] 之外、
   没有 GPU、权重缺失 —— 任何一条命中就原样放行全部击球。

**实测开销**：90s 素材 5.6s（折合 30 分钟约 1.9 分钟）；但完整管线实测 **+226s（约 3.8 分钟）**，
因为 `analyze_players` 之后要重新解码视频。结果按「视频 + 球员框指纹」缓存到 `data/cache/pose/`。

### 5.7 修复 5：`resegment` 复用存下来的球员信号

旧 `resegment` 没有球员框，只能退化成「只看活跃度」——于是同一个工程
「快速重切分」和「完整重跑」给出**不一样的回合数**，而重切分恰恰是调切分参数时最常用的操作。

现在把 `player_motion_full` 和 `player_coverage_full` 都存进 signals，
`resegment` 直接走**完全相同的 `_segment_rallies`**。
⚠️ 另外两个一致性坑：`resegment` 以前漏了 `_join_abutting`；`keep_keys` 漏了 `pose_trace`。都已补。

---

## 6. `pipeline.py` 必须包含的改动（恢复时的验收清单）

> 这些改动**在损坏文件里都还在**（ASCII 未丢），下面是它们的落点，方便你核对。

| # | 位置 | 内容 |
|---|---|---|
| 1 | `run_analysis` 边界锚定段（原 ~392-405，损坏文件 ~404-435） | 删掉「只有非 player_motion 才 refine」的判断，改成**所有路径都 refine**：`RA.refine_with_hits(intervals, hits, pre_roll=params.pre_roll, post_roll=params.post_roll, tail_seconds=params.hit_tail_seconds, trim_start=not player_path)`，其中 `player_path = seg_trace.get("method") == "player_motion"` |
| 2 | `run_analysis` 姿态阶段（损坏文件 288~320） | 在「球员」之后、「羽毛球」之前插入：`pose_sig` / `pose_trace` / `hits_all` / `gate_mask`；调用 `mod.analyze_pose(str(media.proxy_path or media.path), boxes, float(getattr(player_sig,"fps",12.0) or 12.0), on_progress=…, cancel=cancel)`，然后 `mod.gate_hits(hits, pose_sig)` + `mod.filter_hits(hits, gate_mask)`；进度打点 0.64 |
| 3 | `run_analysis` stats/signals（损坏文件 466~476） | `res.stats["pose_trace"] = pose_trace`；signals 增加 `pose_swing` / `pose_ok` / `pose_overhead` / `pose_fps` / `pose_coverage`（用 `_downsample`）。**故意不存 `hit_gate` 掩码**（存下来的 hit_times 已经是门控后的，再套一次会过滤两遍） |
| 4 | `_join_abutting`（损坏文件 775） | 合并时改为 `prev.hit_indices = sorted(set(prev.hit_indices) \| set(iv.hit_indices))`（旧代码清空 → 合并出来的回合一律显示「0 拍」，而 `shot_count` 参与评分）；`tail_gap` 取两侧最大值 |
| 5 | `_merge_by_availability`（损坏文件 976） | 新参数 `coverage=None, player_motion=None, merge_trace=None`；`cov` 三级回退（存下来的覆盖率 → `detection_coverage(boxes)` → 全 1）；新增 `valid_ratio()` / `moving_ratio()` / `has_motion_burst()`；窗口循环按 5.5 节的三情况决策；`used` 计数写进 `merge_trace` |
| 6 | `_segment_rallies`（损坏文件 849） | 新参数 `player_motion=None, player_coverage=None`；当给了球员运动曲线时**复用**：`RV.segment_by_player_motion(RV.SegmentSignals(...))` + `RV.verify_with_shuttle(...)`，并打 `trace["player_signal"]="reused"`；`elif` 分支里算 `pm_arr = RV._resample_to(RV.box_motion(list(boxes), player_fps, window=1.0), player_fps, fused.fps, fused.activity.size)`；`_merge_by_availability(..., coverage=cov_arr, player_motion=pm_arr, merge_trace=trace.setdefault("windows", {}))` |
| 7 | `resegment`（损坏文件 1117） | 重建为：读取 `player_motion_full` / `player_coverage_full` / `player_fps`，然后调用 `_segment_rallies(fused, params, duration, player_sig=None, shuttle_sig=None, player_motion=…, player_coverage=…)`；补上 `intervals = _join_abutting(intervals)`；`keep_keys` 加入 `"pose_trace"`；stats 里保留完整 `seg_trace` 而不是只写 method/count |
| 8 | `_pack_signals`（损坏文件 ~687） | 增加 `out["player_coverage_full"] = [1.0 if v > 0.5 else 0.0 for v in RV.detection_coverage(list(boxes), pfps)]` |
| 9 | 清理 | 我最后一次改动是把试验性的 `hit_times=` 实参和 `_segment_rallies` 的 `hit_times` 形参删掉（那次改动**导致事故，且改动后的状态从未通过验证**）。恢复时确认这两处已删。另外 `hits_all = hits` 赋值了但没用到，可以删 |

---

## 7. 实测数据（验收基准）

### 7.1 逐阶段对比（同一素材：30 分钟 4K clip9）

| 指标 | ① 原代码 | ② +逻辑修复 | ③ +姿态层 | ④ +`valid_ratio`修复+运动爆发兜底 |
|---|---:|---:|---:|---:|
| 回合数 | 46 | 74 | 76 | **51** |
| 平均时长 | 15.44s | 11.17s | 9.23s | **8.76s** |
| 中位时长 | 13.98s | 9.38s | 7.11s | **7.19s** |
| **最长回合** | **69.35s** | 28.78s | 28.73s | **20.48s** |
| >30s 的回合 | **2** | 0 | 0 | **0** |
| 间隔 <1s 的对数 | 5 | 11 | 2 | 4 |
| 时间轴覆盖率 | 39.4% | 45.9% | — | 24.8% |
| 球员检测覆盖率 | **0.48** | 0.996 | 0.996 | **0.998** |
| 实际用的切分方法 | `activity_valleys` | `player_motion+activity` | 同 | 同 |
| `player_motion` 质量分 | — | 0.88 | — | **0.914**（> 活跃度 0.887） |
| `audio_reliability` | **0.04** | 0.04 | **0.225** | 0.226 |
| 音频权重 | 0.023 | 0.023 | 0.137 | 0.139 |
| `shot_count = 0` 的回合 | **46（全部）** | 0 | 0 | **0** |
| 单次分析耗时 | — | 324s | 550s | 340s |

### 7.2 姿态层实测

| 指标 | 值 |
|---|---|
| 姿态覆盖率 | 0.978 ~ **0.993** |
| 检出挥拍峰 | 1943 个 / 1800s（约 1.08 个/秒） |
| 门控保留比例 | **0.416**（1083 / 2604 拍）—— 90s 窗口独立测出 0.413，**跨尺度一致** |
| 证据分中位数 | 改「峰一对一匹配」前 **0.723** → 之后 **0.00** |
| 挥拍信号分布 | p50 1.13~1.39，p90 3.99~4.26（单位：身体高度/秒） |
| 90s 素材耗时 | 5.6s |

### 7.3 球员信号覆盖（`_select_active_players_windowed` 的效果）

| 时间段 | 有球员的时间占比 | 平均人数 |
|---|---:|---:|
| 0–300s | **90.6%**（旧：0%） | 1.54 |
| 300–600s | 98.6% | 1.89 |
| 600–900s | 100% | 2.13 |
| 900–1200s | 100% | 2.64 |
| 1200–1500s | 99.6%（旧：7%） | 2.30 |
| 1500–1800s | 99.7%（旧：4%） | 1.74 |

整体：`active_count` 非零 **0.475 → 0.981**。

### 7.4 怎么复现这些数字

```powershell
$py = "D:\Projects\BadmintonStudio\.venv\Scripts\python.exe"

# 1) 离线对比旧/新切分（毫秒级，不用 GPU）—— 读 data/projects/…analysis.json
& $py scripts\eval_rally_end.py

# 2) 用上次存下来的结果做离线重切分（毫秒级）
& $py scripts\rerun_analysis.py --reuse

# 3) 完整端到端重跑（约 340s，含姿态）
& $py scripts\rerun_analysis.py

# 4) 球员覆盖率
& $py scripts\check_player_coverage.py

# 5) 姿态端到端
& $py scripts\test_pose.py --seconds 90

# 6) 回归测试
& $py tests\test_core.py
```

---

## 8. 已知问题 / 未验证项（诚实清单）

1. ~~**`pipeline.py` 未恢复** —— 这是阻塞项，见第 3 节。~~ **已解决**，见顶部「恢复完成」节。
2. ~~**我最后一次 `pipeline.py` 改动（删 `hit_times` 实参）从未通过任何验证**~~ ——
   恢复时已按重切分路径验证：`rerun_analysis.py --reuse` 复现第 7.1 节第 ④ 列。
3. **`hit_gap_limit` 的 p70 改动只做了离线验证** —— 用 `resegment` 验证过（22.4s 那一回合
   正确裂成 7.76s + 9.86s），但**没有跑过完整的 GPU 端到端**。恢复正常后请补跑。
4. **仍有漏切区段**：15~28s、51~54s 这几段，姿态证据（挥拍峰 + 球员运动）明确显示在比赛，
   但球员切分和活跃度切分**同时**不给候选 → 回合整段消失。根因是
   `segment_by_player_motion` 的门限（`min_core=2.5s` 要求**连续**移动段、
   `min_rest=1.2s`）在这段素材上有刀刃效应：实测有一处因为前置静默段只有 **1.17s**
   （差 0.03s）就整段被否掉，另一处因为最长连续移动段 2.42s < 2.5s 被否掉。
   我离线扫过 `min_core`/`min_rest`，放到 (1.2, 0.8) 能把 19.4~24.9s 找回来
   （86 回合、最长 24.06s、无 >30s），但**基于单个片段调这两个参数就是过拟合**，
   所以**故意保留了模块原默认值**。正确做法见下一条。
5. **覆盖率从 39.4% 降到 24.8%** —— 切得更紧了（符合用户明确选择的「宁紧」口径：
   最后一拍 + 0.6~1.0s），但**没有标注数据就无法判断是否切过头**。
   51 回合 / 30 分钟 ≈ 每 35 秒一分，对业余双打是合理的，但这是定性判断。
6. **缺少 ground truth** —— 这是所有调参问题的根源。强烈建议先做一版标注：
   取一段 5~10 分钟素材，人工标出每个回合的起止（哪怕只标 20 个），
   然后才谈得上调 `min_core` / `min_rest` / `hit_gap_limit` / 姿态门控阈值。
   现有 `scripts/contact_sheet.py` 可以把若干时刻拼成带时间戳的接触印相辅助人工标注。
7. `resegment` 与完整重跑仍不完全一致（因为重切分不做姿态门控、也不重算融合权重），
   但已从「方法都不同」改善到「同一方法、输入略有差异」。
8. 用户上一次操作时 App 正在运行，它**覆盖了** `data/projects/p_e274f1ae3e1e.m_aa6b31f616ee.analysis.json`
   （原来的 46 回合基线被换成了新代码跑出的 71 回合、params `pre_roll=1.0 / post_roll=0.6`）。
   本文档第 7.1 节「原代码」那一列的数字是我在覆盖前记录下来的，**现在文件里已经没有那份基线了**。

---

## 9. 建议的下一步（按优先级）

1. **先做 `git init` + 首次提交**（把除受损文件外的所有东西提交上去）。
   ```powershell
   cd D:\Projects\BadmintonStudio
   git init ; git add -A ; git commit -m "wip: 回合终点优化 + pipeline.py 待恢复"
   ```
2. **问用户有没有 `pipeline.py` 的备份**（路线 C 最省事）。
3. 没有备份就走**路线 A**（第 3.7 节）：快照拼接 + 偏移表补换行，
   用「编译后字节码与 `.pyc` 等价」做验收。
4. 恢复后依次跑：`tests/test_core.py` → `eval_rally_end.py` → `rerun_analysis.py`（完整），
   确认第 7.1 节第 ④ 列的数字能复现。
5. 做一小段**人工标注**，然后校准第 8.4 节的门限问题（这是剩下的精度大头）。
6. 可选：把姿态的 `hit_tail_seconds` 按球种自适应 —— `PoseSignal.overhead` 已经算好了
   （挥拍手是否在肩线以上），头顶球给短尾巴、下手球给长尾巴。这是当初用户选过的
   「按球种自适应」选项，只是当时先做了固定值。

---

## 10. 坑（务必避免，血泪）

| 坑 | 说明 |
|---|---|
| **PowerShell 处理中文源文件** | 见第 3.8 节。这是把项目搞坏的直接原因 |
| **控制台 heredoc 传中文** | `& $py -c @"…"@` 会把中文字面量弄坏，输出也会乱码。要写脚本就**先落成 `.py` 文件**（用编辑工具）再执行 |
| **控制台显示不可信** | 判断文件是否正确，用 `ast.parse` / 字节比对 / 哈希，**不要用肉眼看控制台** |
| **`frame_boxes` 是归一化坐标** | 0~1，不是像素。忘了乘画面尺寸会得到 4×4 的裁剪窗 |
| **`Select-Object -First N`** | 截断管道会让命令以 exit code 1 结束，**这不是命令失败** |
| **进度回调高频触发** | `analyze_players` / `analyze_pose` 的 `on_progress` 每批都调，直接 print 会刷屏几千行。要按百分比节流 |
| **`_seg_quality` 会「择优」换路径** | 改动 `vis`/`act` 任何一路都可能让最终选中的路径变掉，结果整体漂移。改动后一定要看 `stats.segmentation.method` |
| **`_merge_by_availability` 的窗口划分** | 往 `marks` 里多加一路边界会**重新划分所有窗口**，「补空」这个动作会顺带改变已定好的窗口，结果整体漂移（实测第一段回合直接消失）。补空就只补空，别参与划分 |

---

## 11. 命令速查

```powershell
$py = "D:\Projects\BadmintonStudio\.venv\Scripts\python.exe"
$root = "D:\Projects\BadmintonStudio"

& $py "$root\tests\test_core.py"                     # 44 项回归测试
& $py "$root\scripts\integrity_check.py"             # 全项目文件完整性/乱码扫描
& $py "$root\scripts\anchor_table.py"                # 原文件 vs 损坏文件的行号偏移表
& $py "$root\scripts\extract_snapshots.py"           # 重新从会话记录解出逐字快照
& $py "$root\scripts\rerun_analysis.py --reuse"      # 离线重切分（毫秒级）
& $py "$root\scripts\rerun_analysis.py"              # 完整端到端（约 340s）

cd "$root\frontend" ; npm run build                   # 前端构建 + TS 类型检查
```

---

## 12. 关键文件速查

| 路径 | 说明 |
|---|---|
| `backend/bms/analysis/pipeline.py` | ⚠️ **受损，待恢复**（编排层） |
| `backend/bms/analysis/rally.py` | 边界锚定 + 击球空档切分（完好，核心逻辑） |
| `backend/bms/analysis/pose.py` | 姿态层（完好，新增 641 行） |
| `backend/bms/analysis/players.py` | 逐时间窗比赛球员判定（完好） |
| `backend/bms/analysis/rally_vision.py` | 球员静默段 / 活跃度谷值切分（完好） |
| `backend/bms/core/models.py` | `AnalysisParams`（`hit_tail_seconds` / `use_pose`） |
| `tests/test_core.py` | 44 项回归测试 |
| `data/cache/frames/pipeline_broken.py` | 受损文件的精确副本（sha256 `d80c356259e91686…`） |
| `data/cache/frames/snap/` | 22 份损坏前逐字快照（恢复主力） |
| `data/cache/frames/anchor_table.txt` | 行号偏移表（恢复换行的关键） |
| `data/cache/frames/analysis_new.json` | 上次完整分析结果（可离线重切分） |
| `backend/bms/analysis/__pycache__/pipeline.cpython-314.pyc` | 损坏前编译产物（结构 + 字符串常量 + 行号表） |
| `README.md` | 已更新算法说明章节 |
