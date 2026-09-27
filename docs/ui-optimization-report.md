# 界面优化说明文档

本文档记录本轮前端界面优化的全部改动，包含问题描述、优化方案与前后对比，供 review 使用。

范围：剪辑台（Studio）左栏「回合」面板信息架构、全应用按钮文案与交互反馈、其余页面的系统性排查。

验证：`npm run build`（tsc + vite）通过、`npm run lint` 0 error、`tests/test_core.py` 全部通过（含 i18n 双语目录一致性）。

---

## 一、左栏「回合」面板信息架构重构（核心）

### 问题描述

优化前，`RallyPanel.tsx` 把所有控制项（搜索、筛选、范围切换、排序、统一重算、筛选面板、成片现状、重建成片、切分调参、邻场抑制、评分口径、批量操作）堆在一个 `max-h-[46%]` 的可滚动区里：

- 「挑回合」这个主任务被大量调参控件淹没；
- 主 CTA「重建成片」随滚动消失，用户不知道下一步该点哪；
- 信息层级扁平，筛选 / 评分 / 切分三类不同性质的控件无分区。

### 优化方案

把面板拆成「固定顶栏 + 折叠筛选 + 主列表 + 底部成片动作 + 折叠分析设置」五层，主任务优先、高级调参默认收起。

```
优化前（一个滚动区）                    优化后（分层）
┌──────────────────────┐              ┌──────────────────────┐
│ 搜索 / 筛选 / 清空      │              │ 搜索 / 筛选 / 清空      │  ← 固定顶栏
│ 范围切换               │              │ 范围 + 排序            │
│ 统一重算提示            │              │ 匹配 N/M · 时长 · 评分口径│
│ 匹配数 + 排序           │              ├──────────────────────┤
│ 筛选面板（默认展开）     │              │ ▾ 筛选条件（默认折叠）  │  ← 折叠
│ 成片现状               │              ├──────────────────────┤
│ 重建成片 / 追加 / 预览   │              │ 回合列表（主区域）      │  ← 获得更多空间
│ 切分调参 + 邻场抑制      │              ├──────────────────────┤
│ 评分口径               │              │ 成片现状 + 重建成片/追加 │  ← 固定底部
│ （全部可滚动）          │              │ 批量保留/排除 · 前40%   │
└──────────────────────┘              ├──────────────────────┤
                                       │ ▸ 分析设置（默认折叠）  │  ← 评分+切分+邻场抑制
                                       └──────────────────────┘
```

### 具体改动

| 改动 | 优化前 | 优化后 |
|------|--------|--------|
| 筛选面板默认状态 | `showFilter=true`（默认展开） | `showFilter=false`（默认折叠），减少首屏噪音 |
| 成片动作位置 | 在可滚动区，随滚动消失 | 移到固定底部区，主 CTA 始终可见 |
| 切分调参 / 邻场抑制 / 评分口径 | 各自独立常驻 | 合并为一个「分析设置」折叠区，默认收起 |
| 当前评分口径可见性 | 只有展开才能看到 | 固定顶栏常驻显示当前口径名（如「均衡」） |
| 统一重算提示（全部素材范围） | 常驻占位 | 归入「分析设置」折叠区 |

---

## 二、按钮文案改进

| 位置 | 优化前 | 优化后 |
|------|--------|--------|
| 切分调参「再细一档」 | 再细一档 | **切分更细**（tooltip 补充说明「只把切分粒度调细一档」） |
| 切分调参「应用并重新切分」 | 应用并重新切分 | **应用切分参数** |
| 标注页步进按钮 | `« -1s (←)` / `(→) +1s »`（硬编码英文 `s`） | **后退 1s** / **前进 1s**（走 i18n） |
| 时间线「清空时间线」 | Trash2 图标（与「删除片段」同图标易误点） | 改用 **Eraser** 图标区分 |
| 分析对话框「切分依据」 | 用滑杆表达 3 个离散枚举 | 改用 **Segmented 分段控件** |
| 导入按钮图标 | 文件 / 文件夹模式都用 FolderOpen | 文件模式用 **FileVideo**，文件夹用 FolderOpen |

---

## 三、交互反馈统一

| 问题 | 改动 |
|------|------|
| 播放器播放/暂停按钮纯图标无说明 | 补 `title` + `aria-label`（播放 / 暂停） |
| 播放器全屏按钮纯图标无 aria-label | 补 `aria-label` |
| 素材卡片删除、标注列表删除、导出卡片播放预览等 icon-only 按钮无 aria-label | 逐一补 `aria-label` / `title` |
| 时间线撤销/重做始终可点 | 无可撤销/重做内容时置 `disabled`（依据 `history`/`future` 长度） |
| 导出页「刷新」无反馈 | 补 `loading` 态（`refreshing`） |
| 素材库「创建并进入」无 loading | 补 `loading` 态（`saving`） |
| 批量选择「全选/取消全选」无按下态 | 补 `active:opacity-70` |

---

## 四、全面排查问题清单（其余页面）

### 信息架构 / 层级

- 任务面板（JobTray）只渲染前 20 条、超过静默隐藏 → 补「还有 N 个任务未显示」提示。

### 视觉一致性

- 三处卡片网格列宽不一致（330 / 310 / 320px）→ 统一为 **320px**（素材库、导出记录页）。

### i18n 硬编码

- 标注页步进按钮硬编码英文 `s` → 走 i18n（见第二节）。

### 保留项（经评估不改）

- 标注页时间轴使用的十六进制色（`#22c55e` / `#facc15` 等）是「标注时间轴 + 图例」内部自洽的一套语义色板，canvas 绘制与图例同时使用同一套值，强行映射到全局主题 token 会破坏标注颜色语义，故保留。
- 导出对话框兜底路径 `data\exports`、`CSV`、`fps` 等单位/路径为语言无关字面量，保留。

---

## 五、涉及文件

- `frontend/src/components/RallyPanel.tsx` — 信息架构重构
- `frontend/src/components/AnnotatePage.tsx` — 步进按钮 i18n、删除按钮 aria
- `frontend/src/components/Timeline.tsx` — 删除图标区分、撤销/重做禁用态
- `frontend/src/components/AnalysisDialog.tsx` — 切分依据改 Segmented
- `frontend/src/components/ImportVideoButton.tsx` — 模式图标区分
- `frontend/src/components/Player.tsx` — 播放/全屏 aria
- `frontend/src/components/ExportsPage.tsx` — 刷新 loading、播放 aria、网格统一
- `frontend/src/components/LibraryPage.tsx` — 创建 loading、网格统一
- `frontend/src/components/JobTray.tsx` — 任务溢出提示
- `frontend/src/components/StudioPage.tsx` — 素材删除 aria、全选 active
- `frontend/src/i18n/catalog/zh.ts` / `en.ts` — 新增 `rally.settings`、`job.moreHidden`、`exports.play`
- `frontend/src/i18n/catalog/fragments/annotate.ts` — 新增 `annotate.stepBack/stepForward`
- `frontend/src/i18n/catalog/fragments/player.ts` — 新增 `player.play/player.pause`
