import { AnimatePresence, motion } from 'motion/react'
import { useMemo, useState } from 'react'
import { Sparkles, Activity, Wand2, AlertTriangle, Video, Crosshair, Ruler } from 'lucide-react'
import { cn, humanDuration } from '../lib/format'
import { api } from '../lib/api'
import type { AnalysisParams } from '../lib/types'
import { Button, Modal, Progress, SectionTitle, Slider, Toggle } from './ui'
import SizeFilterPanel from './SizeFilterPanel'
import { WEIGHT_PRESETS_MAP, useStore } from '../store/useStore'

const STAGE_LABEL: Record<string, string> = {
  prepare: '准备派生资源',
  proxy: '生成代理视频',
  audio: '提取音轨',
  calibrate: '标定场地与机位',
  hits: '检测击球声',
  motion: '分析画面运动',
  players: '检测与跟踪球员',
  shuttle: '跟踪羽毛球',
  segment: '融合信号并切分回合',
  score: '提取特征并评分',
  done: '完成',
  error: '出错',
}

const VIEWPOINTS: { value: AnalysisParams['viewpoint']; label: string; hint: string }[] = [
  { value: 'auto', label: '自动识别', hint: '标定场地后按球员分布判断，推荐' },
  { value: 'rear', label: '场地后方', hint: '底线后拍，最常见的机位' },
  { value: 'side', label: '边线侧方', hint: '从场地侧面拍，球员左右跑动' },
  { value: 'elevated', label: '高机位斜俯', hint: '看台 / 二楼往下拍' },
  { value: 'overhead', label: '正俯拍', hint: '垂直向下，球场几乎不变形' },
]

export default function AnalysisDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const params = useStore((s) => s.params)
  const setParams = useStore((s) => s.setParams)
  const runAnalysis = useStore((s) => s.runAnalysis)
  const media = useStore((s) => s.currentMedia())
  const analysis = useStore((s) => s.currentAnalysis())
  const jobs = useStore((s) => s.jobs)
  const weights = useStore((s) => s.weights)
  const manualPoly = useStore((s) => s.currentCourtPoly())
  const openCourtEditor = useStore((s) => s.setCourtEditorOpen)
  const [advanced, setAdvanced] = useState(false)

  const activeJob = useMemo(() => {
    return Object.values(jobs)
      .filter((j) => j.kind === 'analyze')
      .sort((a, b) => b.created_at - a.created_at)
      .find((j) => j.status === 'running' || j.status === 'queued')
  }, [jobs])

  const lastJob = useMemo(
    () => Object.values(jobs).filter((j) => j.kind === 'analyze').sort((a, b) => b.created_at - a.created_at)[0],
    [jobs],
  )

  const done = analysis?.status === 'done'

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="AI 自动分析"
      subtitle={media ? `${media.name} · ${humanDuration(media.duration)} · ${media.width}×${media.height}` : undefined}
      width={720}
      footer={
        <div className="flex items-center justify-between gap-3">
          <div className="text-[11px] text-ink-500">
            {media && media.width >= 3000 ? (
              <span className="inline-flex items-center gap-1.5 text-amber-glow">
                <AlertTriangle size={12} /> 4K 长视频首轮分析较慢（会先生成低分辨率代理）
              </span>
            ) : (
              '分析在本机进行，视频不会上传到任何服务器'
            )}
          </div>
          <div className="flex gap-2">
            <Button variant="ghost" onClick={onClose}>
              关闭
            </Button>
            <Button
              variant="primary"
              loading={!!activeJob}
              disabled={!media || !!activeJob}
              onClick={async () => {
                await runAnalysis()
              }}
            >
              <Sparkles size={14} />
              {done ? '重新分析' : '开始分析'}
            </Button>
          </div>
        </div>
      }
    >
      {/* 进行中 */}
      <AnimatePresence>
        {activeJob && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mb-4 overflow-hidden"
          >
            <div className="panel-flat border-court-500/25 bg-court-500/[0.07] px-4 py-3.5">
              <div className="flex items-center gap-2">
                <span className="spin h-3.5 w-3.5 rounded-full border-[2px] border-court-400 border-t-transparent" />
                <span className="text-[12.5px] font-medium text-court-200">
                  {STAGE_LABEL[activeJob.stage] || activeJob.stage || '处理中'}
                </span>
                <span className="mono ml-auto text-[12px] text-court-300">
                  {Math.round(activeJob.progress * 100)}%
                </span>
              </div>
              <Progress value={activeJob.progress} className="mt-2.5" />
              <div className="mt-2 text-[11px] text-ink-400">{activeJob.message}</div>
              <Button
                variant="ghost"
                size="sm"
                className="mt-2"
                onClick={() => api.cancelJob(activeJob.id)}
              >
                取消分析
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {lastJob?.status === 'error' && !activeJob && (
        <div className="mb-4 rounded-xl border border-rose-hot/30 bg-rose-hot/10 px-4 py-3">
          <div className="flex items-center gap-2 text-[12.5px] font-medium text-rose-hot">
            <AlertTriangle size={13} /> 上次分析失败
          </div>
          <div className="mono mt-1.5 max-h-[120px] overflow-y-auto text-[10.5px] whitespace-pre-wrap text-rose-hot/80">
            {lastJob.error || lastJob.message}
          </div>
        </div>
      )}

      {done && analysis && (
        <div className="mb-4 grid grid-cols-4 gap-2">
          {[
            ['回合数', analysis.stats?.count ?? 0],
            ['有效时长', humanDuration(analysis.stats?.active_duration ?? 0)],
            ['总拍数', analysis.stats?.total_shots ?? 0],
            ['平均分', (analysis.stats?.avg_score ?? 0).toFixed(1)],
          ].map(([k, v]) => (
            <div key={k as string} className="panel-flat px-3 py-2">
              <div className="text-[10.5px] text-ink-400">{k}</div>
              <div className="tabular mt-0.5 text-[15px] font-semibold text-white">{v}</div>
            </div>
          ))}
        </div>
      )}

      {/* 分析模块 */}
      <SectionTitle>
        <Wand2 size={12} /> 分析模块
      </SectionTitle>
      <div className="mb-4 grid gap-1.5 md:grid-cols-2">
        <Toggle
          checked={params.use_audio}
          onChange={(v) => setParams({ use_audio: v })}
          label="击球声检测"
          hint="用球拍触球的中高频瞬态定位每一拍"
        />
        <Toggle
          checked={params.use_motion}
          onChange={(v) => setParams({ use_motion: v })}
          label="画面运动分析"
          hint="场地内运动能量，判断回合起止"
        />
        <Toggle
          checked={params.use_players}
          onChange={(v) => setParams({ use_players: v })}
          label="球员检测与跟踪"
          hint="YOLO + 多目标跟踪，最可靠的信号"
        />
        <Toggle
          checked={params.use_pose}
          onChange={(v) => setParams({ use_pose: v })}
          label="姿态辅助（击球归属）"
          hint="用 YOLO-pose 判断每一记击球声是不是我们这场比赛打的。多球场球馆里音频分不清是谁在击球，这一步能让回合终点收得更准。失败会自动降级"
        />
        <Toggle
          checked={params.use_shuttle}
          onChange={(v) => setParams({ use_shuttle: v })}
          label="羽毛球轨迹跟踪"
          hint="背景建模 + 抛物线约束，估计球速与击球时刻。计算量大，长视频建议先只分析片段"
        />
      </div>

      {params.use_shuttle && (
        <div className="mb-4 flex items-center gap-3 rounded-xl border border-amber-glow/25 bg-amber-glow/[0.07] px-3.5 py-3">
          <AlertTriangle size={14} className="shrink-0 text-amber-glow" />
          <div className="min-w-0 flex-1">
            <div className="text-[11.5px] leading-relaxed text-amber-glow/90">
              轨迹跟踪的耗时大致和「时长 × 分辨率」成正比。设置一个时间预算可以控制总耗时；
              覆盖不足 60% 时该路信号会被自动忽略（只分析一部分会把融合结果带偏）。
            </div>
            <div className="mt-2 max-w-[320px]">
              <Slider
                label="时间预算"
                value={params.shuttle_budget_seconds}
                min={60}
                max={1800}
                step={30}
                onChange={(v) => setParams({ shuttle_budget_seconds: v })}
                format={(v) => (v >= 1800 ? '全片' : `${Math.round(v / 60)} 分钟`)}
              />
            </div>
          </div>
        </div>
      )}

      {/* 机位与场地标定 */}
      <SectionTitle>
        <Video size={12} /> 机位与场地
      </SectionTitle>
      <div className="mb-4 rounded-xl border border-white/8 bg-white/[0.02] px-3.5 py-3">
        <div className="text-[11.5px] leading-relaxed text-ink-400">
          不同拍法下「谁离相机近」「球员在画面里该多大」完全不同。分析会先从画面里
          自动找出球场范围（颜色 + 多边形，弯边素材会自动多取几个点）并消掉看台与
          其他场地的人，再据此判断机位；固定用一个机位的话也可以直接指定，省掉猜测。
        </div>
        <div className="mt-2.5 flex flex-wrap gap-1.5">
          {VIEWPOINTS.map((v) => (
            <button
              key={v.value}
              onClick={() => setParams({ viewpoint: v.value })}
              title={v.hint}
              className={cn(
                'rounded-lg border px-2.5 py-1.5 text-left transition-all',
                params.viewpoint === v.value
                  ? 'border-court-500/50 bg-court-500/12'
                  : 'border-white/8 bg-white/[0.025] hover:border-white/18',
              )}
            >
              <div className={cn('text-[11.5px] font-medium',
                params.viewpoint === v.value ? 'text-court-200' : 'text-ink-100')}>
                {v.label}
              </div>
              <div className="mt-0.5 text-[10px] text-ink-500">{v.hint}</div>
            </button>
          ))}
        </div>
        <div className="mt-2.5 flex flex-wrap items-center gap-x-4">
          <Toggle
            checked={params.auto_calibrate}
            onChange={(v) => setParams({ auto_calibrate: v })}
            label="自动标定场地"
            hint="找出球场范围并只在场内找人；关掉则退回全画幅（多球场或颜色异常时可用）"
          />
          <Button variant="ghost" size="sm" onClick={() => openCourtEditor(true)}>
            <Crosshair size={13} />
            {manualPoly ? `重新手动画场地（${manualPoly.length} 点）` : '手动标定场地'}
          </Button>
          {manualPoly && (
            <span className="text-[11px] text-court-300">
              已使用手动标定（优先于自动识别）
              {manualPoly.length > 4 && ` · ${manualPoly.length} 点多边形`}
            </span>
          )}
        </div>
        <div className="mt-2.5 max-w-[360px]">
          <Slider
            label="切分依据"
            value={params.segment_mode === 'activity' ? 1 : params.segment_mode === 'hybrid' ? 2 : 0}
            min={0}
            max={2}
            step={1}
            onChange={(v) =>
              setParams({ segment_mode: v === 0 ? 'auto' : v === 1 ? 'activity' : 'hybrid' })
            }
            format={(v) => ['球员运动（推荐）', '画面活跃度', '两者对比择优'][v] ?? ''}
            hint="球员运动的静默段就是回合边界；画面活跃度会被观众和隔壁场地干扰"
          />
        </div>
      </div>

      {done && analysis?.calibration && (
        <div className="mb-4 rounded-xl border border-white/8 bg-white/[0.02] px-3.5 py-3">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11.5px]">
            <span className="text-ink-400">识别机位</span>
            <span className="font-medium text-court-200">
              {analysis.calibration.viewpoint_label}
            </span>
            {analysis.calibration.ok && (
              <span className="text-ink-500">
                场地色 {analysis.calibration.court_color} · 占画面{' '}
                {Math.round((analysis.calibration.court_area_ratio ?? 0) * 100)}%
                {(analysis.calibration.point_count ?? 4) > 4 &&
                  ` · 边界 ${analysis.calibration.point_count} 点多边形`}
                {analysis.calibration.source === 'manual' && ' · 手动标定'}
              </span>
            )}
          </div>
          {(analysis.calibration.notes?.length ?? 0) > 0 && (
            <ul className="mt-1.5 space-y-0.5">
              {analysis.calibration.notes.map((n, i) => (
                <li key={i} className="text-[11px] leading-relaxed text-ink-500">
                  · {n}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* 人物框尺寸筛选 */}
      <SectionTitle>
        <Ruler size={12} /> 找谁：人物框尺寸
      </SectionTitle>
      <SizeFilterPanel />

      {/* 评分口径 */}
      <SectionTitle>
        <Sparkles size={12} /> 评分口径
      </SectionTitle>
      <div className="mb-4 flex flex-wrap gap-1.5">
        {WEIGHT_PRESETS_MAP.map((w) => (
          <button
            key={w.value}
            onClick={() => useStore.getState().setWeights(w.value)}
            className={cn(
              'rounded-lg border px-3 py-2 text-left transition-all',
              weights === w.value
                ? 'border-court-500/50 bg-court-500/12'
                : 'border-white/8 bg-white/[0.025] hover:border-white/18',
            )}
          >
            <div className={cn('text-[12px] font-medium', weights === w.value ? 'text-court-200' : 'text-ink-100')}>
              {w.label}
            </div>
            <div className="mt-0.5 text-[10.5px] text-ink-400">{w.hint}</div>
          </button>
        ))}
      </div>

      {/* 关键参数 */}
      <SectionTitle right={
        <button onClick={() => setAdvanced((v) => !v)} className="text-[11px] text-court-300 hover:underline">
          {advanced ? '收起高级参数' : '展开高级参数'}
        </button>
      }>
        <Activity size={12} /> 关键参数
      </SectionTitle>
      <div className="grid gap-x-5 gap-y-1 md:grid-cols-2">
        <Slider
          label="回合间隔判定"
          value={params.gap_seconds}
          min={1}
          max={8}
          step={0.1}
          onChange={(v) => setParams({ gap_seconds: v })}
          format={(v) => `${v.toFixed(1)}s`}
          hint="静默超过这个时长就认为回合结束"
        />
        <Slider
          label="最短回合时长"
          value={params.min_rally_seconds}
          min={0.5}
          max={12}
          step={0.5}
          onChange={(v) => setParams({ min_rally_seconds: v })}
          format={(v) => `${v}s`}
          hint="短于这个时长的片段会被丢弃"
        />
        <Slider
          label="片段前留白"
          value={params.pre_roll}
          min={0}
          max={4}
          step={0.1}
          onChange={(v) => setParams({ pre_roll: v })}
          format={(v) => `${v.toFixed(1)}s`}
          hint="发球前保留的准备动作"
        />
        <Slider
          label="死球余量"
          value={params.hit_tail_seconds}
          min={0.2}
          max={2.5}
          step={0.1}
          onChange={(v) => setParams({ hit_tail_seconds: v })}
          format={(v) => `${v.toFixed(1)}s`}
          hint="最后一拍打出去之后，球还要飞这么久才落地。回合终点 = 最后一拍 + 这个值。调大更保险，调小更紧凑（可能吃掉高远球的落地瞬间）"
        />
        <Slider
          label="片段后留白"
          value={params.post_roll}
          min={0}
          max={4}
          step={0.1}
          onChange={(v) => setParams({ post_roll: v })}
          format={(v) => `${v.toFixed(1)}s`}
          hint="终点之后再留一点收尾，只影响导出片段的长度"
        />

        {advanced && (
          <>
            <Slider
              label="击球检测灵敏度"
              value={params.hit_sensitivity}
              min={0}
              max={1}
              step={0.05}
              onChange={(v) => setParams({ hit_sensitivity: v })}
              format={(v) => v.toFixed(2)}
              hint="越高越容易检出轻击，但更容易被环境噪声干扰"
            />
            <Slider
              label="最长回合时长"
              value={params.max_rally_seconds}
              min={10}
              max={240}
              step={5}
              onChange={(v) => setParams({ max_rally_seconds: v })}
              format={(v) => `${v}s`}
              hint="超过会被自动切分，避免两个回合粘连"
            />
            <Slider
              label="分析帧率"
              value={params.sample_fps}
              min={5}
              max={30}
              step={1}
              onChange={(v) => setParams({ sample_fps: v })}
              format={(v) => `${v} fps`}
              hint="越高越准，速度越慢"
            />
          </>
        )}
      </div>
    </Modal>
  )
}
