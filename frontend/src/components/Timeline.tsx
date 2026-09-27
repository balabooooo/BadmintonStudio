import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import {
  ZoomIn,
  ZoomOut,
  Scissors,
  Trash2,
  Eraser,
  Magnet,
  ChevronsLeftRight,
  Plus,
  Undo2,
  Redo2,
  Sparkles,
  Clapperboard,
} from 'lucide-react'
import { cn, scoreColor, timecode } from '../lib/format'
import { Button, ContextMenu, Tooltip } from './ui'
import { useStore } from '../store/useStore'
import { useT } from '../i18n/useT'

const RULER_H = 26
const LANE_H = 30
const TRACK_H = 54
const PAD_RIGHT = 320
/** 时间线至少要有这么高，三条带子（标尺 + 回合带 + 成片轨）才不会被截断 */
export const TIMELINE_MIN_H = RULER_H + LANE_H + TRACK_H + 58

type DragState =
  | { kind: 'none' }
  | { kind: 'playhead' }
  | { kind: 'move'; clipId: string; grabOffset: number; origStart: number }
  | { kind: 'trim-in'; clipId: string; origIn: number; origOut: number; origStart: number; startX: number }
  | { kind: 'trim-out'; clipId: string; origIn: number; origOut: number; startX: number }

export default function Timeline() {
  const project = useStore((s) => s.project)
  const analysis = useStore((s) => s.currentAnalysis())
  const zoom = useStore((s) => s.zoom)
  const setZoom = useStore((s) => s.setZoom)
  const currentTime = useStore((s) => s.currentTime)
  const seek = useStore((s) => s.seek)
  const selectedClipId = useStore((s) => s.selectedClipId)
  const selectedClipIds = useStore((s) => s.selectedClipIds)
  const selectClip = useStore((s) => s.selectClip)
  const selectedRallyId = useStore((s) => s.selectedRallyId)
  const selectRally = useStore((s) => s.selectRally)
  const setPreviewMode = useStore((s) => s.setPreviewMode)
  const updateClip = useStore((s) => s.updateClip)
  const removeClips = useStore((s) => s.removeClips)
  const splitClipAt = useStore((s) => s.splitClipAt)
  const splitClipAtSourceTime = useStore((s) => s.splitClipAtSourceTime)
  const previewMode = useStore((s) => s.previewMode)
  const reorder = useStore((s) => s.reorderTrack)
  const undo = useStore((s) => s.undo)
  const redo = useStore((s) => s.redo)
  const canUndo = useStore((s) => s.history.length > 0)
  const canRedo = useStore((s) => s.future.length > 0)
  const clearTimeline = useStore((s) => s.clearTimeline)
  const addClipFromRally = useStore((s) => s.addClipFromRally)
  const setUserSeeking = useStore((s) => s.setUserSeeking)
  const pushHistory = useStore((s) => s.pushHistory)
  const tr = useT()

  const scrollRef = useRef<HTMLDivElement>(null)
  const railRef = useRef<HTMLDivElement>(null)
  const [snap, setSnap] = useState(true)
  const [drag, setDrag] = useState<DragState>({ kind: 'none' })
  const [menu, setMenu] = useState<{ x: number; y: number; clipId: string } | null>(null)
  // 一次拖动只在第一次真正移动时压入撤销栈，单击选中不再产生空的撤销记录
  const movedRef = useRef(false)

  const timeline = project?.timeline
  const track = timeline?.tracks?.[0]
  const clips = useMemo(() => (track?.clips ?? []).slice().sort((a, b) => a.tl_start - b.tl_start), [track])
  const media = useStore((s) => s.currentMedia())
  const hasRallyLane = !!analysis && analysis.rallies.length > 0

  const contentDuration = useMemo(() => {
    const end = clips.reduce((m, c) => Math.max(m, c.tl_start + (c.src_out - c.src_in) / c.speed), 0)
    return Math.max(end, media?.duration ?? 0, 30)
  }, [clips, media])

  const width = contentDuration * zoom + PAD_RIGHT

  const pxToTime = useCallback((px: number) => px / zoom, [zoom])
  const timeToPx = useCallback((t: number) => t * zoom, [zoom])

  /** 跳到某个「原片时刻」：成片模式下换算成成片时间，不在成片里就切回源片 */
  const seekSource = useCallback(
    (src: number) => {
      if (previewMode !== 'timeline') {
        seek(src)
        return
      }
      const c = clips.find((x) => src >= x.src_in && src <= x.src_out)
      if (c) {
        seek(c.tl_start + (src - c.src_in) / c.speed)
        return
      }
      setPreviewMode('source')
      seek(src)
    },
    [previewMode, clips, seek, setPreviewMode],
  )

  const snapPoints = useMemo(() => {
    if (!snap) return []
    // 吸附点必须和正在拖动的坐标同一套：片段位置是「成片时间」，而回合
    // start/end 是「原片时间」、源片模式下 currentTime 也是原片时间。
    // 混在一起会让片段一经过回合边界就莫名其妙地跳一下。
    const pts = [0]
    if (previewMode === 'timeline') pts.push(currentTime)
    clips.forEach((c) => {
      // 排除正在拖的片段自身：否则它的当前边缘一直在吸附容忍范围内，
      // 拖动会被自己「吸住」，表现为每 8px 卡一下、根本拖不快。
      if (drag.kind === 'move' && drag.clipId === c.id) return
      pts.push(c.tl_start, c.tl_start + (c.src_out - c.src_in) / c.speed)
    })
    return pts
  }, [snap, currentTime, previewMode, clips, drag])

  const applySnap = useCallback(
    (t: number, tolPx = 8) => {
      if (!snap) return t
      const tol = tolPx / zoom
      let best = t
      let bestD = Infinity
      for (const p of snapPoints) {
        const d = Math.abs(p - t)
        if (d < tol && d < bestD) {
          bestD = d
          best = p
        }
      }
      return best
    },
    [snap, snapPoints, zoom],
  )

  // ------------------------------------------------------ 指针交互
  const onPointerMove = useCallback(
    (e: PointerEvent) => {
      if (!scrollRef.current) return
      const rect = scrollRef.current.getBoundingClientRect()
      const x = e.clientX - rect.left + scrollRef.current.scrollLeft
      const t = pxToTime(x)

      // 第一次真正移动时才存撤销快照：单击选中不该产生一条空的撤销记录
      const beginDrag = () => {
        if (!movedRef.current) {
          pushHistory()
          movedRef.current = true
        }
      }
      if (drag.kind === 'playhead') {
        seek(Math.max(0, t))
      } else if (drag.kind === 'move') {
        let ns = Math.max(0, pxToTime(x - drag.grabOffset))
        ns = applySnap(ns)
        beginDrag()
        updateClip(drag.clipId, { tl_start: Number(ns.toFixed(3)) }, false)
      } else if (drag.kind === 'trim-in') {
        const c = clips.find((v) => v.id === drag.clipId)
        if (!c) return
        // 按片段自己所属素材的长度裁剪：跨素材成片时用当前素材会裁错边界
        const lim = project?.media.find((m) => m.id === c.media_id)?.duration ?? media?.duration ?? 1e9
        const delta = (e.clientX - drag.startX) / zoom
        const nIn = Math.max(0, Math.min(drag.origOut - 0.2, Math.min(lim - 0.2, drag.origIn + delta * c.speed)))
        // 左端裁剪必须同时挪 tl_start：片段占位是 [tl_start, tl_start+(src_out-src_in)/speed]，
        // 只改 src_in 会让右端跟着缩、左把手不跟手，看起来像在裁尾部。
        const newStart = Math.max(0, drag.origStart + (nIn - drag.origIn) / c.speed)
        beginDrag()
        updateClip(drag.clipId, { src_in: Number(nIn.toFixed(3)), tl_start: Number(newStart.toFixed(3)) }, false)
      } else if (drag.kind === 'trim-out') {
        const c = clips.find((v) => v.id === drag.clipId)
        if (!c) return
        const lim = project?.media.find((m) => m.id === c.media_id)?.duration ?? media?.duration ?? 1e9
        const delta = (e.clientX - drag.startX) / zoom
        const nOut = Math.min(lim, Math.max(drag.origIn + 0.2, drag.origOut + delta * c.speed))
        beginDrag()
        updateClip(drag.clipId, { src_out: Number(nOut.toFixed(3)) }, false)
      }
    },
    [drag, pxToTime, seek, applySnap, updateClip, clips, zoom, media, pushHistory, project],
  )

  const onPointerUp = useCallback(() => {
    setDrag({ kind: 'none' })
    // 松手后仍要给 video 一点时间追上目标时间，所以 setUserSeeking(false)
    // 内部会再留一个几百毫秒的屏蔽窗口
    setUserSeeking(false)
  }, [setUserSeeking])

  useEffect(() => {
    if (drag.kind === 'none') return
    window.addEventListener('pointermove', onPointerMove)
    window.addEventListener('pointerup', onPointerUp)
    window.addEventListener('pointercancel', onPointerUp)
    document.body.classList.add('grabbing')
    // 拖播放头期间告诉全局：别再让 video 的 timeupdate 回写时间
    if (drag.kind === 'playhead') setUserSeeking(true)
    return () => {
      window.removeEventListener('pointermove', onPointerMove)
      window.removeEventListener('pointerup', onPointerUp)
      window.removeEventListener('pointercancel', onPointerUp)
      document.body.classList.remove('grabbing')
    }
  }, [drag, onPointerMove, onPointerUp, setUserSeeking])

  // Ctrl+滚轮缩放。React 的 onWheel 走的是 passive 监听，preventDefault 无效，
  // 页面会跟着一起缩放，所以改用原生监听 + { passive: false }。
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const handler = (e: WheelEvent) => {
      if (!(e.ctrlKey || e.metaKey)) return
      e.preventDefault()
      const z = useStore.getState().zoom
      setZoom(z * (e.deltaY < 0 ? 1.14 : 0.88))
    }
    el.addEventListener('wheel', handler, { passive: false })
    return () => el.removeEventListener('wheel', handler)
  }, [setZoom])

  // 自动滚动跟随播放头。
  // 必须在拖动期间关闭：拖动时 currentTime 跟着鼠标变，一旦播放头靠近视口
  // 边缘就会触发 scrollLeft 调整，而下一帧 pointermove 是用「clientX - 容器左边
  // + scrollLeft」换算内容坐标的——滚动一变，同一个鼠标位置就对应到另一个时间，
  // 播放头被甩走，看起来就是「拖一下就跳」。
  useEffect(() => {
    if (drag.kind !== 'none') return
    const el = scrollRef.current
    if (!el) return
    const x = timeToPx(currentTime)
    const left = el.scrollLeft
    const right = left + el.clientWidth - 120
    if (x < left + 60 || x > right) {
      el.scrollLeft = Math.max(0, x - el.clientWidth * 0.35)
    }
  }, [currentTime, timeToPx, drag.kind])

  // 新选中的片段如果在视口外就滚过去。
  // 「加入时间线」是把片段追加到成片轨末尾，不滚过去的话点完那一下屏幕上什么都没变，
  // 很容易以为没生效。
  const lastClipRef = useRef<string | null>(null)
  useEffect(() => {
    if (!selectedClipId) {
      lastClipRef.current = null
      return
    }
    if (lastClipRef.current === selectedClipId) return
    // 拖动 / 菜单打开时先别滚：拖动时自动滚动会把播放头甩走，菜单则会被滚动事件关掉。
    // 必须在这些早退之后再写 lastClipRef——否则本次没滚成，之后同 id 就永远不滚了。
    if (drag.kind !== 'none' || menu) return
    lastClipRef.current = selectedClipId
    const el = scrollRef.current
    const c = clips.find((v) => v.id === selectedClipId)
    if (!el || !c) return
    const left = timeToPx(c.tl_start)
    const right = left + Math.max(6, timeToPx((c.src_out - c.src_in) / c.speed))
    if (left < el.scrollLeft + 40 || right > el.scrollLeft + el.clientWidth - 40) {
      el.scrollLeft = Math.max(0, left - el.clientWidth * 0.3)
    }
  }, [selectedClipId, clips, timeToPx, drag.kind, menu])

  if (!project) return null

  return (
    <div className="flex h-full min-h-0 flex-col border-t border-white/8 bg-ink-950/45">
      {/* 工具栏 */}
      <div className="flex items-center gap-1.5 border-b border-white/7 px-3 py-1.5">
        <span className="mr-1 shrink-0 text-[10.5px] font-semibold tracking-[0.14em] whitespace-nowrap text-ink-400 uppercase">
          {tr('timeline.title')}
        </span>
        <Tooltip content={tr('timeline.undoHint')} kbd="Ctrl+Z">
          <Button variant="ghost" size="icon" onClick={undo} disabled={!canUndo}>
            <Undo2 size={13} />
          </Button>
        </Tooltip>
        <Tooltip content={tr('timeline.redoHint')} kbd="Ctrl+Y">
          <Button variant="ghost" size="icon" onClick={redo} disabled={!canRedo}>
            <Redo2 size={13} />
          </Button>
        </Tooltip>
        <div className="mx-1 h-4 w-px bg-white/10" />
        <Tooltip content={tr('timeline.splitAtPlayheadHint')} kbd="S">
          <Button
            variant="ghost"
            size="icon"
            disabled={!selectedClipId}
            onClick={() => {
              if (!selectedClipId) return
              // 源片模式下播放头是原片时间，必须换算，否则会切错位置
              if (previewMode === 'source') splitClipAtSourceTime(selectedClipId, currentTime)
              else splitClipAt(selectedClipId, currentTime)
            }}
          >
            <Scissors size={13} />
          </Button>
        </Tooltip>
        <Tooltip
          kbd="Delete"
          content={
            selectedClipIds.length > 1
              ? tr('timeline.deleteSelectedHint', { n: selectedClipIds.length })
              : tr('timeline.deleteClipHint')
          }
        >
          <Button
            variant="ghost"
            size="icon"
            className="relative"
            disabled={!selectedClipIds.length}
            onClick={() => removeClips(selectedClipIds)}
          >
            <Trash2 size={13} />
            {selectedClipIds.length > 1 && (
              <span className="absolute -top-1 -right-1 grid h-[13px] min-w-[13px] place-items-center rounded-full bg-rose-hot px-[3px] text-[8.5px] leading-none font-bold text-white">
                {selectedClipIds.length}
              </span>
            )}
          </Button>
        </Tooltip>
        <Tooltip content={tr('timeline.joinClipsHint')}>
          <Button variant="ghost" size="icon" onClick={reorder} disabled={!clips.length}>
            <ChevronsLeftRight size={13} />
          </Button>
        </Tooltip>
        <Tooltip content={snap ? tr('timeline.snapOn') : tr('timeline.snapOff')}>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setSnap((v) => !v)}
            className={snap ? 'text-court-300' : ''}
          >
            <Magnet size={13} />
          </Button>
        </Tooltip>

        <div className="flex-1" />

        <span className="mono mr-1 shrink-0 text-[10.5px] whitespace-nowrap text-ink-400">
          {tr('timeline.clipCountDuration', { n: clips.length, duration: timecode(contentDuration, false) })}
        </span>
        <Tooltip content={tr('timeline.zoomOutHint')}>
          <Button variant="ghost" size="icon" onClick={() => setZoom(zoom * 0.8)}>
            <ZoomOut size={13} />
          </Button>
        </Tooltip>
        <input
          type="range"
          min={6}
          max={600}
          step={1}
          value={zoom}
          onChange={(e) => setZoom(parseFloat(e.target.value))}
          className="w-[110px]"
        />
        <Tooltip content={tr('timeline.zoomInHint')}>
          <Button variant="ghost" size="icon" onClick={() => setZoom(zoom * 1.25)}>
            <ZoomIn size={13} />
          </Button>
        </Tooltip>
        <Tooltip content={tr('timeline.clearTimeline')}>
          <Button variant="ghost" size="icon" onClick={clearTimeline} disabled={!clips.length}>
            <Eraser size={13} className="text-rose-hot/75" />
          </Button>
        </Tooltip>
      </div>

      {/* 主体 */}
      <div className="relative min-h-0 flex-1">
        {clips.length === 0 && (
          <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center">
            <div className="rounded-xl border border-dashed border-white/12 bg-ink-900/70 px-5 py-4 text-center backdrop-blur">
              <div className="text-[12.5px] text-ink-200">{tr('timeline.emptyTitle')}</div>
              <div className="mt-1 text-[11.5px] text-ink-400">
                {tr('timeline.emptyHintBefore')}{' '}
                <Plus size={11} className="inline" />{' '}
                {tr('timeline.emptyHintAfter')}
                <br />
                {tr('timeline.emptyHintSecond')}
              </div>
            </div>
          </div>
        )}

        <div className="flex h-full min-h-0">
          {/*
            左侧固定窄栏：两条轨道的说明图标都放在这里。
            以前图标是塞在轨道内部的 sticky 元素：横向滚动时它会压在片段上，
            时间线被拖矮时又整块跑到可视区外（成片那个图标就是这么「消失」的）。
            放到滚动容器外面以后，它永远贴在左边、永远看得见。
          */}
          <div
            ref={railRef}
            className="flex w-[27px] shrink-0 flex-col overflow-hidden border-r border-white/7 bg-ink-950/45"
          >
            <div className="shrink-0 border-b border-white/7 bg-ink-900/92" style={{ height: RULER_H }} />
            {hasRallyLane && (
              <div
                className="flex shrink items-center justify-center"
                style={{ height: LANE_H, background: 'rgb(255 255 255 / 0.018)' }}
              >
                <LaneChip
                  icon={<Sparkles size={11} />}
                  title={tr('timeline.rallyLaneTitle')}
                  tone="#ffb020"
                  content={
                    <span>
                      <b className="text-amber-glow">{tr('timeline.rallyLaneTitle')}</b>
                      {'\n\n'}
                      {tr('timeline.rallyLaneIntroBefore')}
                      <b>{tr('timeline.rallyLaneIntroBold')}</b>
                      {tr('timeline.rallyLaneIntroAfter')}
                      {'\n\n'}
                      <b>{tr('timeline.rallyLaneColorBold')}</b>
                      {tr('timeline.rallyLaneColorAfter')}
                      {'\n\n'}
                      {tr('timeline.rallyLaneClick')}
                      {'\n'}
                      {tr('timeline.rallyLaneNote')}
                    </span>
                  }
                />
              </div>
            )}
            <div
              className="flex shrink items-center justify-center"
              style={{ height: TRACK_H, background: 'rgb(92 157 255 / 0.045)' }}
            >
              <LaneChip
                icon={<Clapperboard size={11} />}
                title={tr('timeline.exportTrackTitle')}
                tone="#5c9dff"
                content={
                  <span>
                    <b className="text-flux-400">{tr('timeline.exportTrackTitle')}</b>
                    {'\n\n'}
                    {tr('timeline.exportTrackIntroBefore')}
                    <b>{tr('timeline.exportTrackIntroBold')}</b>
                    {tr('timeline.exportTrackIntroAfter')}
                    {'\n\n'}
                    <b>{tr('timeline.exportTrackAddBold')}</b>
                    {tr('timeline.exportTrackAddAfter')}
                    {'\n\n'}
                    {tr('timeline.exportTrackEdit')}
                    {'\n'}
                    {tr('timeline.exportTrackSpeed')}
                    {'\n\n'}
                    {tr('timeline.exportTrackNote')}
                  </span>
                }
              />
            </div>
          </div>

          <div
            ref={scrollRef}
            className="h-full min-w-0 flex-1 overflow-x-auto overflow-y-auto"
            onScroll={() => {
              // 时间线被拖得很矮时纵向也能滚，图标栏跟着一起滚，免得和轨道错位
              const el = scrollRef.current
              if (el && railRef.current) railRef.current.scrollTop = el.scrollTop
            }}
          >
            <div className="relative" style={{ width, minWidth: '100%' }}>
              {/* 标尺 */}
              <div
                className="sticky top-0 z-30 cursor-ew-resize border-b border-white/7 bg-ink-900/92 backdrop-blur"
                style={{ height: RULER_H }}
                onPointerDown={(e) => {
                  // 右键不定位、也不进入拖播放头
                  if (e.button !== 0) return
                  // 用滚动容器（视口内固定不动）的矩形换算内容坐标，再加 scrollLeft。
                  // 不能用 ruler 自身：它横向随内容滚动，rect.left 已经减掉了 scrollLeft，
                  // 再加一次就多算一个 scrollLeft——滚得越远点击越靠后，直接跳到片尾。
                  const el = scrollRef.current
                  if (!el) return
                  const rect = el.getBoundingClientRect()
                  setDrag({ kind: 'playhead' })
                  seek(Math.max(0, pxToTime(e.clientX - rect.left + el.scrollLeft)))
                }}
                onContextMenu={(e) => e.preventDefault()}
              >
                <Ruler zoom={zoom} duration={contentDuration} />
              </div>

              {/* 源片回合标记带：把 AI 在原片时间轴上识别出的回合画出来 */}
              {hasRallyLane && (
                <div
                  className="relative border-b border-white/7"
                  style={{ height: LANE_H, background: 'rgb(255 255 255 / 0.018)' }}
                >
                {analysis.rallies.map((r) => {
                  const selected = r.id === selectedRallyId
                  const c = scoreColor(r.scores.total)
                  const w = Math.max(3, timeToPx(r.end - r.start))
                  return (
                    // 定位交给外层这个 grid，Tooltip 的锚点就只负责当色块的容器。
                    // 之前把 Tooltip 直接套在色块上：它的锚点是 inline-flex，会被前面
                    // 轨道标签那一块挤到带子下面一行，色块整个画到成片轨里、再被成片轨
                    // 盖住——所以这条带子看上去永远是空的。
                    <div
                      key={r.id}
                      className="absolute top-[7px] grid h-[16px]"
                      style={{ left: timeToPx(r.start), width: w }}
                    >
                      <Tooltip
                        width={300}
                        content={[
                          tr('timeline.rallyTooltipLine1', {
                            index: r.index,
                            score: r.scores.total.toFixed(0),
                            duration: r.duration.toFixed(1),
                            shots: r.features.shot_count,
                          }),
                          tr('timeline.rallyTooltipLine2', {
                            start: timecode(r.start, false),
                            end: timecode(r.end, false),
                          }),
                          tr('timeline.rallyTooltipLine3'),
                          tr('timeline.rallyTooltipLine4'),
                        ].join('\n')}
                      >
                        <div
                          className={cn(
                            'flex h-full w-full cursor-pointer items-center overflow-hidden rounded-[3px] transition-all',
                            selected ? 'ring-2 ring-white/70' : 'hover:brightness-125',
                          )}
                          style={{
                            background: `linear-gradient(180deg, ${c}, ${c.replace('rgb', 'rgba').replace(')', ' / 0.55)')})`,
                            opacity: r.keep ? 0.92 : 0.28,
                          }}
                          onClick={() => {
                            selectRally(r.id)
                            seekSource(r.start)
                          }}
                          onDoubleClick={() => addClipFromRally(r)}
                        >
                          {w > 26 && (
                            <span className="pointer-events-none truncate px-1 text-[9.5px] font-semibold text-ink-950/75">
                              #{r.index}
                              {w > 92 && ` · ${tr('timeline.scoreSuffix', { n: r.scores.total.toFixed(0) })}`}
                            </span>
                          )}
                        </div>
                      </Tooltip>
                    </div>
                  )
                })}
              </div>
            )}

            {/* 轨道 */}
            <div
              className="relative"
              style={{ height: TRACK_H, background: 'rgb(92 157 255 / 0.045)' }}
              onContextMenu={(e) => e.preventDefault()}
            >
              <div className="absolute inset-x-0 top-[18px] bottom-1.5 rounded-md bg-white/[0.022]" />
              {clips.map((c) => {
                const dur = (c.src_out - c.src_in) / c.speed
                const left = timeToPx(c.tl_start)
                const w = Math.max(6, timeToPx(dur))
                const selected = selectedClipIds.includes(c.id)
                const rally = analysis?.rallies.find((r) => r.id === c.rally_id)
                const col = rally ? scoreColor(rally.scores.total) : '#3b7ff0'
                return (
                  <div
                    key={c.id}
                    className={cn(
                      'clip group absolute top-[18px] bottom-1.5 overflow-hidden rounded-md border border-black/40',
                      selected && 'clip-selected',
                    )}
                    style={{
                      left,
                      width: w,
                      background: `linear-gradient(180deg, ${col.replace('rgb', 'rgba').replace(')', ' / 0.5)')}, ${col
                        .replace('rgb', 'rgba')
                        .replace(')', ' / 0.24)')})`,
                    }}
                    onPointerDown={(e) => {
                      // 只响应左键：右键是「打开菜单」，不该顺带选中/进入拖拽状态，
                      // 否则按住右键拖一下就把片段挪走了
                      if (e.button !== 0) return
                      e.stopPropagation()
                      // Ctrl/Cmd = 点选累加，Shift = 连选；这两种只改选区，不进入拖拽，
                      // 否则用修饰键多选时一按下去片段就被拖走了。
                      const mode = e.shiftKey ? 'range' : e.ctrlKey || e.metaKey ? 'toggle' : 'replace'
                      selectClip(c.id, mode)
                      if (mode !== 'replace') return
                      // 撤销快照留到第一次真正 pointermove 时再存：只单击选中的话，
                      // 以前也会压一条「什么都不发生」的撤销记录，还会清空重做栈。
                      movedRef.current = false
                      const rect = scrollRef.current!.getBoundingClientRect()
                      const x = e.clientX - rect.left + scrollRef.current!.scrollLeft
                      const grab = x - left
                      if (grab < 10 && w > 26) {
                        setDrag({ kind: 'trim-in', clipId: c.id, origIn: c.src_in, origOut: c.src_out, origStart: c.tl_start, startX: e.clientX })
                      } else if (w - grab < 10 && w > 26) {
                        setDrag({ kind: 'trim-out', clipId: c.id, origIn: c.src_in, origOut: c.src_out, startX: e.clientX })
                      } else {
                        setDrag({ kind: 'move', clipId: c.id, grabOffset: grab, origStart: c.tl_start })
                      }
                    }}
                    onContextMenu={(e) => {
                      e.preventDefault()
                      e.stopPropagation()
                      // 右键已在选区里的片段时保留多选，菜单里的删除就会作用在整个选区上
                      if (!selectedClipIds.includes(c.id)) selectClip(c.id)
                      setMenu({ x: e.clientX, y: e.clientY, clipId: c.id })
                    }}
                    onDoubleClick={() => setPreviewMode('timeline')}
                  >
                    <div className="pointer-events-none flex h-full flex-col justify-center px-1.5">
                      <div className="truncate text-[10.5px] font-medium text-white/95">{c.label || tr('timeline.clipFallbackLabel')}</div>
                      <div className="mono truncate text-[9.5px] text-white/65">
                        {timecode(c.src_in, false)} → {timecode(c.src_out, false)}
                        {c.speed !== 1 && ` · ${c.speed}×`}
                      </div>
                    </div>
                    {/* 波形条（占位视觉） */}
                    <div className="pointer-events-none absolute inset-x-0 bottom-0 h-[9px] opacity-45">
                      <WaveBars seed={c.id} />
                    </div>
                    <div className="absolute inset-y-0 left-0 w-[6px] cursor-w-resize bg-white/0 group-hover:bg-white/25" />
                    <div className="absolute inset-y-0 right-0 w-[6px] cursor-e-resize bg-white/0 group-hover:bg-white/25" />
                  </div>
                )
              })}
            </div>

            {/* 播放头 */}
            <div
              className="pointer-events-none absolute top-0 bottom-0 z-40 w-px bg-court-300"
              style={{ left: timeToPx(currentTime) }}
            >
              <div className="absolute -top-0 -left-[5px] h-[9px] w-[11px] rounded-b-[3px] bg-court-300 shadow-[0_0_10px_rgb(56_224_162/0.8)]" />
            </div>
            </div>
          </div>
        </div>
      </div>

      {menu && (
        <ContextMenu
          x={menu.x}
          y={menu.y}
          onClose={() => setMenu(null)}
          items={[
            {
              label: tr('timeline.menuSplit'),
              hint: 'S',
              onClick: () => {
                if (previewMode === 'source') splitClipAtSourceTime(menu.clipId, currentTime)
                else splitClipAt(menu.clipId, currentTime)
              },
            },
            {
              label: tr('timeline.menuSpeedHalf'),
              onClick: () => updateClip(menu.clipId, { speed: 0.5 }),
            },
            {
              label: tr('timeline.menuSpeedDouble'),
              onClick: () => updateClip(menu.clipId, { speed: 2 }),
            },
            {
              label: tr('timeline.menuSpeedNormal'),
              onClick: () => updateClip(menu.clipId, { speed: 1 }),
            },
            {
              label:
                selectedClipIds.length > 1 && selectedClipIds.includes(menu.clipId)
                  ? tr('timeline.menuDeleteSelected', { n: selectedClipIds.length })
                  : tr('timeline.menuDelete'),
              hint: 'Del',
              danger: true,
              onClick: () =>
                removeClips(selectedClipIds.includes(menu.clipId) ? selectedClipIds : [menu.clipId]),
            },
          ]}
        />
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ 轨道标签 */

/**
 * 轨道左侧的小图标标签（现在由外层那条固定窄栏摆放）。
 *
 * 原来是 9.5px 的灰字（"源片 AI 回合"/"视频轨 1"），在深色带上几乎看不清，
 * 两条带子挤在一起也分不出哪条是原料、哪条是成品。改成：
 * 一个图标 + 悬停展开的完整说明。只负责画那颗按钮，定位交给外面的栏。
 */
function LaneChip({
  icon,
  title,
  content,
  tone,
}: {
  icon: ReactNode
  title: string
  content: ReactNode
  tone: string
}) {
  return (
    <Tooltip
      side="right"
      width={340}
      content={
        <div>
          <div className="mb-1 font-medium" style={{ color: tone }}>
            {title}
          </div>
          {content}
        </div>
      }
    >
      <button
        className="grid h-[18px] w-[18px] shrink-0 cursor-help place-items-center rounded-[5px] border transition-colors"
        style={{
          color: tone,
          borderColor: `${tone}55`,
          background: 'rgb(6 8 11 / 0.86)',
        }}
      >
        {icon}
      </button>
    </Tooltip>
  )
}

/* ------------------------------------------------------------------ 标尺 */

function Ruler({ zoom, duration }: { zoom: number; duration: number }) {
  const ticks = useMemo(() => {
    const targets = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600]
    let step = targets[targets.length - 1]
    for (const t of targets) {
      if (t * zoom >= 62) {
        step = t
        break
      }
    }
    const out: number[] = []
    for (let t = 0; t <= duration + step; t += step) out.push(t)
    return { out, step }
  }, [zoom, duration])

  return (
    <div className="relative h-full select-none">
      {ticks.out.map((t) => (
        <div key={t} className="absolute top-0 h-full" style={{ left: t * zoom }}>
          <div className="absolute bottom-0 h-[7px] w-px bg-white/22" />
          <div className="mono absolute top-[4px] left-1 text-[9.5px] text-ink-400">{timecode(t, false)}</div>
        </div>
      ))}
    </div>
  )
}

/* ------------------------------------------------------------------ 装饰波形 */

function WaveBars({ seed }: { seed: string }) {
  const bars = useMemo(() => {
    let h = 0
    for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) >>> 0
    const out: number[] = []
    for (let i = 0; i < 90; i++) {
      h = (h * 1103515245 + 12345) >>> 0
      out.push(0.2 + ((h >>> 16) % 1000) / 1250)
    }
    return out
  }, [seed])
  return (
    <div className="flex h-full items-end gap-[1px] px-1">
      {bars.map((b, i) => (
        <div key={i} className="flex-1 rounded-t-[1px] bg-white/60" style={{ height: `${b * 100}%` }} />
      ))}
    </div>
  )
}
