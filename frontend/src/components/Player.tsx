import { AnimatePresence, motion } from 'motion/react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  ChevronFirst,
  ChevronLast,
  Volume2,
  VolumeX,
  Repeat,
  Sparkles,
  Maximize2,
  ListFilter,
  Crosshair,
  Film,
  Clapperboard,
} from 'lucide-react'
import { api } from '../lib/api'
import { cn, scoreColor, timecode } from '../lib/format'
import { clipDur, filmClipAt, filmTimeOfSource, filmTimeToSrc, orderedClips } from '../lib/timeline'
import type { Clip, Rally } from '../lib/types'
import { Badge, Button, Progress, SpeedMenu, Tooltip } from './ui'
import { DEFAULT_FILTER, orderedInScope, useStore } from '../store/useStore'
import { useT } from '../i18n/useT'

const COURT_COLORS: Record<string, string> = {
  green: '#4ade80',
  blue: '#60a5fa',
  yellow: '#fbbf24',
  cyan: '#22d3ee',
  orange: '#fb923c',
  red: '#f87171',
  unknown: '#a3a3a3',
}

/** 在预览画面上画出 AI 识别到的球场范围。
 *
 * 视频用 `object-contain` 显示，所以画面实际占据的区域是**居中的等比矩形**，
 * 用一个同样按比例定位的绝对层叠上去，归一化坐标就能直接映射到像素。
 *
 * 画的是**多边形**（4~24 点）：全景 / 鱼眼素材的场地边界是弯的，
 * 用四个角画出来会让人以为「AI 圈错了」，其实只是形状描述不够。
 * 多边形之外还压了一层半透明遮罩 —— 「它到底在看哪一块」一眼就能看出来，
 * 而这正是标定最容易出错的地方（把看台当成场地）。
 */
function CourtOverlay({
  polygon,
  mediaAspect,
  color,
  label,
}: {
  polygon: number[][]
  mediaAspect: number
  color: string
  label: string
}) {
  const tr = useT()
  const ref = useRef<HTMLDivElement>(null)
  const [box, setBox] = useState({ w: 0, h: 0 })

  useEffect(() => {
    const el = ref.current?.parentElement
    if (!el) return
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect()
      setBox({ w: r.width, h: r.height })
    })
    ro.observe(el)
    const r = el.getBoundingClientRect()
    setBox({ w: r.width, h: r.height })
    return () => ro.disconnect()
  }, [])

  if (!polygon || polygon.length < 3 || box.w <= 0 || box.h <= 0) return null

  // object-contain：画面按自身比例缩放后居中
  const containerAspect = box.w / Math.max(1, box.h)
  let vw = box.w
  let vh = box.h
  if (containerAspect > mediaAspect) {
    vh = box.h
    vw = vh * mediaAspect
  } else {
    vw = box.w
    vh = vw / mediaAspect
  }
  const ox = (box.w - vw) / 2
  const oy = (box.h - vh) / 2
  const px = (x: number) => ox + x * vw
  const py = (y: number) => oy + y * vh
  const pts = polygon.map(([x, y]) => `${px(x)},${py(y)}`).join(' ')
  const stroke = COURT_COLORS[color] ?? COURT_COLORS.unknown
  // 外圈矩形 + 多边形，用 evenodd 把多边形挖空 = 只压暗场地之外
  const maskPath =
    `M ${ox},${oy} H ${ox + vw} V ${oy + vh} H ${ox} Z ` +
    polygon.map(([x, y], i) => `${i === 0 ? 'M' : 'L'} ${px(x)},${py(y)}`).join(' ') +
    ' Z'

  return (
    <div ref={ref} className="pointer-events-none absolute inset-0">
      <svg className="h-full w-full">
        <path d={maskPath} fill="#000" fillOpacity={0.42} fillRule="evenodd" />
        <polygon
          points={pts}
          fill={stroke}
          fillOpacity={0.06}
          stroke={stroke}
          strokeOpacity={0.9}
          strokeWidth={2}
          strokeDasharray="7 4"
        />
        {polygon.map(([x, y], i) => (
          <circle
            key={i}
            cx={px(x)}
            cy={py(y)}
            r={polygon.length > 8 ? 2 : 3}
            fill={stroke}
            fillOpacity={0.9}
          />
        ))}
        <text x={px(polygon[0][0]) + 6} y={py(polygon[0][1]) - 8} fill={stroke} fontSize={11} opacity={0.95}>
          {tr('player.courtOverlayLabel', { label })}
          {polygon.length > 4 ? ` · ${tr('player.courtPointCount', { n: polygon.length })}` : ''}
        </text>
      </svg>
    </div>
  )
}


/** 播放进度条：点击 / 拖动定位，方向键逐帧、Home/End 跳转。
 *
 * 值就是当前预览模式的时间轴时间（源片模式=原片时间、成片模式=成片时间），
 * 与 Player 下面算出的 duration 同一套单位，所以这里不需要再做模式换算。
 *
 * 视觉上用颜色区分播放内容：成片预览是蓝色（并画出片段覆盖块，空隙留黑），
 * 源片预览是绿色，和画面左上角的模式徽标、进度条旁的模式标签一致。
 */
function SeekBar({
  value,
  max,
  onSeek,
  onSeekingChange,
  disabled,
  className,
  mode = 'source',
  segments,
}: {
  value: number
  max: number
  onSeek: (t: number) => void
  onSeekingChange: (v: boolean) => void
  disabled?: boolean
  className?: string
  mode?: 'source' | 'timeline'
  /** 成片模式下每个片段在进度条上的位置（百分比），用来可视化片段覆盖与空隙 */
  segments?: { left: number; width: number }[]
}) {
  const tr = useT()
  const ref = useRef<HTMLDivElement>(null)
  const [dragging, setDragging] = useState(false)
  const [hover, setHover] = useState<{ left: number; t: number; width: number } | null>(null)

  const metric = useCallback(
    (clientX: number) => {
      const el = ref.current
      if (!el) return { t: 0, left: 0, width: 0 }
      const r = el.getBoundingClientRect()
      const w = Math.max(1, r.width)
      const left = Math.max(0, Math.min(w, clientX - r.left))
      return { t: max > 0 ? (left / w) * max : 0, left, width: w }
    },
    [max],
  )

  useEffect(() => {
    if (!dragging) return
    const move = (e: PointerEvent) => onSeek(metric(e.clientX).t)
    const up = () => {
      setDragging(false)
      onSeekingChange(false)
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
    window.addEventListener('pointercancel', up)
    document.body.classList.add('grabbing')
    return () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
      window.removeEventListener('pointercancel', up)
      document.body.classList.remove('grabbing')
    }
  }, [dragging, metric, onSeek, onSeekingChange])

  const pct = max > 0 ? Math.max(0, Math.min(100, (value / max) * 100)) : 0

  return (
    <div
      ref={ref}
      role="slider"
      tabIndex={disabled ? -1 : 0}
      aria-label={tr('player.seek')}
      aria-valuemin={0}
      aria-valuemax={Math.round(max) || 0}
      aria-valuenow={Math.round(value) || 0}
      aria-disabled={disabled}
      title={
        mode === 'timeline' ? tr('player.seekFilmHint') : tr('player.seekSourceHint')
      }
      onPointerDown={(e) => {
        if (disabled || e.button !== 0) return
        e.preventDefault()
        ref.current?.focus()
        onSeekingChange(true)
        setDragging(true)
        onSeek(metric(e.clientX).t)
      }}
      onMouseMove={(e) => {
        if (disabled || max <= 0) return
        const m = metric(e.clientX)
        setHover({ left: m.left, t: m.t, width: m.width })
      }}
      onMouseLeave={() => setHover(null)}
      onKeyDown={(e) => {
        if (disabled || max <= 0) return
        const fs = useStore.getState().frameStep()
        if (e.key === 'ArrowLeft') {
          e.preventDefault()
          onSeek(value - fs)
        } else if (e.key === 'ArrowRight') {
          e.preventDefault()
          onSeek(value + fs)
        } else if (e.key === 'Home') {
          e.preventDefault()
          onSeek(0)
        } else if (e.key === 'End') {
          e.preventDefault()
          onSeek(max)
        } else if (e.key === 'PageDown') {
          e.preventDefault()
          onSeek(value - 5)
        } else if (e.key === 'PageUp') {
          e.preventDefault()
          onSeek(value + 5)
        }
      }}
      className={cn(
        'group/seek relative flex h-5 cursor-pointer touch-none items-center outline-none',
        disabled && 'cursor-not-allowed opacity-50',
        className,
      )}
    >
      <div className="relative h-[5px] w-full overflow-hidden rounded-full bg-white/12">
        {segments?.map((s, i) => (
          <div
            key={i}
            className="absolute inset-y-0 bg-white/10"
            style={{ left: `${s.left}%`, width: `${s.width}%` }}
          />
        ))}
        <div
          className={cn(
            'absolute inset-y-0 left-0 rounded-full bg-gradient-to-r',
            mode === 'timeline' ? 'from-flux-500 to-flux-400' : 'from-court-500 to-court-300',
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div
        className="pointer-events-none absolute top-1/2 -translate-x-1/2 -translate-y-1/2"
        style={{ left: `${pct}%` }}
      >
        <div
          className={cn(
            'h-3.5 w-3.5 rounded-full border-2 border-ink-950 transition-transform',
            mode === 'timeline'
              ? 'bg-flux-400 shadow-[0_0_0_1px_rgb(92_157_255/0.6)]'
              : 'bg-court-300 shadow-[0_0_0_1px_rgb(56_224_162/0.6)]',
            dragging ? 'scale-110' : 'group-hover/seek:scale-110',
          )}
        />
      </div>
      {hover && (
        <div
          className="pointer-events-none absolute -top-7 -translate-x-1/2 rounded-md border border-white/10 bg-ink-850/95 px-1.5 py-0.5 text-[10.5px] text-ink-100 shadow-lg"
          style={{ left: Math.max(22, Math.min(hover.left, hover.width - 22)) }}
        >
          {timecode(hover.t, false)}
        </div>
      )}
    </div>
  )
}

export default function Player() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const project = useStore((s) => s.project)
  const mediaId = useStore((s) => s.mediaId)
  const media = useStore((s) => s.currentMedia())
  const proxyReady = !!media?.proxy_path
  const analysis = useStore((s) => s.currentAnalysis())
  const playing = useStore((s) => s.playing)
  const currentTime = useStore((s) => s.currentTime)
  const setPlaying = useStore((s) => s.setPlaying)
  const seek = useStore((s) => s.seek)
  const syncTime = useStore((s) => s.syncTime)
  const setUserSeeking = useStore((s) => s.setUserSeeking)
  const userSeeking = useStore((s) => s.userSeeking)
  const previewMode = useStore((s) => s.previewMode)
  const setPreviewMode = useStore((s) => s.setPreviewMode)
  const selectedRallyId = useStore((s) => s.selectedRallyId)
  const selectedClipId = useStore((s) => s.selectedClipId)
  const selectRally = useStore((s) => s.selectRally)
  const toast = useStore((s) => s.toast)
  const previewFiltered = useStore((s) => s.previewFiltered)
  const setPreviewFiltered = useStore((s) => s.setPreviewFiltered)
  const filter = useStore((s) => s.filter)
  const scope = useStore((s) => s.rallyScope)
  const switchMediaAt = useStore((s) => s.switchMediaAt)
  const nextFilteredTarget = useStore((s) => s.nextFilteredTarget)
  // AI 识别到的球场范围：可以叠在画面上，用来核对「它到底在看哪块场地」
  const calibration = analysis?.calibration ?? null
  const showCourt = useStore((s) => s.showCourtOverlay)
  const setShowCourtOverlay = useStore((s) => s.setShowCourtOverlay)
  const manualPoly = useStore((s) => s.currentCourtPoly())
  const setCourtEditorOpen = useStore((s) => s.setCourtEditorOpen)
  const jobs = useStore((s) => s.jobs)
  const tr = useT()

  // 当前素材正在生成的预览副本任务（用于把静态占位换成真实进度条）
  const prepareJob = useMemo(() => {
    if (!mediaId) return null
    return (
      Object.values(jobs)
        .filter(
          (j) =>
            j.kind === 'prepare' &&
            j.media_id === mediaId &&
            (j.status === 'running' || j.status === 'queued'),
        )
        .sort((a, b) => a.created_at - b.created_at)
        .at(-1) ?? null
    )
  }, [jobs, mediaId])

  const [muted, setMuted] = useState(false)
  const [volume, setVolume] = useState(1)
  const [speed, setSpeed] = useState(1)
  // 回合循环默认关：开着的时候一旦选中某个回合，播放就再也离不开它，
  // 用户会以为播放器坏了。需要反复看一个回合时再手动打开。
  const [loop, setLoop] = useState(false)
  const [ready, setReady] = useState(false)

  const timeline = project?.timeline
  const clips = useMemo(() => orderedClips(timeline), [timeline])
  // 帧率来源：成片模式用成片帧率，源片模式用代理帧率（代理最高 30fps）。
  // 之前逐帧按钮直接读源片 fps，成片模式下读数是成片时间，帧号就对不上了。
  const displayFps = previewMode === 'timeline' ? timeline?.fps || 30 : media?.proxy_fps || media?.fps || 30
  const frameStep = 1 / displayFps

  /** 成片模式：播放头落点对应的片段（片段之间的空隙 / 首尾之外返回 null）。 */
  const clipAtPlayhead = useCallback((t: number) => filmClipAt(clips, t), [clips])

  /** 成片模式下播放头走的是成片时间，这里换算回它对应的当前素材内时间；
   *  落点在别的素材的片段上时对当前素材没有意义，返回 -1（匹配不到任何回合）。 */
  const srcTime = useMemo(() => {
    if (previewMode !== 'timeline') return currentTime
    const c = clipAtPlayhead(currentTime)
    if (!c || c.media_id !== mediaId) return -1
    return filmTimeToSrc(c, currentTime)
  }, [previewMode, currentTime, mediaId, clipAtPlayhead])

  /** 跳到某个原片时刻：成片模式下换算成成片时间，不在成片里就切回源片 */
  const goToSource = useCallback(
    (src: number) => {
      if (previewMode === 'timeline') {
        const ft = filmTimeOfSource(clips, mediaId, src)
        if (ft !== null) {
          seek(ft)
          return
        }
        setPreviewMode('source')
      }
      seek(src)
    },
    [previewMode, clips, mediaId, seek, setPreviewMode],
  )

  const rallyAtTime = useMemo(() => {
    if (!analysis) return null
    return analysis.rallies.find((r) => srcTime >= r.start && srcTime <= r.end) ?? null
  }, [analysis, srcTime])

  const selectedRally = useMemo(
    () => analysis?.rallies.find((r) => r.id === selectedRallyId) ?? null,
    [analysis, selectedRallyId],
  )

  const activeRally = selectedRally ?? rallyAtTime

  /** 当前筛选出来的回合，按「素材顺序 → 素材内开始时间」排出播放顺序（跨素材队列） */
  const filteredRallies = useMemo(
    () => orderedInScope(project, scope, mediaId, filter),
    [project, scope, mediaId, filter],
  )

  /** 范围内全部回合（未过筛选），用于上一/下一回合跨素材跳转 */
  const scopeRallies = useMemo(
    () => orderedInScope(project, scope, mediaId, DEFAULT_FILTER),
    [project, scope, mediaId],
  )

  const filteredIndex = useMemo(
    () =>
      filteredRallies.findIndex(
        (r) => r.media_id === mediaId && currentTime >= r.start && currentTime <= r.end,
      ),
    [filteredRallies, mediaId, currentTime],
  )

  // ------------------------------------------------ 同步 video 与状态
  // 只在切换素材时重置 ready。切换「源片/成片预览」不会换 src，
  // 若把 previewMode 也放进依赖，loadedmetadata 不再触发，遮罩会永远盖住画面。
  useEffect(() => {
    setReady(false)
    // 代理生成完成后要给 video 换成代理源：同一路径的内容变了，浏览器不一定会重新解，
    // 所以这里把 proxy_path 也列入依赖，下面再用 key 强制重新挂载。
  }, [mediaId, media?.proxy_path])

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    if (playing) {
      // 换源 / 重挂载时 play() 可能被新的加载请求打断（AbortError），这不是用户想暂停，
      // 忽略它；只有真正的自动播放限制才把 playing 复位。
      v.play().catch((err: unknown) => {
        if ((err as { name?: string } | null)?.name !== 'AbortError') setPlaying(false)
      })
    } else {
      v.pause()
    }
    // proxyReady / mediaId：代理就绪或换素材时 video 会被 key 强制重挂，
    // 重挂后要恢复播放状态，否则 store 里 playing=true 而画面是冻结的。
  }, [playing, setPlaying, previewMode, proxyReady, mediaId])

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    // 成片预览下叠加片段变速：播放头所在片段 speed=2 时画面也要 2× 快放，
    // 否则画面 1× 而播放头按 2× 映射前进，只能靠校正 effect 每 0.3 秒左右
    // 跳跃追赶（变速片段预览卡顿、声画不同步）。空隙 / 源片模式按 1×。
    const clipSpeed =
      previewMode === 'timeline' ? clipAtPlayhead(currentTime)?.speed ?? 1 : 1
    v.playbackRate = speed * clipSpeed
    // 重挂载后新元素会丢掉这些属性；播放头跨片段时变速比也要跟着换
  }, [speed, proxyReady, mediaId, previewMode, clipAtPlayhead, currentTime])

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    v.muted = muted
    v.volume = volume
  }, [muted, volume, proxyReady, mediaId])

  // 播放头由 store 驱动 -> 校正 video。这里同时兜住两类「时间线与画面不同步」：
  // 1) 播放头落在片段之间的空隙（编辑/撤销后播放头可能被留在已删除的区域）：
  //    吸附回最近的片段范围，播放头指的永远是成片里真实存在的内容；
  // 2) 播放头指向的片段属于别的素材（跨素材成片）：成片预览必须切换视频源，
  //    否则 video 会拿别的素材的原片时间解当前素材，播出一堆不相干的画面。
  useEffect(() => {
    const v = videoRef.current
    if (!v || !ready) return
    let target: number | null = null
    if (previewMode === 'timeline') {
      const c = clipAtPlayhead(currentTime)
      if (!c) {
        // 拖动播放头期间不抢鼠标（松手时 setUserSeeking(false) 会统一吸附）
        const st = useStore.getState()
        if (!st.userSeeking && Date.now() >= st.seekingUntil) {
          st.snapPlayheadToContent()
        }
        return
      }
      if (c.media_id !== mediaId) {
        // 拖动中先不切源；松手后 userSeeking 变化会让本 effect 重新执行
        const st = useStore.getState()
        if (!st.userSeeking) st.switchMediaInFilm(c.media_id, currentTime)
        return
      }
      target = filmTimeToSrc(c, currentTime)
    } else {
      target = currentTime
    }
    // 暂停时的容差要小得多：逐帧按钮一次只走 1/30 秒，
    // 阈值给 0.34 的话点十几下画面都不动（只有播放头在走）。
    const tol = playing ? 0.34 : 0.012
    if (target !== null && Math.abs(v.currentTime - target) > tol) {
      try {
        v.currentTime = target
      } catch {
        /* ignore */
      }
    }
  }, [currentTime, ready, previewMode, clipAtPlayhead, playing, mediaId, userSeeking])

  const onTimeUpdate = () => {
    const v = videoRef.current
    if (!v || !ready) return
    const st = useStore.getState()
    let t = v.currentTime
    if (previewMode === 'timeline') {
      // 只在「当前素材」的片段里找映射：成片里别的素材的片段可能覆盖同一个
      // 原片时间，但画面不是它，不能拿来推播放头位置。
      let mapped: number | null = null
      let hit: Clip | null = null
      for (const c of clips) {
        if (c.media_id !== mediaId) continue
        if (t >= c.src_in && t < c.src_out) {
          hit = c
          mapped = c.tl_start + (t - c.src_in) / c.speed
          break
        }
      }
      if (hit && mapped !== null) {
        const dur = clipDur(hit)
        // 快进到片段末尾时跳到下一段；下一段在别的素材上就换源续播
        if (mapped >= hit.tl_start + dur - 0.06 && playing && !st.userSeeking) {
          const next = clips[clips.indexOf(hit) + 1]
          if (next) {
            if (next.media_id !== mediaId) {
              st.switchMediaInFilm(next.media_id, next.tl_start)
            } else {
              try {
                v.currentTime = next.src_in
              } catch {
                /* ignore */
              }
              seek(next.tl_start)
            }
            return
          }
          // 最后一段：到头就停，别让原片时间继续往前跑进没剪进成片的素材
          v.pause()
          setPlaying(false)
          seek(hit.tl_start + dur)
          return
        }
        syncTime(mapped)
        return
      }
      // 画面落点没有对应的成片位置（编辑 / 换源后的残留位置）。播放中就地把
      // 画面拉回播放头所指的内容（或续播到下一段），绝不裸放没剪进成片的素材；
      // 暂停中不动画面，播放头的吸附由校正 effect 负责。
      // 注意 timeupdate 采样可能直接落过片段末端（采样间隔 ~250ms），此时上面的
      // 「片段内临近末端」分支永远不会命中；必须在这里按「视频已越过某段末端」
      // 确定性判定播完并续播下一段，否则画面会被拉回上一段结尾无限回跳，
      // 播放头就冻结在片段边界（跨片段播放卡死、时间线与画面失同步的根源）。
      if (playing && !st.userSeeking && Date.now() >= st.seekingUntil) {
        let ended: Clip | null = null
        for (const c of clips) {
          if (c.media_id !== mediaId) continue
          // 窗口放宽到数倍片段时长：换源屏蔽窗口 / 采样间隔内视频可能已多跑一段
          if (t >= c.src_out && t < c.src_out + 3 * (c.speed || 1) && (ended === null || c.src_out > ended.src_out)) {
            ended = c
          }
        }
        if (ended) {
          const next = clips[clips.indexOf(ended) + 1] ?? null
          if (next) {
            if (next.media_id !== mediaId) {
              st.switchMediaInFilm(next.media_id, next.tl_start)
            } else {
              try {
                v.currentTime = next.src_in
              } catch {
                /* ignore */
              }
              seek(next.tl_start)
            }
          } else {
            // 最后一段：到头就停，别让原片时间继续往前跑进没剪进成片的素材
            v.pause()
            setPlaying(false)
            seek(timeline?.duration ?? 0)
          }
          return
        }
        const c = filmClipAt(clips, st.currentTime)
        if (c && c.media_id === mediaId) {
          try {
            v.currentTime = filmTimeToSrc(c, st.currentTime)
          } catch {
            /* ignore */
          }
        } else if (c) {
          st.switchMediaInFilm(c.media_id, st.currentTime)
        } else {
          const next = clips.find((x) => x.tl_start >= st.currentTime - 1e-6) ?? null
          if (next) {
            if (next.media_id !== mediaId) {
              st.switchMediaInFilm(next.media_id, next.tl_start)
            } else {
              try {
                v.currentTime = next.src_in
              } catch {
                /* ignore */
              }
              seek(next.tl_start)
            }
          } else {
            v.pause()
            setPlaying(false)
            seek(timeline?.duration ?? 0)
          }
        }
      }
      return
    } else if (previewFiltered && playing && mediaId) {
      // 只看筛选片段：跳过空档。timeupdate 大约 4 次/秒，
      // 所以在回合末尾提前 60ms 就起跳，避免看到下一段捡球的画面。
      // 全部素材模式下，落点在别的素材时直接切换视频源并定位。
      const jump = nextFilteredTarget(mediaId, v.currentTime)
      if (jump === 'end') {
        setPlaying(false)
        return
      }
      if (jump !== null) {
        if (jump.media_id !== mediaId) {
          switchMediaAt(jump.media_id, jump.start, jump.rally_id)
          return
        }
        try {
          v.currentTime = jump.start
        } catch {
          /* ignore */
        }
        seek(jump.start)
        return
      }
    }
    // 用 syncTime 而不是 seek：拖动播放头期间 video 会以旧时间继续触发
    // timeupdate，直接回写会把播放头拽回去，表现为「跳变」。
    syncTime(t)
  }

  const jumpRally = (dir: 1 | -1) => {
    const list = scopeRallies
    if (!list.length) return
    const order = new Map((project?.media ?? []).map((m, i) => [m.id, i]))
    const myIdx = mediaId ? order.get(mediaId) ?? 0 : 0
    // 查找用原片时间：成片模式下 currentTime 是成片时间，不能直接和回合范围比。
    // 全部素材模式下跨素材时，selectRally 会自动切源并定位（同素材才用 goToSource）。
    if (dir === 1) {
      const nxt = list.find((r) => {
        const ri = order.get(r.media_id) ?? 0
        return ri > myIdx || (ri === myIdx && r.start > srcTime + 0.05)
      })
      if (nxt) {
        selectRally(nxt.id)
        if (nxt.media_id === mediaId) goToSource(nxt.start)
      }
    } else {
      const prev = [...list]
        .reverse()
        .find((r) => {
          const ri = order.get(r.media_id) ?? 0
          return ri < myIdx || (ri === myIdx && r.end < srcTime - 0.35)
        })
      if (prev) {
        selectRally(prev.id)
        if (prev.media_id === mediaId) goToSource(prev.start)
      }
    }
  }

  // 选中回合时自动定位
  const lastSelected = useRef<string | null>(null)
  useEffect(() => {
    if (!selectedRally || lastSelected.current === selectedRally.id) return
    lastSelected.current = selectedRally.id
    goToSource(selectedRally.start)
  }, [selectedRally, goToSource])

  // 回合循环（默认关闭）。
  // activeRally 在播放头越过回合终点后会立刻变成 null（rallyAtTime 找不到当前回合），
  // 所以用 ref 记住最近一次所在的回合，否则「只是播到某个回合、没有显式选中」时
  // 循环永远不触发，用户会以为按钮坏了。
  const loopTargetRef = useRef<Rally | null>(null)
  useEffect(() => {
    const t = selectedRally ?? rallyAtTime
    if (t) loopTargetRef.current = t
    else if (!playing) loopTargetRef.current = null
  }, [selectedRally, rallyAtTime, playing])

  useEffect(() => {
    if (!loop || !playing) return
    const target = selectedRally ?? loopTargetRef.current
    if (!target) return
    if (srcTime > target.end + 0.15) {
      goToSource(target.clip_start ?? target.start)
    }
  }, [srcTime, selectedRally, loop, playing, goToSource])

  /** 当前选中的片段在成片时间轴上的范围（用于给分割提示兜错） */
  const selectedClipRange = useMemo(() => {
    if (previewMode !== 'timeline') return null
    const c = clips.find((x) => x.id === selectedClipId)
    if (!c) return null
    return { start: c.tl_start, end: c.tl_start + (c.src_out - c.src_in) / c.speed }
  }, [previewMode, selectedClipId, clips])

  const splitHint = useMemo(() => {
    if (!selectedClipId || !selectedClipRange) return null
    return currentTime >= selectedClipRange.start && currentTime <= selectedClipRange.end
  }, [selectedClipId, selectedClipRange, currentTime])

  // 代理就绪后加一个参数当缓存戳，避免浏览器继续拿代理生成前那次响应
  const src = mediaId && project ? `${api.proxyUrl(project.id, mediaId)}${proxyReady ? '?ready=1' : ''}` : undefined
  const duration = previewMode === 'timeline' ? timeline?.duration ?? 0 : media?.duration ?? 0

  /** 成片模式下把每个片段画到进度条上：浅色块 = 有内容，留黑 = 空隙 */
  const filmSegments = useMemo(() => {
    if (previewMode !== 'timeline' || duration <= 0) return undefined
    return clips.map((c) => ({
      left: Math.min(100, (c.tl_start / duration) * 100),
      width: Math.max(0, (clipDur(c) / duration) * 100),
    }))
  }, [previewMode, clips, duration])

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* 工具条 */}
      <div className="flex items-center gap-2 px-3 py-2">
        <div className="flex items-center gap-1.5">
          <Button
            variant={previewMode === 'source' ? 'outline' : 'ghost'}
            size="sm"
            onClick={() => setPreviewMode('source')}
          >
            {tr('player.modeSource')}
          </Button>
          <Button
            variant={previewMode === 'timeline' ? 'outline' : 'ghost'}
            size="sm"
            onClick={() => {
              setPreviewMode('timeline')
              // 成片预览本身就是「按成片顺序播」，两个模式叠加只会互相打架
              setPreviewFiltered(false)
            }}
            disabled={!clips.length}
          >
            {tr('player.modeTimeline')}
            {clips.length > 0 && (
              <span className="mono ml-1 text-[10px] text-court-300">{clips.length}</span>
            )}
          </Button>
          <Tooltip
            side="top"
            width={300}
            content={
              previewFiltered
                ? tr('player.onlyFilteredOnHint', { n: filteredRallies.length })
                : tr('player.onlyFilteredOffHint', { n: filteredRallies.length })
            }
          >
            <Button
              variant={previewFiltered ? 'outline' : 'ghost'}
              size="sm"
              disabled={!filteredRallies.length}
              onClick={() => {
                if (previewFiltered) setPreviewFiltered(false)
                else useStore.getState().previewFilteredRallies()
              }}
              className={previewFiltered ? 'text-court-300' : ''}
            >
              <ListFilter size={12} />
              {tr('player.onlyFiltered')}
            </Button>
          </Tooltip>
        </div>

        <div className="flex-1" />

        <AnimatePresence>
          {activeRally && (
            <motion.div
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              className="flex items-center gap-2 rounded-lg border border-white/8 bg-white/4 px-2.5 py-1"
            >
              <Sparkles size={11} style={{ color: scoreColor(activeRally.scores.total) }} />
              <span className="text-[11.5px] text-ink-200">
                {tr('player.rallyIndex', { index: activeRally.index })}
              </span>
              <span
                className="mono text-[12px] font-semibold"
                style={{ color: scoreColor(activeRally.scores.total) }}
              >
                {activeRally.scores.total.toFixed(0)}
              </span>
              <span className="text-[10.5px] text-ink-500">
                {tr('player.shotDuration', {
                  shots: activeRally.features.shot_count,
                  duration: activeRally.duration.toFixed(1),
                })}
              </span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* 画面：让视频按自身比例铺满整个区域（object-contain），
          容器本身不再留多余的黑色边距 */}
      <div className="relative mx-3 flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-xl border border-white/8 bg-black">
        {src ? (
          // key 带上 mediaId / proxy_path：换素材或代理重生成时强制重挂载，
          // 保证 loadedmetadata 一定触发（否则 ready 会卡在 false，遮罩一直盖着）。
          <video
            key={`${mediaId ?? ''}:${proxyReady ? media.proxy_path ?? 'proxy' : 'source'}`}
            ref={videoRef}
            src={src}
            className="block h-full w-full object-contain"
            onLoadedMetadata={(e) => {
              setReady(true)
              // 重挂载后恢复播放（换素材 / 代理就绪时 store.playing 可能已经是 true）
              if (playing) e.currentTarget.play().catch(() => undefined)
            }}
            onTimeUpdate={onTimeUpdate}
            onClick={() => setPlaying(!playing)}
            playsInline
          />
        ) : (
          <div className="text-[12.5px] text-ink-500">{tr('player.importFirst')}</div>
        )}

        {calibration?.ok && showCourt && media && media.width > 0 && (
          <CourtOverlay
            polygon={calibration.polygon?.length ? calibration.polygon : calibration.quad}
            mediaAspect={media.width / Math.max(1, media.height)}
            color={calibration.court_color}
            label={calibration.viewpoint_label}
          />
        )}

        {!ready && src && (
          <div className="pointer-events-none absolute inset-0 grid place-items-center bg-ink-950/60 px-8">
            <div className="w-[300px] text-center">
              <div className="text-[12px] text-ink-200">
                {prepareJob
                  ? prepareJob.stage === 'queued'
                    ? tr('player.prepareQueued')
                    : prepareJob.message || tr('player.prepareGenerating')
                  : tr('player.preparePreparing')}
              </div>
              {prepareJob && <Progress value={prepareJob.progress} className="mt-2" />}
              <div className="mono mt-1 text-[10.5px] text-ink-500">
                {prepareJob ? `${Math.round(prepareJob.progress * 100)}%` : tr('player.prepareDirectPlay')}
              </div>
              {!prepareJob && media && !proxyReady && mediaId && (
                <button
                  onClick={() => useStore.getState().ensurePrepare(mediaId)}
                  className="pointer-events-auto mt-2 rounded-lg bg-white/10 px-3 py-1.5 text-[11.5px] text-ink-100 hover:bg-white/16"
                >
                  {tr('player.generateProxy')}
                </button>
              )}
            </div>
          </div>
        )}

        {/* 内容来源标识：一眼看清现在播的是原片还是成片、是否只看筛选 */}
        <div className="pointer-events-none absolute top-2.5 left-1/2 flex -translate-x-1/2 items-center gap-1.5">
          <Badge
            color={previewMode === 'timeline' ? '#38e0a2' : '#94a3b8'}
            className="backdrop-blur bg-black/55"
          >
            {previewMode === 'timeline' ? <Clapperboard size={9} /> : <Film size={9} />}
            {previewMode === 'timeline'
              ? tr('player.previewTimelineBadge', { n: clips.length })
              : tr('player.previewSourceBadge')}
          </Badge>
          {previewMode === 'source' && previewFiltered && (
            <Badge color="#5c9dff" className="backdrop-blur bg-black/55">
              <ListFilter size={9} />
              {tr('player.onlyFiltered')}
            </Badge>
          )}
        </div>

        {/* 角标 */}
        <div className="pointer-events-none absolute top-2.5 left-3 flex items-center gap-2">
          <Badge className="bg-black/50 backdrop-blur">{media?.width}×{media?.height}</Badge>
          {media && media.fps > 0 && <Badge className="bg-black/50 backdrop-blur">{media.fps.toFixed(2)} fps</Badge>}
          {media && (
            <button
              onClick={() => setCourtEditorOpen(true)}
              title={
                manualPoly
                  ? tr('player.manualCalibrated', { n: manualPoly.length })
                  : calibration?.ok
                    ? tr('player.aiViewpointHint', { viewpoint: calibration.viewpoint_label })
                    : tr('player.manualCalibrateHint')
              }
              className="pointer-events-auto"
            >
              <Badge
                color={manualPoly ? '#38e0a2' : calibration?.ok ? COURT_COLORS[calibration.court_color] ?? undefined : '#f59e0b'}
                className={cn('backdrop-blur', manualPoly || calibration?.ok ? '' : 'bg-black/50')}
              >
                <Crosshair size={9} />
                {manualPoly
                  ? tr('player.manualCalibratedBadge', { n: manualPoly.length })
                  : calibration?.ok
                    ? tr('player.viewpointBadge', { viewpoint: calibration.viewpoint_label })
                    : tr('player.calibrateCourt')}
              </Badge>
            </button>
          )}
          {calibration?.ok && (
            <button
              onClick={() => setShowCourtOverlay(!showCourt)}
              title={showCourt ? tr('player.hideCourtRange') : tr('player.showCourtRangeHint')}
              className="pointer-events-auto"
            >
              <Badge className={cn('backdrop-blur', showCourt ? 'bg-court-500/25 text-court-200' : 'bg-black/50')}>
                {showCourt ? tr('player.hideRange') : tr('player.showRange')}
              </Badge>
            </button>
          )}
        </div>
        <div className="pointer-events-none absolute right-3 bottom-2.5 flex items-center gap-2">
          {loop && activeRally && (
            <Badge color="#38e0a2" className="backdrop-blur">
              {tr('player.loopingRally', { index: activeRally.index })}
            </Badge>
          )}
          {previewMode === 'timeline' && selectedClipId && (
            <Badge color={splitHint ? '#5c9dff' : '#6b7787'} className="backdrop-blur">
              {splitHint ? tr('player.splitHintAt') : tr('player.playheadOutsideClip')}
            </Badge>
          )}
          {rallyAtTime && previewMode === 'source' && (
            <Badge color={scoreColor(rallyAtTime.scores.total)} className="backdrop-blur">
              {tr('player.rallyInProgress')}
            </Badge>
          )}
          {previewMode === 'source' && previewFiltered && (
            <Badge color="#5c9dff" className="backdrop-blur">
              <ListFilter size={9} />
              {tr('player.onlyFilteredClips')}
              {filteredRallies.length > 0 && ` ${filteredIndex >= 0 ? filteredIndex + 1 : '-'}/${filteredRallies.length}`}
            </Badge>
          )}
        </div>

        {/* 全屏按钮。绝对定位放在外层 div 上：
            如果交给 Tooltip 的 span（它是 0×0 的 relative 盒子），
            absolute 的包含块就变成那个 span，按钮会跑到画面正中间。 */}
        <div className="absolute top-2.5 right-3">
          <Tooltip content={tr('player.fullscreen')} side="left">
            <button
              onClick={() => videoRef.current?.parentElement?.requestFullscreen?.()}
              aria-label={tr('player.fullscreen')}
              className="grid h-7 w-7 place-items-center rounded-md bg-black/45 text-ink-300 backdrop-blur transition-colors hover:bg-black/70 hover:text-white"
            >
              <Maximize2 size={13} />
            </button>
          </Tooltip>
        </div>
      </div>

      {/* 播放进度条：点击 / 拖动定位。值用当前模式的时间轴（源片或成片）。 */}
      <div className="flex items-center gap-2.5 px-3 pt-1.5">
        {/* 模式标签：颜色与进度条一致 —— 蓝色=成片、绿色=源片，一眼分清在播什么 */}
        <span
          title={previewMode === 'timeline' ? tr('player.seekFilmHint') : tr('player.seekSourceHint')}
          className={cn(
            'flex shrink-0 items-center gap-1 rounded-md px-1.5 py-[3px] text-[10px] font-semibold whitespace-nowrap',
            previewMode === 'timeline' ? 'bg-flux-500/15 text-flux-400' : 'bg-court-500/10 text-court-300',
          )}
        >
          {previewMode === 'timeline' ? <Clapperboard size={10} /> : <Film size={10} />}
          {previewMode === 'timeline' ? tr('player.seekModeFilm') : tr('player.seekModeSource')}
        </span>
        <span className="mono w-[54px] shrink-0 text-right text-[11px] text-white tabular">
          {timecode(currentTime, false)}
        </span>
        <SeekBar
          value={currentTime}
          max={duration}
          disabled={!src || duration <= 0}
          onSeek={seek}
          onSeekingChange={setUserSeeking}
          mode={previewMode}
          segments={filmSegments}
          className="flex-1"
        />
        <span className="mono w-[54px] shrink-0 text-[11px] text-ink-500 tabular">
          {timecode(duration, false)}
        </span>
      </div>

      {/* 传输控制 */}
      <div className="flex items-center gap-2 px-3 py-2.5">
        <Tooltip content={tr('player.prevRally')} side="top">
          <Button variant="ghost" size="icon" onClick={() => jumpRally(-1)}>
            <ChevronFirst size={15} />
          </Button>
        </Tooltip>
        <Tooltip content={tr('player.stepBack')} kbd="←" side="top">
          <Button variant="ghost" size="icon" onClick={() => seek(currentTime - frameStep)}>
            <SkipBack size={15} />
          </Button>
        </Tooltip>
        <Tooltip content={playing ? tr('player.pause') : tr('player.play')} kbd="Space" side="top">
          <button
            onClick={() => setPlaying(!playing)}
            aria-label={playing ? tr('player.pause') : tr('player.play')}
            className={cn(
              'grid h-9 w-9 place-items-center rounded-full transition-all active:scale-95',
              playing ? 'bg-white/12 text-white' : 'bg-gradient-to-b from-court-400 to-court-600 text-ink-950',
            )}
          >
            {playing ? <Pause size={16} fill="currentColor" /> : <Play size={16} fill="currentColor" className="ml-0.5" />}
          </button>
        </Tooltip>
        <Tooltip content={tr('player.stepForward')} kbd="→" side="top">
          <Button variant="ghost" size="icon" onClick={() => seek(currentTime + frameStep)}>
            <SkipForward size={15} />
          </Button>
        </Tooltip>
        <Tooltip content={tr('player.nextRally')} side="top">
          <Button variant="ghost" size="icon" onClick={() => jumpRally(1)}>
            <ChevronLast size={15} />
          </Button>
        </Tooltip>

        <span className="mono ml-2 text-[11px] text-ink-400" title={tr('player.frameIndex')}>
          f{Math.floor((currentTime % 1) * displayFps)}
        </span>

        <div className="flex-1" />

        <Tooltip content={loop ? tr('player.loopOnHint') : tr('player.loopOffHint')} side="top">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => {
              const next = !loop
              setLoop(next)
              if (next) toast({ kind: 'info', title: tr('player.loopEnabledTitle'), detail: tr('player.loopEnabledDetail') })
            }}
            className={loop ? 'text-court-300' : ''}
          >
            <Repeat size={14} />
          </Button>
        </Tooltip>

        <SpeedMenu value={speed} onChange={setSpeed} title={tr('player.playbackSpeed')} />

        <Tooltip content={muted ? tr('player.unmute') : tr('player.mute')} side="top">
          <Button variant="ghost" size="icon" onClick={() => setMuted((v) => !v)}>
            {muted || volume === 0 ? <VolumeX size={14} /> : <Volume2 size={14} />}
          </Button>
        </Tooltip>
        <input
          type="range"
          min={0}
          max={1}
          step={0.02}
          value={muted ? 0 : volume}
          onChange={(e) => {
            setVolume(parseFloat(e.target.value))
            setMuted(false)
          }}
          className="w-[78px]"
        />
      </div>
    </div>
  )
}
