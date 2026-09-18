import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Crosshair, Minus, Plus, RotateCcw, Spline, Wand2, X } from 'lucide-react'
import { api } from '../lib/api'
import { cn } from '../lib/format'
import { Button, Modal } from './ui'
import { useStore } from '../store/useStore'

/** 归一化多边形顶点（0~1）。4 个点就是传统四边形，多点用来描述弯掉的边界。 */
export type Poly = [number, number][]

/** 后端接受的点数区间（见 court_calib.MIN_POLY_POINTS / MAX_POLY_POINTS）。 */
const MIN_POINTS = 4
const MAX_POINTS = 24

/** 一键生成的初始多边形：贴着画面下半部分的一个梯形。
 *
 * 之所以给个「梯形」而不是矩形：绝大多数的拍法下场地在画面里都是
 * 近大远小的梯形，从这个形状起步用户往往只需要微调一两个角。
 */
function defaultQuad(): Poly {
  return [
    [0.16, 0.99],
    [0.84, 0.99],
    [0.62, 0.52],
    [0.38, 0.52],
  ]
}

function polyArea(q: Poly): number {
  let s = 0
  for (let i = 0; i < q.length; i++) {
    const [x1, y1] = q[i]
    const [x2, y2] = q[(i + 1) % q.length]
    s += x1 * y2 - x2 * y1
  }
  return Math.abs(s) / 2
}

/**
 * 「边界弯得有多厉害」：多边形面积比「同样几个角围出来的四边形」多多少。
 *
 * 取「拐得最厉害的四个顶点」当四边形的角（转弯角度最大的地方就是角），
 * 再用两个面积之比给出一个直观数字。它只是个提示：弯边比例大的时候，
 * 提示用户「这就是畸变，多加几个点会圈得更准」。
 * 真正的量化口径在分析结果里（`calibration.distortion`，后端算的）。
 */
function bulgeRatio(q: Poly): number {
  if (q.length <= 4) return 0
  const a = polyArea(q)
  if (a <= 1e-6) return 0
  const n = q.length
  const turns: { i: number; turn: number }[] = []
  for (let i = 0; i < n; i++) {
    const [px, py] = q[(i - 1 + n) % n]
    const [cx, cy] = q[i]
    const [nx, ny] = q[(i + 1) % n]
    const a1 = Math.atan2(cy - py, cx - px)
    const a2 = Math.atan2(ny - cy, nx - cx)
    let d = Math.abs(a2 - a1)
    if (d > Math.PI) d = 2 * Math.PI - d
    turns.push({ i, turn: d })
  }
  const corners = turns.sort((x, y) => y.turn - x.turn).slice(0, 4).map((t) => t.i).sort((x, y) => x - y)
  const quad = corners.map((i) => q[i]) as Poly
  if (quad.length < 4) return 0
  return Math.max(0, 1 - polyArea(quad) / a)
}

/** 在多边形第 i 条边（i → i+1）的中点插入一个顶点，返回新数组与插入位置。 */
function insertMidpoint(q: Poly, i: number): { next: Poly; index: number } {
  const [x1, y1] = q[i]
  const [x2, y2] = q[(i + 1) % q.length]
  const mid: [number, number] = [(x1 + x2) / 2, (y1 + y2) / 2]
  const next = [...q]
  next.splice(i + 1, 0, mid)
  return { next, index: i + 1 }
}

/** 每条边插一个中点：给弯边（鱼眼 / 全景）一次补足顶点，比逐条加省事。
 *
 * 加到点数上限就停（上限由后端定），保证存下去的一定是后端能吃的形状。
 */
function subdivideAll(q: Poly): Poly {
  const out: Poly = []
  for (let i = 0; i < q.length; i++) {
    const [x1, y1] = q[i]
    const [x2, y2] = q[(i + 1) % q.length]
    out.push([x1, y1])
    const remainingOriginal = q.length - i - 1
    if (out.length + remainingOriginal + 1 <= MAX_POINTS) {
      out.push([(x1 + x2) / 2, (y1 + y2) / 2])
    }
  }
  return out.slice(0, MAX_POINTS)
}

/**
 * 手动标定场地：在视频画面上拖点，把场地圈出来。
 *
 * 为什么需要它：自动标定靠「地面颜色 + 逐行分析」找场地，在大多数素材上够用，
 * 但多球场、地胶颜色异常、场地只占画面一角时它会失败（失败时会在分析结果里
 * 写明原因）。这种情况下让用户拖一次，比继续调算法有效得多，而且标定结果会
 * 跟着工程保存，下次分析直接复用。
 *
 * **为什么支持任意点数**：全景相机 / 鱼眼镜头拍出来的场地边界是弯的
 * （桶形畸变），四个角只能框出一个直线四边形 —— 要么切掉边角，要么把
 * 场地外的看台一起圈进来，而边角恰恰是背景人员最密集的地方。
 * 所以这里可以：
 *
 * - 拖任意顶点（点上去直接拖）；
 * - 在边的中点上「+」加一个顶点（弯的地方多加点）；
 * - 选中顶点后按「−」删掉；
 * - 一键「每条边加中点」（`Spline`）——弯边素材最快的一步到位。
 */
export default function CourtEditor({ open, onClose }: { open: boolean; onClose: () => void }) {
  const project = useStore((s) => s.project)
  const media = useStore((s) => s.currentMedia())
  const analysis = useStore((s) => s.currentAnalysis())
  // 已保存的手动标定从**工程 ui** 里读（按素材分开存）：这样刷新页面、
  // 重新打开工程都能拿到，而不是只有「本次会话刚保存过」才有。
  const savedPoly = useStore((s) => s.currentCourtPoly())
  const setCourtPoly = useStore((s) => s.setCourtPoly)
  const toast = useStore((s) => s.toast)

  const [poly, setPoly] = useState<Poly>(defaultQuad)
  const [drag, setDrag] = useState<number | null>(null)
  const [selected, setSelected] = useState<number | null>(null)
  const boxRef = useRef<HTMLDivElement>(null)

  // 打开时用「已保存的手动标定 > AI 自动标定 > 默认梯形」初始化，
  // 这样用户是在 AI 的结果上微调，而不是从零开始。
  useEffect(() => {
    if (!open) return
    if (savedPoly && savedPoly.length >= MIN_POINTS) {
      setPoly(savedPoly.map((p) => [p[0], p[1]] as [number, number]))
      return
    }
    const auto = analysis?.calibration?.polygon ?? analysis?.calibration?.quad
    if (auto && auto.length >= MIN_POINTS) {
      setPoly(auto.map((p) => [p[0], p[1]] as [number, number]))
      return
    }
    setPoly(defaultQuad())
  }, [open, savedPoly, analysis?.calibration?.polygon, analysis?.calibration?.quad])

  const posterUrl = useMemo(() => {
    if (!project || !media) return null
    // 加时间戳：重新分析后封面会变，浏览器缓存会导致看到旧图
    return `${api.posterUrl(project.id, media.id)}?t=${analysis?.finished_at ?? 0}`
  }, [project, media?.id, analysis?.finished_at])

  const mediaAspect = media && media.height > 0 ? media.width / media.height : 16 / 9

  const movePoint = useCallback(
    (e: React.PointerEvent | PointerEvent, index: number) => {
      if (!boxRef.current) return
      const r = boxRef.current.getBoundingClientRect()
      const x = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width))
      const y = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))
      setPoly((q) => q.map((p, i) => (i === index ? ([x, y] as [number, number]) : p)))
    },
    [],
  )

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (drag === null) return
      movePoint(e, drag)
    },
    [drag, movePoint],
  )

  const area = polyArea(poly)
  const valid = poly.length >= MIN_POINTS && poly.length <= MAX_POINTS && area > 0.02
  const bulge = bulgeRatio(poly)

  const addPoint = (edgeIndex: number) => {
    if (poly.length >= MAX_POINTS) {
      toast({ kind: 'warn', title: `最多 ${MAX_POINTS} 个点`, detail: '再多也不会更准，反而更难调' })
      return
    }
    const { next, index } = insertMidpoint(poly, edgeIndex)
    setPoly(next)
    setSelected(index)
  }

  const removePoint = (index: number) => {
    if (poly.length <= MIN_POINTS) {
      toast({
        kind: 'warn',
        title: `至少要 ${MIN_POINTS} 个点`,
        detail: '四个点才能算出「谁离相机更近」，两个点只是一条线',
      })
      return
    }
    setPoly((q) => q.filter((_, i) => i !== index))
    setSelected(null)
  }

  const save = () => {
    if (!valid) {
      toast({
        kind: 'error',
        title: '这块区域太小了',
        detail: '请把点拉开一些，围出一个像球场的区域（至少占画面 2%）',
      })
      return
    }
    setCourtPoly(poly.map((p) => [p[0], p[1]] as [number, number]))
    toast({
      kind: 'success',
      title: '已保存手动标定',
      detail:
        poly.length > 4
          ? `${poly.length} 个点的多边形已写入工程，下次分析直接用它圈定场地`
          : '下次分析会直接用这个范围（不再自动猜测）',
    })
    onClose()
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="手动标定场地"
      subtitle="沿着球场边界点一圈：拖顶点微调，在边的中点上按「+」加点，选中顶点后按「−」删点"
      width={880}
      footer={
        <div className="flex items-center justify-between gap-3">
          <div className="text-[11px] text-ink-500">
            {!valid
              ? '区域太小，请把点拉开'
              : `${poly.length} 个点 · 占画面 ${(area * 100).toFixed(0)}%` +
                (poly.length > 4 ? ` · 弯边约 ${(bulge * 100).toFixed(0)}%` : '')}
          </div>
          <div className="flex flex-wrap gap-2">
            <Button variant="ghost" onClick={() => setPoly(defaultQuad())}>
              <RotateCcw size={13} /> 复位
            </Button>
            <Button
              variant="ghost"
              disabled={poly.length >= MAX_POINTS}
              onClick={() => {
                const next = subdivideAll(poly)
                setPoly(next)
                setSelected(null)
                toast({
                  kind: 'info',
                  title: '每条边都加了一个中点',
                  detail: `现在是 ${next.length} 个点，把弯边上的点拖到边界上即可`,
                })
              }}
              title="全景 / 鱼眼素材的边界是弯的，先在每条边加中点再微调最省事"
            >
              <Spline size={13} /> 每条边加中点
            </Button>
            <Button
              variant="ghost"
              onClick={() => {
                setCourtPoly(null)
                toast({ kind: 'info', title: '已清除手动标定', detail: '下次分析改为自动识别场地' })
                onClose()
              }}
            >
              <X size={13} /> 清除
            </Button>
            <Button variant="primary" onClick={save} disabled={!valid}>
              <Crosshair size={13} /> 保存并用于下次分析
            </Button>
          </div>
        </div>
      }
    >
      <div className="mb-3 rounded-xl border border-white/8 bg-white/[0.02] px-3.5 py-2.5 text-[11.5px] leading-relaxed text-ink-400">
        标定结果只影响「在哪里找比赛球员、谁离相机更近、竖屏往哪边裁」，
        不会改动原始视频。<span className="text-court-300">全景 / 鱼眼素材请多加几个点</span>
        —— 边界在画面里是弯的，用四个角会把边角切掉（那里往往正是看台和其他场地的人）。
        如果这张画面上看不清场地，可以先在预览里拖到场地完整出现的时刻，再重新打开本窗口。
      </div>

      <div className="flex items-center justify-center rounded-xl border border-white/8 bg-black p-2">
        <div
          className="relative touch-none select-none"
          style={{ aspectRatio: `${mediaAspect}`, maxHeight: '58vh', width: '100%' }}
          ref={boxRef}
          onPointerMove={onPointerMove}
          onPointerUp={() => setDrag(null)}
          onPointerLeave={() => setDrag(null)}
          onDoubleClick={(e) => {
            // 双击空白处＝在最长的边上加一个点（不用去够那个小「+」按钮）
            const r = boxRef.current?.getBoundingClientRect()
            if (!r) return
            const x = (e.clientX - r.left) / r.width
            const y = (e.clientY - r.top) / r.height
            let best = 0
            let bestD = Infinity
            for (let i = 0; i < poly.length; i++) {
              const [x1, y1] = poly[i]
              const [x2, y2] = poly[(i + 1) % poly.length]
              const mx = (x1 + x2) / 2
              const my = (y1 + y2) / 2
              // 横向按画面宽高比加权：归一化坐标下 1 个单位的横向距离
              // 在屏幕上比纵向长，不加权会在宽画面上选错边
              const d = Math.hypot((x - mx) * mediaAspect, y - my)
              if (d < bestD) {
                bestD = d
                best = i
              }
            }
            addPoint(best)
          }}
        >
          {posterUrl ? (
            <img
              src={posterUrl}
              alt=""
              draggable={false}
              className="pointer-events-none absolute inset-0 h-full w-full object-contain"
            />
          ) : (
            <div className="absolute inset-0 grid place-items-center text-[12px] text-ink-500">
              请先导入素材
            </div>
          )}

          <svg
            className="pointer-events-none absolute inset-0 h-full w-full"
            viewBox="0 0 1 1"
            preserveAspectRatio="none"
          >
            <polygon
              points={poly.map(([x, y]) => `${x},${y}`).join(' ')}
              fill="rgba(56,224,162,0.14)"
              stroke="#38e0a2"
              strokeWidth={0.004}
              vectorEffect="non-scaling-stroke"
            />
          </svg>

          {/* 边的中点：点一下就在那里插一个顶点 */}
          {poly.map(([x1, y1], i) => {
            const [x2, y2] = poly[(i + 1) % poly.length]
            const mx = (x1 + x2) / 2
            const my = (y1 + y2) / 2
            const len = Math.hypot(x2 - x1, y2 - y1)
            // 太短的边就不显示「+」了，否则按钮会挤在一起点不准
            if (len < 0.05) return null
            return (
              <button
                key={`mid-${i}`}
                onClick={() => addPoint(i)}
                title="在这条边的中点插入一个顶点（边界弯的时候用）"
                className={cn(
                  'absolute z-10 grid h-4 w-4 -translate-x-1/2 -translate-y-1/2 place-items-center',
                  'rounded-full border border-court-400/60 bg-ink-950/70 text-court-300',
                  'opacity-60 transition-opacity hover:opacity-100 hover:bg-court-500/30',
                )}
                style={{ left: `${mx * 100}%`, top: `${my * 100}%` }}
              >
                <Plus size={9} />
              </button>
            )
          })}

          {poly.map(([x, y], i) => (
            <button
              key={`v-${i}`}
              onPointerDown={(e) => {
                e.preventDefault()
                setDrag(i)
                setSelected(i)
              }}
              onFocus={() => setSelected(i)}
              className={cn(
                'absolute z-20 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 shadow-lg',
                'h-6 w-6 cursor-grab active:cursor-grabbing',
                drag === i
                  ? 'border-white bg-court-400'
                  : selected === i
                    ? 'border-white/80 bg-court-500/60'
                    : 'border-court-300 bg-ink-900/85 hover:bg-court-500/40',
              )}
              style={{ left: `${x * 100}%`, top: `${y * 100}%` }}
              title={`顶点 ${i + 1}（拖动移动；选中后可用下面的「−」删除）`}
            >
              <span className="mono text-[9px] text-white">{i + 1}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="mt-2.5 flex flex-wrap items-center gap-x-4 gap-y-2 text-[11px] text-ink-500">
        <span className="inline-flex items-center gap-1.5">
          <Wand2 size={11} /> 前两个点是靠近镜头的两个角，后面是远端；点的顺序会被自动整理
        </span>
        <span className="inline-flex items-center gap-1.5">
          <Plus size={11} /> 边的中点＝加点
        </span>
        <Button
          variant="ghost"
          size="sm"
          disabled={selected === null || poly.length <= MIN_POINTS}
          onClick={() => selected !== null && removePoint(selected)}
          title="删掉当前选中的顶点"
        >
          <Minus size={12} />
          {selected === null ? '删点（先选一个点）' : `删掉第 ${selected + 1} 个点`}
        </Button>
        <span className="text-ink-600">点数 {poly.length} / {MAX_POINTS}</span>
      </div>
    </Modal>
  )
}
