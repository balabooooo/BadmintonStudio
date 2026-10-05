/** 通用 UI 原语：按钮、徽章、滑块、进度、弹窗、提示等。 */

import { AnimatePresence, motion } from 'motion/react'
import { Gauge } from 'lucide-react'
import {
  createContext,
  useContext,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { createPortal } from 'react-dom'
import { cn } from '../lib/format'
import { SPEEDS } from '../lib/playback'
import { useStore } from '../store/useStore'
import { useT } from '../i18n/useT'

/* ------------------------------------------------------------------ 按钮 */

type BtnVariant = 'primary' | 'ghost' | 'outline' | 'danger' | 'subtle'
type BtnSize = 'sm' | 'md' | 'lg' | 'icon'

const V: Record<BtnVariant, string> = {
  primary:
    'bg-gradient-to-b from-court-400 to-court-600 text-ink-950 font-semibold shadow-[0_6px_20px_-6px_rgb(22_201_138/0.7)] hover:from-court-300 hover:to-court-500',
  ghost: 'text-ink-200 hover:bg-white/8 hover:text-white',
  outline: 'border border-white/12 text-ink-100 hover:border-court-500/60 hover:bg-court-500/10',
  danger: 'bg-rose-hot/15 text-rose-hot border border-rose-hot/30 hover:bg-rose-hot/25',
  subtle: 'bg-white/6 text-ink-100 hover:bg-white/12',
}
const S: Record<BtnSize, string> = {
  sm: 'h-7 px-2.5 text-[12px] rounded-lg gap-1.5',
  md: 'h-9 px-3.5 text-[13px] rounded-[10px] gap-2',
  lg: 'h-11 px-5 text-[14px] rounded-xl gap-2',
  icon: 'h-8 w-8 rounded-lg justify-center',
}

export function Button({
  variant = 'subtle',
  size = 'md',
  className,
  children,
  loading,
  ...rest
}: {
  variant?: BtnVariant
  size?: BtnSize
  loading?: boolean
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...rest}
      disabled={rest.disabled || loading}
      className={cn(
        'inline-flex select-none items-center justify-center whitespace-nowrap transition-all duration-150 active:scale-[0.97]',
        'disabled:cursor-not-allowed disabled:opacity-40 disabled:active:scale-100',
        V[variant],
        S[size],
        className,
      )}
    >
      {loading && (
        <span className="spin h-3.5 w-3.5 rounded-full border-[2px] border-current border-t-transparent" />
      )}
      {children}
    </button>
  )
}

/* ------------------------------------------------------------------ 卡片 */

export function Card({
  className,
  children,
  hover,
  ...rest
}: { hover?: boolean } & React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div {...rest} className={cn('panel', hover && 'panel-hover', className)}>
      {children}
    </div>
  )
}

export function SectionTitle({
  children,
  right,
  className,
}: {
  children: ReactNode
  right?: ReactNode
  className?: string
}) {
  return (
    <div className={cn('mb-2 flex items-center justify-between gap-2', className)}>
      <div className="flex items-center gap-2 text-[11px] font-semibold tracking-[0.14em] text-ink-400 uppercase">
        {children}
      </div>
      {right}
    </div>
  )
}

/* ------------------------------------------------------------------ 徽章 */

export function Badge({
  children,
  color,
  className,
  dot,
}: {
  children: ReactNode
  color?: string
  className?: string
  dot?: boolean
}) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-md px-1.5 py-[2px] text-[10.5px] font-medium leading-4 tabular',
        !color && 'bg-white/8 text-ink-200',
        className,
      )}
      style={color ? { background: `${color}22`, color, boxShadow: `inset 0 0 0 1px ${color}44` } : undefined}
    >
      {dot && <span className="h-1.5 w-1.5 rounded-full" style={{ background: color || '#8f9aa9' }} />}
      {children}
    </span>
  )
}

export function ScoreRing({
  score,
  size = 44,
  label,
  color,
}: {
  score: number
  size?: number
  label?: string
  color: string
}) {
  const r = (size - 6) / 2
  const c = 2 * Math.PI * r
  const pct = Math.max(0, Math.min(100, score)) / 100
  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgb(255 255 255 / 0.08)" strokeWidth={3} />
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={3}
          strokeLinecap="round"
          strokeDasharray={c}
          initial={{ strokeDashoffset: c }}
          animate={{ strokeDashoffset: c * (1 - pct) }}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="tabular font-bold leading-none" style={{ color, fontSize: size * 0.3 }}>
          {score.toFixed(0)}
        </span>
        {label && <span className="mt-0.5 text-[8px] text-ink-400">{label}</span>}
      </div>
    </div>
  )
}

export function ScoreBar({ value, color, height = 4 }: { value: number; color: string; height?: number }) {
  return (
    <div className="w-full overflow-hidden rounded-full bg-white/8" style={{ height }}>
      <motion.div
        className="h-full rounded-full"
        style={{ background: color }}
        initial={{ width: 0 }}
        animate={{ width: `${Math.max(0, Math.min(100, value))}%` }}
        transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
      />
    </div>
  )
}

/* ------------------------------------------------------------------ 进度 */

export function Progress({
  value,
  className,
  indeterminate,
  color = 'var(--color-court-400)',
}: {
  value: number
  className?: string
  indeterminate?: boolean
  color?: string
}) {
  return (
    <div className={cn('relative h-1.5 w-full overflow-hidden rounded-full bg-white/8', className)}>
      {indeterminate ? (
        <div className="sheen absolute inset-0 bg-white/10" />
      ) : (
        <motion.div
          className="h-full rounded-full"
          style={{ background: `linear-gradient(90deg, ${color}, color-mix(in oklab, ${color} 60%, white))` }}
          animate={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
          transition={{ duration: 0.35, ease: 'easeOut' }}
        />
      )}
    </div>
  )
}

export function Spinner({ size = 16, className }: { size?: number; className?: string }) {
  return (
    <span
      className={cn('spin inline-block rounded-full border-[2px] border-court-400/30 border-t-court-400', className)}
      style={{ width: size, height: size }}
    />
  )
}

/* ------------------------------------------------------------------ 滑块 */

export function Slider({
  value,
  min,
  max,
  step = 1,
  onChange,
  onStart,
  label,
  format,
  disabled,
  hint,
}: {
  value: number
  min: number
  max: number
  step?: number
  onChange: (v: number) => void
  /** 开始拖动时触发一次，用于记录撤销快照（拖动过程中用 pushHistory=false） */
  onStart?: () => void
  label: string
  format?: (v: number) => string
  disabled?: boolean
  hint?: string
}) {
  const id = useId()
  return (
    <div className={cn('select-none', disabled && 'opacity-45')}>
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <label htmlFor={id} className="text-[12px] text-ink-200" title={hint}>
          {label}
        </label>
        <span className="mono text-[11.5px] text-court-300">{format ? format(value) : value}</span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onPointerDown={onStart}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </div>
  )
}

/* ------------------------------------------------------------------ 倍速 */

/** 预设倍速菜单：Gauge 按钮 + 向上/向下展开的档位弹层。
 *
 * 工作室播放器和标注页共用，避免两处各写一套（标注页此前用裸 range，
 * 还被全局 `input[type=range]{width:100%}` 撑成整行，几乎看不出是倍速）。
 *
 * `direction` 决定弹层展开方向：控件在底部用 up，在顶部用 down。
 */
export function SpeedMenu({
  value,
  onChange,
  title,
  direction = 'up',
}: {
  value: number
  onChange: (v: number) => void
  /** 按钮 tooltip 文案 */
  title: string
  direction?: 'up' | 'down'
}) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)
  const up = direction === 'up'

  // Close on any pointer press outside this control (button + menu).
  // Do NOT use a body-level fixed overlay for this: the page shell `.anim-in`
  // animates transform, so it is its own stacking context and the portal overlay
  // (z-40 at root level) paints above the panel (z-50 trapped inside), swallowing
  // every option click — the menu opens but no speed can be picked.
  useEffect(() => {
    if (!open) return
    const onPointerDown = (e: PointerEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('pointerdown', onPointerDown)
    return () => document.removeEventListener('pointerdown', onPointerDown)
  }, [open])

  return (
    <div ref={rootRef} className="relative">
      <Tooltip content={title} side={up ? 'top' : 'bottom'}>
        <Button variant="ghost" size="sm" onClick={() => setOpen((v) => !v)}>
          <Gauge size={13} />
          <span className="mono">{value}×</span>
        </Button>
      </Tooltip>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: up ? 6 : -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: up ? 6 : -6 }}
            className={cn(
              'panel absolute z-50 flex flex-col p-1',
              up ? 'right-0 bottom-full mb-1.5' : 'left-0 top-full mt-1.5',
            )}
          >
            {SPEEDS.map((s) => (
              <button
                key={s}
                onClick={() => {
                  onChange(s)
                  setOpen(false)
                }}
                className={cn(
                  'mono rounded-md px-3 py-1 text-[11.5px] transition-colors',
                  value === s ? 'bg-court-500/20 text-court-300' : 'text-ink-300 hover:bg-white/8',
                )}
              >
                {s}×
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

/* ------------------------------------------------------------------ 开关 */

export function Toggle({
  checked,
  onChange,
  label,
  hint,
  disabled,
}: {
  checked: boolean
  onChange: (v: boolean) => void
  label: string
  hint?: string
  disabled?: boolean
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={() => onChange(!checked)}
      title={hint}
      className={cn(
        'flex w-full items-center justify-between gap-3 rounded-lg px-2.5 py-2 text-left transition-colors',
        'hover:bg-white/5 disabled:cursor-not-allowed disabled:opacity-40',
      )}
    >
      <span className="min-w-0 flex-1">
        <span className="block truncate text-[12.5px] text-ink-100">{label}</span>
        {hint && <span className="mt-0.5 block truncate text-[10.5px] text-ink-400">{hint}</span>}
      </span>
      <span
        className={cn(
          'relative h-[18px] w-[32px] shrink-0 rounded-full transition-colors duration-200',
          checked ? 'bg-court-500' : 'bg-white/14',
        )}
      >
        <motion.span
          layout
          transition={{ type: 'spring', stiffness: 620, damping: 34 }}
          className="absolute top-[2px] h-[14px] w-[14px] rounded-full bg-white shadow"
          style={{ left: checked ? 16 : 2 }}
        />
      </span>
    </button>
  )
}

/* ------------------------------------------------------------------ 分段控件 */

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  className,
  size = 'md',
}: {
  value: T
  options: { value: T; label: string; hint?: string }[]
  onChange: (v: T) => void
  className?: string
  size?: 'sm' | 'md'
}) {
  return (
    <div className={cn('relative inline-flex shrink-0 rounded-[10px] bg-white/6 p-[3px]', className)}>
      {options.map((o) => (
        <button
          key={o.value}
          title={o.hint}
          onClick={() => onChange(o.value)}
          className={cn(
            'relative rounded-lg whitespace-nowrap transition-colors',
            size === 'sm' ? 'px-2 py-1 text-[11px]' : 'px-3 py-1.5 text-[12px]',
            value === o.value ? 'text-ink-950' : 'text-ink-300 hover:text-ink-100',
          )}
        >
          {value === o.value && (
            <motion.span
              layoutId={`seg-${options.map((x) => x.value).join('')}`}
              className="absolute inset-0 rounded-lg bg-gradient-to-b from-court-300 to-court-500"
              transition={{ type: 'spring', stiffness: 520, damping: 36 }}
            />
          )}
          <span className="relative font-medium">{o.label}</span>
        </button>
      ))}
    </div>
  )
}

/* ------------------------------------------------------------------ 弹窗 */

/**
 * 已打开弹窗的栈（按打开顺序）。
 *
 * 每个 Modal 都监听 window 的 Escape；堆叠时（例如场地编辑器上又弹出确认框）
 * 两个监听器会同时触发、一次 Escape 关掉两层。用这个栈记录打开顺序，只让
 * 最上层响应。卸载 / 关闭时对应 id 出栈。
 */
const modalStack: symbol[] = []

export function Modal({
  open,
  onClose,
  title,
  subtitle,
  children,
  width = 640,
  footer,
}: {
  open: boolean
  onClose: () => void
  title: string
  subtitle?: string
  children: ReactNode
  width?: number
  footer?: ReactNode
}) {
  const tr = useT()
  // onClose 可能是每次渲染新建的内联函数；放进 ref，让下面的 effect 只依赖 open，
  // 这样堆叠中下层弹窗重渲染时不会重新入栈、把自己顶到最上层。
  const closeRef = useRef(onClose)
  useEffect(() => {
    closeRef.current = onClose
  })
  const [stackId] = useState(() => Symbol('modal'))
  useEffect(() => {
    if (!open) return
    modalStack.push(stackId)
    const h = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      if (modalStack[modalStack.length - 1] !== stackId) return
      e.stopPropagation()
      closeRef.current()
    }
    window.addEventListener('keydown', h)
    return () => {
      window.removeEventListener('keydown', h)
      const i = modalStack.indexOf(stackId)
      if (i >= 0) modalStack.splice(i, 1)
    }
  }, [open, stackId])

  // 弹窗也挂到 body：页面外壳（.anim-in）上带着 transform 动画，
  // 留在它里面的话 fixed 会以那个盒子为参照，遮罩盖不住顶栏、定位也会偏。
  return createPortal(
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center p-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <div className="absolute inset-0 bg-ink-950/72 backdrop-blur-[3px]" onClick={onClose} />
          <motion.div
            role="dialog"
            aria-modal="true"
            aria-label={title}
            initial={{ opacity: 0, scale: 0.975, y: 14 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.98, y: 8 }}
            transition={{ duration: 0.24, ease: [0.22, 1, 0.36, 1] }}
            className="panel relative z-10 flex max-h-[86vh] flex-col overflow-hidden"
            style={{ width, boxShadow: 'var(--shadow-pop)' }}
          >
            <div className="flex items-start justify-between gap-4 border-b border-white/8 px-5 py-4">
              <div className="min-w-0">
                <h2 className="truncate text-[15px] font-semibold text-white">{title}</h2>
                {subtitle && <p className="mt-0.5 text-[11.5px] text-ink-400">{subtitle}</p>}
              </div>
              <Button variant="ghost" size="icon" onClick={onClose} aria-label={tr('common.close')}>
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 6 6 18M6 6l12 12" />
                </svg>
              </Button>
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">{children}</div>
            {footer && <div className="border-t border-white/8 px-5 py-3">{footer}</div>}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>,
    document.body,
  )
}

/* ------------------------------------------------------------------ 提示气泡 */

/**
 * 悬停提示。
 *
 * 气泡用 portal 挂到 document.body：
 * 时间线里到处是 sticky + z-index 的层（标尺 z-30、播放头 z-40、轨道标签 z-20），
 * 只要气泡还留在这些元素的子树里，它的 z-90 就只是「层内第一」——标尺照样盖在
 * 气泡上面。挂到 body 之后，气泡才真正浮在整个界面之上，也不会被时间线的
 * overflow 裁掉。
 */
export function Tooltip({
  children,
  content,
  side = 'top',
  width = 264,
  className,
  block,
  kbd,
}: {
  children: ReactNode
  content: ReactNode
  side?: 'top' | 'bottom' | 'left' | 'right'
  width?: number
  /** 挂到锚点（那个 inline-flex 的 span）上，需要时用来让它撑满父容器，例如 flex-1 */
  className?: string
  /**
   * 锚点用块级盒子（自己占一行）。
   * 里面放 w-full 的元素时必须打开：inline-flex 的宽度是「缩到内容那么宽」，
   * 子元素的 w-full 会算成 0，整块就看不见了。
   */
  block?: boolean
  /** 快捷键提示：一个或多个按键，渲染成 Kbd 徽章跟在说明下方。 */
  kbd?: string | string[]
}) {
  const [show, setShow] = useState(false)
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null)
  const anchor = useRef<HTMLSpanElement>(null)
  const bubble = useRef<HTMLSpanElement>(null)

  // 在测量真实尺寸之后再定位，并夹在视口内——否则靠右侧/顶部的提示框会跑出窗口。
  // 气泡是 fixed 定位，所以滚动 / 改窗口大小都要重新量一次，否则会留在原地飘着。
  useLayoutEffect(() => {
    if (!show) return
    const place = () => {
      const a = anchor.current?.getBoundingClientRect()
      if (!a) return
      // 用 offsetWidth/offsetHeight 量尺寸：气泡的入场动画带着 scale(0.97)，
      // getBoundingClientRect 会小 3%，夹取边界就会算偏、气泡探出窗口。
      const bw = bubble.current?.offsetWidth || width
      const bh = bubble.current?.offsetHeight ?? 40
      const pad = 8
      let left: number
      let top: number
      if (side === 'left' || side === 'right') {
        left = side === 'right' ? a.right + 8 : a.left - bw - 8
        top = a.top + a.height / 2 - bh / 2
      } else {
        left = a.left + a.width / 2 - bw / 2
        top = side === 'bottom' ? a.bottom + 8 : a.top - bh - 8
      }
      left = Math.max(pad, Math.min(left, window.innerWidth - bw - pad))
      top = Math.max(pad, Math.min(top, window.innerHeight - bh - pad))
      setPos({ left, top })
    }
    place()
    window.addEventListener('scroll', place, true)
    window.addEventListener('resize', place)
    return () => {
      window.removeEventListener('scroll', place, true)
      window.removeEventListener('resize', place)
    }
  }, [show, side, width])

  return (
    <span
      ref={anchor}
      className={cn('relative', block ? 'flex' : 'inline-flex', className)}
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => {
        setShow(false)
        setPos(null)
      }}
    >
      {children}
      {createPortal(
        <AnimatePresence>
          {show && (
            <motion.span
              ref={bubble}
              initial={{ opacity: 0, scale: 0.97 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              transition={{ duration: 0.14 }}
              style={{
                position: 'fixed',
                left: pos?.left ?? -9999,
                top: pos?.top ?? -9999,
                width,
                visibility: pos ? 'visible' : 'hidden',
              }}
              className={cn(
                'pointer-events-none z-[90] rounded-lg border border-white/10 bg-ink-850/96 px-2.5 py-2',
                'text-[11.5px] leading-relaxed text-ink-100 shadow-xl backdrop-blur whitespace-pre-line',
              )}
            >
              {content}
              {kbd && (
                <span className="mt-1.5 flex items-center gap-1 border-t border-white/10 pt-1.5">
                  {(Array.isArray(kbd) ? kbd : [kbd]).map((k) => (
                    <Kbd key={k}>{k}</Kbd>
                  ))}
                </span>
              )}
            </motion.span>
          )}
        </AnimatePresence>,
        document.body,
      )}
    </span>
  )
}

/* ------------------------------------------------------------------ 通知 */

export function ToastHost() {
  const toasts = useStore((s) => s.toasts)
  const dismiss = useStore((s) => s.dismissToast)
  const colors: Record<string, string> = {
    info: '#5c9dff',
    success: '#38e0a2',
    warn: '#ffb020',
    error: '#ff5470',
  }
  return (
    <div className="pointer-events-none fixed right-4 bottom-4 z-[80] flex w-[330px] flex-col gap-2">
      <AnimatePresence>
        {toasts.map((t) => (
          <motion.div
            key={t.id}
            layout
            initial={{ opacity: 0, x: 40, scale: 0.96 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 40, scale: 0.96 }}
            transition={{ duration: 0.26, ease: [0.22, 1, 0.36, 1] }}
            onClick={() => dismiss(t.id)}
            className="panel pointer-events-auto cursor-pointer overflow-hidden px-3.5 py-2.5"
          >
            <div className="flex gap-2.5">
              <span
                className="mt-[5px] h-2 w-2 shrink-0 rounded-full"
                style={{ background: colors[t.kind], boxShadow: `0 0 10px ${colors[t.kind]}` }}
              />
              <div className="min-w-0 flex-1">
                <div className="text-[12.5px] font-medium text-white">{t.title}</div>
                {t.detail && <div className="mt-0.5 break-words text-[11px] text-ink-300">{t.detail}</div>}
              </div>
            </div>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  )
}

/* ------------------------------------------------------------------ 空状态 */

export function Empty({
  icon,
  title,
  desc,
  action,
}: {
  icon?: ReactNode
  title: string
  desc?: string
  action?: ReactNode
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 px-6 py-12 text-center">
      {icon && <div className="text-ink-600">{icon}</div>}
      <div>
        <div className="text-[13.5px] font-medium text-ink-200">{title}</div>
        {desc && <div className="mt-1 max-w-[380px] text-[12px] leading-relaxed text-ink-400">{desc}</div>}
      </div>
      {action}
    </div>
  )
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn('skeleton rounded-lg', className)} />
}

/* ------------------------------------------------------------------ 统计块 */

export function Stat({
  label,
  value,
  unit,
  color,
  hint,
}: {
  label: string
  value: string | number
  unit?: string
  color?: string
  hint?: string
}) {
  return (
    <Tooltip content={hint || label}>
      <div className="panel-flat px-3 py-2.5">
        <div className="text-[10.5px] tracking-wide text-ink-400">{label}</div>
        <div className="mt-1 flex items-baseline gap-1">
          <span className="tabular text-[19px] font-semibold leading-none" style={{ color: color || '#dde4ec' }}>
            {value}
          </span>
          {unit && <span className="text-[11px] text-ink-400">{unit}</span>}
        </div>
      </div>
    </Tooltip>
  )
}

/* ------------------------------------------------------------------ 上下文菜单 */

export function ContextMenu({
  items,
  x,
  y,
  onClose,
}: {
  items: { label: string; onClick: () => void; danger?: boolean; disabled?: boolean; hint?: string }[]
  x: number
  y: number
  onClose: () => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null)

  // 先量真实尺寸再摆放：右键点在屏幕右下角时菜单会顶出去，一半在窗口外面点不到。
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const r = el.getBoundingClientRect()
    const pad = 8
    // 放不下就翻到光标上方 / 左侧，仍然放不下才夹到视口里
    const left = x + r.width + pad > window.innerWidth ? Math.max(pad, x - r.width) : x
    const top = y + r.height + pad > window.innerHeight ? Math.max(pad, y - r.height) : y
    setPos({
      left: Math.min(left, Math.max(pad, window.innerWidth - r.width - pad)),
      top: Math.min(top, Math.max(pad, window.innerHeight - r.height - pad)),
    })
  }, [x, y])

  useEffect(() => {
    const h = () => onClose()
    const key = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('click', h)
    window.addEventListener('contextmenu', h)
    // 菜单是 fixed 定位：底下的时间线一滚它就悬在原地，不如直接关掉
    window.addEventListener('scroll', h, true)
    window.addEventListener('resize', h)
    window.addEventListener('keydown', key)
    return () => {
      window.removeEventListener('click', h)
      window.removeEventListener('contextmenu', h)
      window.removeEventListener('scroll', h, true)
      window.removeEventListener('resize', h)
      window.removeEventListener('keydown', key)
    }
  }, [onClose])

  return createPortal(
    <div
      ref={ref}
      role="menu"
      className="panel fixed z-[70] min-w-[180px] overflow-hidden py-1"
      style={{
        left: pos?.left ?? x,
        top: pos?.top ?? y,
        visibility: pos ? 'visible' : 'hidden',
        boxShadow: 'var(--shadow-pop)',
      }}
      onClick={(e) => e.stopPropagation()}
    >
      {items.map((it, i) => (
        <button
          key={i}
          disabled={it.disabled}
          onClick={() => {
            it.onClick()
            onClose()
          }}
          className={cn(
            'flex w-full items-center justify-between gap-4 px-3 py-1.5 text-left text-[12.5px] transition-colors',
            it.disabled
              ? 'cursor-not-allowed text-ink-600'
              : it.danger
                ? 'text-rose-hot hover:bg-rose-hot/12'
                : 'text-ink-100 hover:bg-white/8',
          )}
        >
          <span>{it.label}</span>
          {it.hint && <span className="mono text-[10.5px] text-ink-500">{it.hint}</span>}
        </button>
      ))}
    </div>,
    document.body,
  )
}

/* ------------------------------------------------------------------ 键盘提示 */

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="mono rounded border border-white/12 bg-white/6 px-1.5 py-[1px] text-[10.5px] text-ink-300">
      {children}
    </kbd>
  )
}

const ConfirmCtx = createContext<(opts: { title: string; desc?: string; danger?: boolean }) => Promise<boolean>>(
  async () => false,
)

export function useConfirm() {
  return useContext(ConfirmCtx)
}

export function ConfirmProvider({ children }: { children: ReactNode }) {
  const tr = useT()
  // 用队列而不是单个 state：并发 ask() 时（例如确认框还没关又来了一个）
  // 覆盖式写法会把前一个 resolve 丢掉，导致它的 Promise 永远挂着。
  const [queue, setQueue] = useState<
    { title: string; desc?: string; danger?: boolean; resolve: (v: boolean) => void }[]
  >([])

  const ask = useMemo(
    () => (opts: { title: string; desc?: string; danger?: boolean }) =>
      new Promise<boolean>((resolve) => setQueue((q) => [...q, { ...opts, resolve }])),
    [],
  )

  const current = queue[0]
  const settle = (v: boolean) => {
    current?.resolve(v)
    setQueue((q) => q.slice(1))
  }

  return (
    <ConfirmCtx.Provider value={ask}>
      {children}
      <Modal
        open={!!current}
        onClose={() => settle(false)}
        title={current?.title ?? ''}
        subtitle={current?.desc}
        width={440}
        footer={
          <div className="flex justify-end gap-2">
            <Button variant="ghost" onClick={() => settle(false)}>
              {tr('common.cancel')}
            </Button>
            <Button variant={current?.danger ? 'danger' : 'primary'} onClick={() => settle(true)}>
              {tr('common.confirm')}
            </Button>
          </div>
        }
      >
        <p className="text-[12.5px] leading-relaxed text-ink-300">{tr('common.actionImmediate')}</p>
      </Modal>
    </ConfirmCtx.Provider>
  )
}
