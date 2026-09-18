export function cn(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(' ')
}

/** 00:12.34 / 1:02:03.45 */
export function timecode(sec: number, withFrames = true, fps = 30): string {
  if (!isFinite(sec) || sec < 0) sec = 0
  const h = Math.floor(sec / 3600)
  const m = Math.floor((sec % 3600) / 60)
  const s = Math.floor(sec % 60)
  const f = Math.floor((sec - Math.floor(sec)) * fps)
  const base = h > 0
    ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
    : `${m}:${String(s).padStart(2, '0')}`
  return withFrames ? `${base}.${String(f).padStart(2, '0')}` : base
}

/** 12.3s / 1分23秒 */
export function humanDuration(sec: number): string {
  if (!isFinite(sec) || sec < 0) sec = 0
  if (sec < 60) return `${sec.toFixed(1)}秒`
  let m = Math.floor(sec / 60)
  let s = Math.round(sec % 60)
  // 四舍五入会把 119.7 的秒数进位成 60，直接拼会得到「1分60秒」
  if (s === 60) {
    s = 0
    m += 1
  }
  if (m < 60) return `${m}分${String(s).padStart(2, '0')}秒`
  const h = Math.floor(m / 60)
  return `${h}小时${String(m % 60).padStart(2, '0')}分`
}

export function bytes(n: number): string {
  if (!n || n < 1) return '0 B'
  const u = ['B', 'KB', 'MB', 'GB', 'TB']
  // 0 < n < 1 时 log 为负，i 会变成 -1，输出「0.5 undefined」
  const i = Math.max(0, Math.min(u.length - 1, Math.floor(Math.log(n) / Math.log(1024))))
  return `${(n / Math.pow(1024, i)).toFixed(i === 0 ? 0 : 1)} ${u[i]}`
}

export function relTime(ms: number): string {
  const d = Date.now() - ms
  if (d < 60_000) return '刚刚'
  if (d < 3_600_000) return `${Math.floor(d / 60_000)} 分钟前`
  if (d < 86_400_000) return `${Math.floor(d / 3_600_000)} 小时前`
  if (d < 7 * 86_400_000) return `${Math.floor(d / 86_400_000)} 天前`
  return new Date(ms).toLocaleDateString('zh-CN')
}

/** 分数 -> 颜色（红→黄→绿），用于评分徽章 */
export function scoreColor(score: number): string {
  const s = Math.max(0, Math.min(100, score)) / 100
  // 0 -> #ff5470, 0.5 -> #ffb020, 1 -> #38e0a2
  const stops: [number, [number, number, number]][] = [
    [0, [255, 84, 112]],
    [0.55, [255, 176, 32]],
    [0.8, [150, 220, 90]],
    [1, [56, 224, 162]],
  ]
  let a = stops[0]
  let b = stops[stops.length - 1]
  for (let i = 0; i < stops.length - 1; i++) {
    if (s >= stops[i][0] && s <= stops[i + 1][0]) {
      a = stops[i]
      b = stops[i + 1]
      break
    }
  }
  const t = (s - a[0]) / Math.max(1e-6, b[0] - a[0])
  const c = a[1].map((v, i) => Math.round(v + (b[1][i] - v) * t))
  return `rgb(${c[0]} ${c[1]} ${c[2]})`
}

export function scoreGrade(score: number): string {
  if (score >= 85) return 'S'
  if (score >= 72) return 'A'
  if (score >= 58) return 'B'
  if (score >= 42) return 'C'
  return 'D'
}

export const TAG_COLORS: Record<string, string> = {
  超长多拍: '#a874ff',
  多拍: '#5c9dff',
  快节奏: '#ffb020',
  末段提速: '#ff8a3d',
  高强度跑动: '#38e0a2',
  高速球: '#16c98a',
  长回合: '#3b7ff0',
  短回合: '#6b7787',
  高分: '#ffd12e',
  低置信: '#ff5470',
}

export function tagColor(tag: string): string {
  return TAG_COLORS[tag] || '#6b7787'
}

export function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v))
}

export function median(a: number[]): number {
  if (!a.length) return 0
  const s = [...a].sort((x, y) => x - y)
  return s[Math.floor(s.length / 2)]
}
