/**
 * Frontend debug logging.
 *
 * The desktop app gives users no devtools by default, so `console.*` calls are lost the moment a
 * page is closed. This module keeps an in-memory ring buffer of the last RING_SIZE log records at
 * ALL levels (regardless of console verbosity) and exposes `getLogs()` / `downloadLogs()` plus
 * `window.__bmsLogs` for post-mortem export (Settings → 导出调试日志).
 *
 * Console verbosity is controlled separately:
 * - default: info and above go to the console;
 * - localStorage `bms.debug=1`: debug also goes to the console.
 *
 * Every record carries a scope (module name) and optional structured data. Data is serialized
 * defensively (Errors, circular refs, bigints) so logging itself must never throw.
 */

export type LogLevel = 'debug' | 'info' | 'warn' | 'error'

export interface LogEntry {
  ts: number
  level: LogLevel
  scope: string
  msg: string
  data?: unknown
}

const RING_SIZE = 1500
const LEVEL_WEIGHT: Record<LogLevel, number> = { debug: 10, info: 20, warn: 30, error: 40 }
const DEBUG_STORAGE_KEY = 'bms.debug'

/** Fixed-capacity ring buffer; newest writes overwrite the oldest record. */
class RingBuffer {
  private items: (LogEntry | null)[] = new Array(RING_SIZE).fill(null)
  private head = 0
  private filled = 0

  push(entry: LogEntry): void {
    this.items[this.head] = entry
    this.head = (this.head + 1) % RING_SIZE
    if (this.filled < RING_SIZE) this.filled += 1
  }

  /** Records in chronological (oldest → newest) order. */
  toArray(): LogEntry[] {
    if (this.filled < RING_SIZE) {
      return this.items.slice(0, this.filled) as LogEntry[]
    }
    return [
      ...(this.items.slice(this.head) as LogEntry[]),
      ...(this.items.slice(0, this.head) as LogEntry[]),
    ]
  }

  clear(): void {
    this.items = new Array(RING_SIZE).fill(null)
    this.head = 0
    this.filled = 0
  }
}

const buffer = new RingBuffer()

export interface Logger {
  debug(msg: string, data?: unknown): void
  info(msg: string, data?: unknown): void
  warn(msg: string, data?: unknown): void
  error(msg: string, data?: unknown): void
}

function debugConsoleEnabled(): boolean {
  try {
    return window.localStorage.getItem(DEBUG_STORAGE_KEY) === '1'
  } catch {
    return false
  }
}

function serializeError(value: unknown): unknown {
  if (value instanceof Error) {
    return { name: value.name, message: value.message, stack: value.stack }
  }
  return value
}

/** JSON stringify that survives circular references, bigints and Error instances. */
function safeStringify(value: unknown): string {
  const seen = new WeakSet<object>()
  try {
    return JSON.stringify(value, (_key, v) => {
      const rv = serializeError(v)
      if (typeof rv === 'bigint') return rv.toString()
      if (rv && typeof rv === 'object') {
        if (seen.has(rv as object)) return '[circular]'
        seen.add(rv as object)
      }
      return rv
    })
  } catch {
    return String(value)
  }
}

function emit(level: LogLevel, scope: string, msg: string, data?: unknown): void {
  const entry: LogEntry = { ts: Date.now(), level, scope, msg }
  if (data !== undefined) entry.data = data
  buffer.push(entry)

  const consoleLevel: LogLevel = debugConsoleEnabled() ? 'debug' : 'info'
  if (LEVEL_WEIGHT[level] < LEVEL_WEIGHT[consoleLevel]) return
  const prefix = `[${scope}]`
  const detail = data !== undefined ? safeStringify(serializeError(data)) : undefined
  const fn = level === 'debug' ? console.debug : level === 'info' ? console.log : level === 'warn' ? console.warn : console.error
  if (detail) fn(`${prefix} ${msg}`, detail)
  else fn(`${prefix} ${msg}`)
}

/** Create a scope-bound logger, e.g. `const log = createLogger('view')`. */
export function createLogger(scope: string): Logger {
  return {
    debug: (msg, data) => emit('debug', scope, msg, data),
    info: (msg, data) => emit('info', scope, msg, data),
    warn: (msg, data) => emit('warn', scope, msg, data),
    error: (msg, data) => emit('error', scope, msg, data),
  }
}

/** All buffered records, oldest first. */
export function getLogs(): LogEntry[] {
  return buffer.toArray()
}

export function clearLogs(): void {
  buffer.clear()
}

/** Format buffered records as a plain-text report (for download / paste into issues). */
export function formatLogs(logs: LogEntry[] = getLogs()): string {
  return logs
    .map((e) => {
      const when = new Date(e.ts).toISOString()
      const data = e.data !== undefined ? ` ${safeStringify(serializeError(e.data))}` : ''
      return `${when} ${e.level.toUpperCase().padEnd(5)} [${e.scope}] ${e.msg}${data}`
    })
    .join('\n')
}

/** Trigger a browser download of the buffered log. Returns false when unavailable (jsdom). */
export function downloadLogs(filename = `bms-debug-${new Date().toISOString().replace(/[:.]/g, '-')}.txt`): boolean {
  if (typeof document === 'undefined' || typeof URL === 'undefined' || typeof Blob === 'undefined') return false
  const blob = new Blob([formatLogs()], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
  return true
}

declare global {
  interface Window {
    __bmsLogs?: {
      get: () => LogEntry[]
      clear: () => void
      download: () => boolean
    }
  }
}

if (typeof window !== 'undefined') {
  window.__bmsLogs = { get: getLogs, clear: clearLogs, download: () => downloadLogs() }
}
