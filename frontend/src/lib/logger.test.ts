import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  clearLogs,
  createLogger,
  downloadLogs,
  formatLogs,
  getLogs,
} from './logger'

describe('frontend logger', () => {
  beforeEach(() => {
    clearLogs()
    window.localStorage.removeItem('bms.debug')
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('buffers every level regardless of console verbosity', () => {
    const log = createLogger('scope-a')
    log.debug('d-msg')
    log.info('i-msg')
    log.warn('w-msg')
    log.error('e-msg')
    const logs = getLogs()
    expect(logs.map((l) => [l.level, l.msg])).toEqual([
      ['debug', 'd-msg'],
      ['info', 'i-msg'],
      ['warn', 'w-msg'],
      ['error', 'e-msg'],
    ])
    expect(logs.every((l) => typeof l.ts === 'number' && l.scope === 'scope-a')).toBe(true)
  })

  it('hides debug from the console by default, shows it with bms.debug=1', () => {
    const debugSpy = vi.spyOn(console, 'debug').mockImplementation(() => undefined)
    createLogger('s').debug('hidden')
    expect(debugSpy).not.toHaveBeenCalled()

    window.localStorage.setItem('bms.debug', '1')
    createLogger('s').debug('shown')
    expect(debugSpy).toHaveBeenCalledTimes(1)
  })

  it('serializes Error and circular data without throwing', () => {
    const log = createLogger('s')
    const circular: Record<string, unknown> = {}
    circular.self = circular
    expect(() => {
      log.error('boom', { err: new Error('x'), circular })
    }).not.toThrow()
    const text = formatLogs()
    expect(text).toContain('boom')
    expect(text).toContain('[circular]')
    expect(text).toContain('"name":"Error"')
  })

  it('caps the ring buffer at the newest 1500 records', () => {
    const log = createLogger('bulk')
    for (let i = 0; i < 1600; i++) log.debug(`m${i}`)
    const logs = getLogs()
    expect(logs).toHaveLength(1500)
    expect(logs[0].msg).toBe('m100')
    expect(logs[1499].msg).toBe('m1599')
  })

  it('formatLogs includes level, scope and message; clearLogs empties the buffer', () => {
    createLogger('fmt').info('hello', { a: 1 })
    const text = formatLogs()
    expect(text).toContain('INFO')
    expect(text).toContain('[fmt]')
    expect(text).toContain('hello')
    expect(text).toContain('{"a":1}')
    clearLogs()
    expect(getLogs()).toEqual([])
  })

  it('downloadLogs triggers an anchor click and returns true', () => {
    const click = vi.fn()
    const create = vi.spyOn(document, 'createElement').mockReturnValue({
      click,
      remove: vi.fn(),
      set href(_v: string) {},
      get href() {
        return ''
      },
      set download(_v: string) {},
    } as unknown as HTMLAnchorElement)
    const append = vi.spyOn(document.body, 'appendChild').mockImplementation(() => null as unknown as Node)
    const ok = downloadLogs('x.txt')
    expect(ok).toBe(true)
    expect(create).toHaveBeenCalledWith('a')
    expect(click).toHaveBeenCalledTimes(1)
    append.mockRestore()
  })
})
