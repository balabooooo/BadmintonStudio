/** WebSocket 客户端：断线自动重连 + 订阅分发。 */

import type { JobInfo } from './types'
import { getLang } from '../i18n'

export type WsMessage =
  | { type: 'hello'; version: string }
  | { type: 'pong'; t: number }
  | { type: 'job'; job: JobInfo }
  | { type: 'media'; project_id: string; media: any }
  | { type: 'timeline'; project_id: string; timeline: any }
  | { type: 'analysis'; project_id: string; media_id: string; result: any }

type Handler = (m: WsMessage) => void

class WsClient {
  private ws: WebSocket | null = null
  private handlers = new Set<Handler>()
  private retry = 0
  private timer: number | null = null
  private pingTimer: number | null = null
  connected = false

  connect() {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) return
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const url = `${proto}://${location.host}/ws?lang=${encodeURIComponent(getLang())}`
    try {
      this.ws = new WebSocket(url)
    } catch {
      this.scheduleReconnect()
      return
    }
    this.ws.onopen = () => {
      this.connected = true
      this.retry = 0
      this.emit({ type: 'hello', version: 'connected' })
      this.pingTimer = window.setInterval(() => {
        try {
          this.ws?.send(JSON.stringify({ type: 'ping' }))
        } catch {
          /* ignore */
        }
      }, 20_000)
    }
    this.ws.onmessage = (e) => {
      try {
        this.emit(JSON.parse(e.data) as WsMessage)
      } catch {
        /* ignore */
      }
    }
    this.ws.onclose = () => {
      this.connected = false
      if (this.pingTimer) window.clearInterval(this.pingTimer)
      this.scheduleReconnect()
    }
    this.ws.onerror = () => {
      try {
        this.ws?.close()
      } catch {
        /* ignore */
      }
    }
  }

  /** 语言切换后重新建立连接，让 URL 带上新的 lang 参数。 */
  reconnect() {
    const old = this.ws
    this.ws = null
    if (old) {
      old.onclose = null
      old.onerror = null
      try {
        old.close()
      } catch {
        /* ignore */
      }
    }
    if (this.timer) {
      window.clearTimeout(this.timer)
      this.timer = null
    }
    if (this.pingTimer) {
      window.clearInterval(this.pingTimer)
      this.pingTimer = null
    }
    this.retry = 0
    this.connect()
  }

  private scheduleReconnect() {
    if (this.timer) return
    this.retry = Math.min(this.retry + 1, 8)
    const delay = Math.min(800 * this.retry, 6000)
    this.timer = window.setTimeout(() => {
      this.timer = null
      this.connect()
    }, delay)
  }

  private emit(m: WsMessage) {
    this.handlers.forEach((h) => {
      try {
        h(m)
      } catch {
        /* ignore */
      }
    })
  }

  subscribe(h: Handler): () => void {
    this.handlers.add(h)
    return () => this.handlers.delete(h)
  }
}

export const ws = new WsClient()
