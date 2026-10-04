import type { Clip } from './types'

/** 片段在成片时间轴上占据的时长（变速已折算）。 */
export function clipDur(c: Clip): number {
  return (c.src_out - c.src_in) / (c.speed || 1)
}

/** 成片时间轴上的片段，按 tl_start 排序。 */
export function orderedClips(timeline: { tracks?: { clips?: Clip[] }[] } | null | undefined): Clip[] {
  return (timeline?.tracks?.[0]?.clips ?? []).slice().sort((a, b) => a.tl_start - b.tl_start)
}

/**
 * 把一条轨道的片段按当前顺序首尾相接地重排（磁性时间线），就地修改 track.clips。
 *
 * 成片导出（concat 按顺序无缝拼接）与成片预览（空隙处自动跳到下一段）都把
 * 这条轨当作「无缝序列」处理，数据里留着空隙只会让时间轴显示和实际播放不一致。
 * 所以拖动/裁剪松手、工程加载之后都应调用它压实。
 *
 * 返回是否有片段位置被改动，以及压实后的总时长；调用方借此避免无意义的
 * 写盘与撤销记录（单击选中之类的空拖动不该产生任何变更）。
 */
export function compactTrackClips(track: { clips: Clip[] }): { changed: boolean; duration: number } {
  track.clips.sort((a, b) => a.tl_start - b.tl_start)
  let cursor = 0
  let changed = false
  for (const c of track.clips) {
    const start = Number(cursor.toFixed(3))
    if (Math.abs(c.tl_start - start) > 1e-6) {
      c.tl_start = start
      changed = true
    }
    cursor = start + clipDur(c)
  }
  return { changed, duration: Number(cursor.toFixed(3)) }
}

/** 覆盖某个成片时刻的片段；落在片段之间的空隙 / 首段之前 / 末段之后时返回 null。
 *
 * 片段末端必须用「下一段的 tl_start」收口：`tl_start + (src_out - src_in) / speed`
 * 的浮点和可能比下一段的 tl_start 大出 1e-12 量级，恰好落在边界上的时刻
 * （跨片段续播时写入的正是 next.tl_start）会被错误地匹配给上一段，
 * 校正 effect 就会把刚切换的源又切回去，表现为跨片段播放来回振荡。
 * 方向相反时（浮点和偏小）则一切正常——所以这个 bug 只在个别片段对上出现。
 */
export function filmClipAt(clips: Clip[], t: number): Clip | null {
  for (let i = 0; i < clips.length; i++) {
    const c = clips[i]
    if (t < c.tl_start) return null
    const end =
      i + 1 < clips.length
        ? Math.min(c.tl_start + clipDur(c), clips[i + 1].tl_start)
        : c.tl_start + clipDur(c)
    if (t < end) return c
  }
  return null
}

/** 成片时刻 -> 该片段内的原片时间（调用方需自行确认片段归属的素材）。 */
export function filmTimeToSrc(c: Clip, t: number): number {
  return c.src_in + (t - c.tl_start) * (c.speed || 1)
}

/** 原片时刻 -> 成片时刻；只认属于指定素材的片段（跨素材成片时不同素材的原片时间会重叠）。 */
export function filmTimeOfSource(clips: Clip[], mediaId: string | null, src: number): number | null {
  for (const c of clips) {
    if (mediaId !== null && c.media_id !== mediaId) continue
    if (src >= c.src_in && src <= c.src_out) {
      return c.tl_start + (src - c.src_in) / (c.speed || 1)
    }
  }
  return null
}

/**
 * 把一个可能落在空隙里的成片时刻吸附到最近的片段范围内。
 *
 * 返回值保证被某个片段覆盖（片段末端会往回收半个容差，避免恰好落在
 * `t < tl_start + dur` 的开区间之外而吸附失败）。没有片段时返回 null。
 */
export function nearestCoveredFilmTime(clips: Clip[], t: number, eps = 0.001): number | null {
  if (!clips.length) return null
  let best: number | null = null
  let bestD = Infinity
  for (const c of clips) {
    const start = c.tl_start
    const end = c.tl_start + clipDur(c) - eps
    const clamped = Math.min(Math.max(t, start), Math.max(start, end))
    const d = Math.abs(clamped - t)
    if (d < bestD) {
      bestD = d
      best = clamped
    }
  }
  return best
}
