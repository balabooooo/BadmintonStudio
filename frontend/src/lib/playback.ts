/** 播放器预设倍速。HTML5 playbackRate 支持小数，这里只暴露常用档位。 */
export const SPEEDS = [0.25, 0.5, 1, 1.5, 2, 4] as const

/** 在预设档位间移动：dir=+1 加速，-1 减速，越界夹到两端。 */
export function stepSpeedValue(current: number, dir: number): number {
  const list = SPEEDS as readonly number[]
  const i = list.indexOf(current)
  return list[Math.min(Math.max(i + dir, 0), list.length - 1)]
}
