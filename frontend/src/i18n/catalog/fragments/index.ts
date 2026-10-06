/** Aggregates all i18n catalog fragments: `[key, 中文, English]` tuples. */

import { FRAG as analysis } from './analysis'
import { FRAG as inspector } from './inspector'
import { FRAG as annotate } from './annotate'
import { FRAG as player } from './player'
import { FRAG as media } from './media'
import { FRAG as tour } from './tour'

export const FRAGMENTS: [string, string, string][] = [
  ...analysis,
  ...inspector,
  ...annotate,
  ...player,
  ...media,
  ...tour,
]

export function zhDict(): Record<string, string> {
  return Object.fromEntries(FRAGMENTS.map(([k, z]) => [k, z]))
}

export function enDict(): Record<string, string> {
  return Object.fromEntries(FRAGMENTS.map(([k, , e]) => [k, e]))
}
