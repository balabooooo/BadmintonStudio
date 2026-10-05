import { ChevronDown, Film } from 'lucide-react'
import { api } from '../lib/api'
import { cn } from '../lib/format'
import { Tooltip } from './ui'
import { useStore } from '../store/useStore'
import { useT } from '../i18n/useT'

/**
 * Compact media switcher chip shared by the Studio top bar, Annotate header,
 * and the Analysis dialog. Opens the global MediaPickerDialog in switch mode.
 */
export default function MediaChip({ className }: { className?: string }) {
  const tr = useT()
  const media = useStore((s) =>
    s.mediaId ? (s.project?.media.find((m) => m.id === s.mediaId) ?? null) : null,
  )
  const openMediaPicker = useStore((s) => s.openMediaPicker)

  if (!media) return null

  return (
    <Tooltip content={media.path} side="bottom">
      <button
        onClick={() => openMediaPicker('switch')}
        aria-label={tr('mediaPicker.switchTo')}
        title={media.path}
        className={cn(
          'group flex max-w-[280px] items-center gap-2 rounded-lg border border-white/8 bg-white/[0.03] py-1 pr-2 pl-1.5 text-[11.5px] text-ink-200 transition-all hover:border-white/18 hover:bg-white/[0.06]',
          className,
        )}
      >
        <span className="grid h-7 w-11 shrink-0 place-items-center overflow-hidden rounded-md bg-ink-850 text-ink-700">
          {media.poster ? (
            <img src={api.assetUrl(media.poster)} className="h-full w-full object-cover" alt="" />
          ) : (
            <Film size={13} />
          )}
        </span>
        <span className="truncate">{media.name}</span>
        <ChevronDown size={12} className="shrink-0 text-ink-500" />
      </button>
    </Tooltip>
  )
}
