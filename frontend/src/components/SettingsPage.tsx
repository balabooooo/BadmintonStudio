import { useCallback, useEffect, useState } from 'react'
import {
  Cpu,
  HardDrive,
  Trash2,
  Zap,
  CheckCircle2,
  XCircle,
  FolderTree,
  Film,
  AlertTriangle,
  Languages,
  Sparkles,
  ScrollText,
  Download,
  Eraser,
} from 'lucide-react'
import { api } from '../lib/api'
import { bytes, cn } from '../lib/format'
import { clearLogs, downloadLogs, getLogs } from '../lib/logger'
import { Button, Card, Progress, SectionTitle, Segmented, Stat, useConfirm } from './ui'
import CacheClearDialog from './CacheClearDialog'
import { useStore } from '../store/useStore'
import { useTourStore } from '../tour/tourStore'
import { useT } from '../i18n/useT'
import type { Lang } from '../i18n'

/** Frontend ring-buffer capacity; must match RING_SIZE in lib/logger.ts. */
const FRONTEND_LOG_CAPACITY = 1500

export default function SettingsPage() {
  const env = useStore((s) => s.env)
  const lang = useStore((s) => s.lang)
  const setLang = useStore((s) => s.setLang)
  const toast = useStore((s) => s.toast)
  const confirm = useConfirm()
  const t = useT()
  const [cache, setCache] = useState<{ cache: number; proxies: number; thumbs: number; exports: number } | null>(null)
  const [cacheDialogOpen, setCacheDialogOpen] = useState(false)
  // Buffered frontend log count; initialized lazily on mount and refreshed after export/clear.
  const [logCount, setLogCount] = useState(() => getLogs().length)
  const refreshLogCount = useCallback(() => setLogCount(getLogs().length), [])

  const loadCache = useCallback(async () => {
    try {
      setCache(await api.cacheStats())
    } catch (e) {
      // 之前静默失败会一直显示 0，让人以为缓存是空的；这里至少提示一次。
      setCache(null)
      toast({ kind: 'error', title: t('settings.cacheStatsFailed'), detail: String(e) })
    }
  }, [t, toast])
  useEffect(() => {
    void loadCache()
  }, [loadCache])

  const probe = env?.probe
  const probeLabel =
    probe?.active === 'pyav'
      ? t('settings.probePyav')
      : probe?.active === 'ffprobe'
        ? t('settings.probeFfprobe')
        : probe?.active === 'ffmpeg'
          ? t('settings.probeFfmpeg')
          : t('common.unknown')

  const caps = env?.caps || {}
  const capItems: [string, boolean][] = [
    [t('settings.capsNvencH264'), !!caps.nvenc_h264],
    [t('settings.capsNvencHevc'), !!caps.nvenc_hevc],
    [t('settings.capsQsv'), !!caps.qsv],
    [t('settings.capsCudaDecode'), !!caps.cuda_decode],
    [t('settings.capsD3d11'), !!caps.d3d11va],
    [t('settings.capsX264'), !!caps.libx264],
  ]

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-[1000px] px-8 py-8">
        <h1 className="text-[24px] font-semibold tracking-tight text-white">{t('settings.title')}</h1>
        <p className="mt-1 mb-6 text-[12.5px] text-ink-400">{t('settings.subtitle')}</p>

        <Card className="mb-4 p-5">
          <SectionTitle>
            <Languages size={13} /> {t('settings.language')}
          </SectionTitle>
          <div className="flex items-center justify-between gap-3">
            <span className="text-[12.5px] text-ink-300">{t('settings.languageHint')}</span>
            <Segmented<Lang>
              value={lang}
              options={[
                { value: 'zh', label: '中文' },
                { value: 'en', label: 'English' },
              ]}
              onChange={(v) => setLang(v)}
            />
          </div>
        </Card>

        <Card data-tour="settings-guide" className="mb-4 p-5">
          <SectionTitle>
            <Sparkles size={13} /> {t('tour.settings.cardTitle')}
          </SectionTitle>
          <div className="flex items-center justify-between gap-3">
            <span className="text-[12.5px] text-ink-300">{t('tour.settings.cardDesc')}</span>
            <Button variant="outline" size="sm" onClick={() => useTourStore.getState().start('manual')}>
              {t('tour.welcome.start')}
            </Button>
          </div>
        </Card>

        <div className="grid gap-4 md:grid-cols-2">
          <Card className="p-5">
            <SectionTitle>
              <Cpu size={13} /> {t('settings.compute')}
            </SectionTitle>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-[12.5px] text-ink-300">{t('settings.gpuAccel')}</span>
                <span className={cn('text-[12.5px] font-medium', env?.gpu?.available ? 'text-court-400' : 'text-amber-glow')}>
                  {env?.gpu?.available ? t('settings.gpuEnabled') : t('settings.gpuUnavailable')}
                </span>
              </div>
              <div className="text-[12.5px] text-ink-100">{env?.gpu?.name || t('settings.noCuda')}</div>
              <div className="mono text-[11px] text-ink-500">
                PyTorch {env?.gpu?.torch || '-'}
                {env?.gpu?.capability ? ` · sm_${env.gpu.capability.join('')}` : ''}
              </div>
              <div className="mono text-[11px] text-ink-500">Python {env?.python}</div>
              <div className="mono text-[11px] text-ink-500">{env?.platform}</div>
            </div>
          </Card>

          <Card className="p-5">
            <SectionTitle>
              <Zap size={13} /> {t('settings.ffmpegCaps')}
            </SectionTitle>
            <div className="space-y-2">
              {capItems.map(([label, ok]) => (
                <div key={label} className="flex items-center gap-2 text-[12.5px]">
                  {ok ? (
                    <CheckCircle2 size={13} className="text-court-400" />
                  ) : (
                    <XCircle size={13} className="text-ink-600" />
                  )}
                  <span className={ok ? 'text-ink-100' : 'text-ink-500'}>{label}</span>
                </div>
              ))}
            </div>
            <div className="mono mt-3 break-all rounded-lg bg-black/25 px-2.5 py-2 text-[10.5px] text-ink-500">
              {env?.ffmpeg || env?.ffmpeg_error || t('common.notDetected')}
            </div>
          </Card>

          <Card className="p-5 md:col-span-2">
            <SectionTitle>
              <Film size={13} /> {t('settings.probe')}
            </SectionTitle>
            <div className="space-y-2 text-[12.5px]">
              <div className="flex items-center gap-2">
                {probe?.active === 'ffmpeg' ? (
                  <AlertTriangle size={13} className="text-amber-glow" />
                ) : (
                  <CheckCircle2 size={13} className="text-court-400" />
                )}
                <span className="text-ink-100">{t('settings.probeActive', { label: probeLabel })}</span>
              </div>
              <div className="mono text-[11px] break-all text-ink-500">
                {t('settings.ffprobeLabel', { path: probe?.ffprobe || t('settings.ffprobeMissing') })}
              </div>
              <div className="mono text-[11px] break-all text-ink-500">
                {t('settings.pyavLabel', {
                  state: probe?.pyav
                    ? t('common.available')
                    : t('settings.pyavUnavailable', { err: probe?.pyav_error ? ` · ${probe.pyav_error}` : '' }),
                })}
              </div>
              {probe?.active === 'ffmpeg' && (
                <div className="text-[11.5px] text-amber-glow/90">{t('settings.ffprobeHint')}</div>
              )}
            </div>
          </Card>

          <Card data-tour="settings-clear-cache" className="p-5 md:col-span-2">
            <SectionTitle
              right={
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" onClick={() => setCacheDialogOpen(true)}>
                    <Trash2 size={13} />
                    {t('settings.clearCache')}
                  </Button>
                </div>
              }
            >
              <HardDrive size={13} /> {t('settings.disk')}
            </SectionTitle>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <Stat label={t('settings.cacheTotal')} value={bytes(cache?.cache || 0)} />
              <Stat label={t('settings.proxies')} value={bytes(cache?.proxies || 0)} hint={t('settings.proxiesHint')} />
              <Stat label={t('settings.thumbs')} value={bytes(cache?.thumbs || 0)} />
              <Stat label={t('settings.exports')} value={bytes(cache?.exports || 0)} />
            </div>
            <div className="mt-3 space-y-2">
              <div className="flex items-center justify-between text-[11.5px] text-ink-400">
                <span>{t('settings.cacheRatio')}</span>
                <span className="mono">
                  {cache && cache.exports + cache.cache > 0
                    ? ((cache.cache / (cache.cache + cache.exports)) * 100).toFixed(0)
                    : 0}
                  %
                </span>
              </div>
              <Progress
                value={
                  cache && cache.cache + cache.exports > 0 ? cache.cache / (cache.cache + cache.exports) : 0
                }
              />
            </div>
          </Card>

          <Card className="p-5 md:col-span-2">
            <SectionTitle>
              <FolderTree size={13} /> {t('settings.dirs')}
            </SectionTitle>
            <div className="space-y-2">
              {[
                [t('settings.dirData'), env?.data_dir],
                [t('settings.dirCache'), env?.cache_dir],
                [t('settings.dirModels'), env?.models_dir],
              ].map(([k, v]) => (
                <div key={k as string} className="flex items-center gap-3">
                  <span className="w-20 shrink-0 text-[12px] text-ink-400">{k}</span>
                  <code className="mono min-w-0 flex-1 truncate rounded-md bg-black/25 px-2 py-1 text-[11px] text-ink-300">
                    {v || '-'}
                  </code>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      if (v) navigator.clipboard.writeText(v as string)
                      toast({ kind: 'info', title: t('common.copied') })
                    }}
                  >
                    {t('common.copy')}
                  </Button>
                </div>
              ))}
            </div>
          </Card>

          <Card data-tour="settings-debug-logs" className="p-5 md:col-span-2">
            <SectionTitle
              right={
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      const n = getLogs().length
                      if (n === 0 || !downloadLogs()) {
                        toast({ kind: 'info', title: t('settings.debugLogsEmpty') })
                        return
                      }
                      toast({ kind: 'success', title: t('settings.debugLogsExported', { count: n }) })
                    }}
                  >
                    <Download size={13} />
                    {t('settings.exportFrontendLogs')}
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={async () => {
                      const ok = await confirm({
                        title: t('settings.clearFrontendLogs'),
                        desc: t('settings.frontendLogsHint', { max: FRONTEND_LOG_CAPACITY, count: logCount }),
                      })
                      if (!ok) return
                      clearLogs()
                      refreshLogCount()
                      toast({ kind: 'info', title: t('settings.debugLogsCleared') })
                    }}
                  >
                    <Eraser size={13} />
                    {t('settings.clearFrontendLogs')}
                  </Button>
                </div>
              }
            >
              <ScrollText size={13} /> {t('settings.debugLogs')}
            </SectionTitle>
            <div className="space-y-2 text-[12.5px]">
              <p className="text-ink-300">{t('settings.debugLogsDesc')}</p>
              <p className="text-ink-400">
                {t('settings.frontendLogsHint', { max: FRONTEND_LOG_CAPACITY, count: logCount })}
              </p>
              <p className="text-[11.5px] text-ink-500">{t('settings.backendLogsHint')}</p>
              <div className="flex items-center gap-3">
                <code className="mono min-w-0 flex-1 truncate rounded-md bg-black/25 px-2 py-1 text-[11px] text-ink-300">
                  {env?.logs_dir ? `${env.logs_dir}\\bms_debug_YYYY-MM-DD.log` : '-'}
                </code>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    if (env?.logs_dir) navigator.clipboard.writeText(env.logs_dir)
                    toast({ kind: 'info', title: t('common.copied') })
                  }}
                >
                  {t('common.copy')}
                </Button>
              </div>
            </div>
          </Card>
        </div>
      </div>
      <CacheClearDialog
        open={cacheDialogOpen}
        onClose={() => setCacheDialogOpen(false)}
        onCleared={() => void loadCache()}
      />
    </div>
  )
}
