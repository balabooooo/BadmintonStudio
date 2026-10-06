import { motion } from 'motion/react'
import { useEffect, useMemo, useState } from 'react'
import { FolderPlus, Film, Trash2, Clock, Sparkles, Search, Clapperboard, Copy } from 'lucide-react'
import { api } from '../lib/api'
import { humanDuration, relTime } from '../lib/format'
import { Button, Card, Empty, Modal, Skeleton, Tooltip, useConfirm } from './ui'
import ImportVideoButton from './ImportVideoButton'
import { useStore } from '../store/useStore'
import { useT } from '../i18n/useT'
import { getLang } from '../i18n'

export default function LibraryPage() {
  const t = useT()
  const projects = useStore((s) => s.projects)
  const refresh = useStore((s) => s.refreshProjects)
  const createProject = useStore((s) => s.createProject)
  const openProject = useStore((s) => s.openProject)
  const deleteProject = useStore((s) => s.deleteProject)
  const booted = useStore((s) => s.booted)
  const toast = useStore((s) => s.toast)
  const confirm = useConfirm()

  const [q, setQ] = useState('')
  const [creating, setCreating] = useState(false)
  const [saving, setSaving] = useState(false)
  const [name, setName] = useState('')

  function defaultProjectName(): string {
    return t('library.newProjectName', {
      date: new Date().toLocaleDateString(getLang() === 'en' ? 'en-US' : 'zh-CN'),
    })
  }

  /** 新建工程：页脚按钮与输入框回车共用，避免两处逻辑漂移。 */
  async function handleCreate() {
    const n = name.trim()
    if (!n) return
    setSaving(true)
    try {
      const id = await createProject(n)
      setCreating(false)
      if (id) toast({ kind: 'success', title: t('library.created'), detail: t('library.createdDetail') })
    } catch (e) {
      toast({ kind: 'error', title: t('library.createFailed'), detail: String(e) })
    } finally {
      setSaving(false)
    }
  }

  useEffect(() => {
    refresh()
  }, [refresh])

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase()
    return s ? projects.filter((p) => p.name.toLowerCase().includes(s)) : projects
  }, [projects, q])

  const stats = useMemo(
    () => ({
      count: projects.length,
      hours: projects.reduce((a, p) => a + p.duration, 0) / 3600,
      rallies: projects.reduce((a, p) => a + p.rally_count, 0),
    }),
    [projects],
  )

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-[1400px] px-8 py-8">
        {/* 头部 */}
        <div className="mb-8 flex flex-wrap items-end justify-between gap-5">
          <div>
            <motion.h1
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
              className="text-[26px] font-semibold tracking-tight text-white"
            >
              {t('library.title')}
            </motion.h1>
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.08, duration: 0.4 }}
              className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-[12.5px] text-ink-400"
            >
              <span className="inline-flex items-center gap-1.5">
                <Clapperboard size={13} /> {t('library.projectCount', { n: stats.count })}
              </span>
              <span className="inline-flex items-center gap-1.5">
                <Clock size={13} /> {t('library.totalDuration', { hours: stats.hours.toFixed(1) })}
              </span>
              <span className="inline-flex items-center gap-1.5">
                <Sparkles size={13} /> {t('library.ralliesFound', { n: stats.rallies })}
              </span>
            </motion.p>
          </div>
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search size={14} className="absolute top-1/2 left-3 -translate-y-1/2 text-ink-500" />
              <input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder={t('library.searchPlaceholder')}
                className="field w-[220px] pl-8"
              />
            </div>
            <Button
              variant="primary"
              size="lg"
              data-tour="library-new-project"
              onClick={() => {
                setName(defaultProjectName())
                setCreating(true)
              }}
            >
              <FolderPlus size={15} />
              {t('library.newProject')}
            </Button>
          </div>
        </div>

        {/* 列表 */}
        {!booted && !projects.length ? (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(300px,1fr))] gap-4">
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} className="h-[220px]" />
            ))}
          </div>
        ) : filtered.length === 0 ? (
          <Card className="py-6">
            <Empty
              icon={<Film size={34} />}
              title={q ? t('library.noMatch') : t('library.noProjects')}
              desc={q ? t('library.noMatchDesc') : t('library.noProjectsDesc')}
              action={
                !q && (
                  <Button
                    variant="primary"
                    onClick={() => {
                      setName(defaultProjectName())
                      setCreating(true)
                    }}
                  >
                    <FolderPlus size={15} />
                    {t('library.createFirst')}
                  </Button>
                )
              }
            />
          </Card>
        ) : (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(320px,1fr))] gap-4">
            {filtered.map((p, i) => (
              <motion.div
                key={p.id}
                initial={{ opacity: 0, y: 14 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: Math.min(i * 0.045, 0.4), duration: 0.38, ease: [0.22, 1, 0.36, 1] }}
              >
                <Card hover className="group overflow-hidden">
                  <button
                    className="relative block h-[152px] w-full overflow-hidden bg-ink-850"
                    onClick={() => openProject(p.id)}
                  >
                    {p.poster ? (
                      <img
                        src={api.assetUrl(p.poster)}
                        alt=""
                        className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-[1.05]"
                        loading="lazy"
                      />
                    ) : (
                      <div className="flex h-full items-center justify-center text-ink-700">
                        <Film size={34} />
                      </div>
                    )}
                    <div className="absolute inset-0 bg-gradient-to-t from-ink-950/92 via-ink-950/25 to-transparent" />
                    <div className="absolute bottom-2 left-3 flex items-center gap-2 text-[11px] text-ink-200">
                      <span className="mono">{humanDuration(p.duration)}</span>
                      <span className="text-ink-500">·</span>
                      <span>{t('library.mediaCount', { n: p.media_count })}</span>
                    </div>
                    {p.analyzed && (
                      <span className="absolute top-2.5 right-2.5 inline-flex items-center gap-1 rounded-md bg-court-500/85 px-1.5 py-[2px] text-[10.5px] font-semibold text-ink-950">
                        <Sparkles size={10} />
                        {t('library.rallyCount', { n: p.rally_count })}
                      </span>
                    )}
                  </button>
                  <div
                    className="group/name flex cursor-pointer items-start justify-between gap-2 px-3.5 py-3"
                    onClick={() => openProject(p.id)}
                  >
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-[13px] font-medium text-white transition-colors group-hover/name:text-court-300">
                        {p.name}
                      </div>
                      <div className="mt-0.5 text-[11px] text-ink-500">
                        {t('library.updatedAt', { time: relTime(p.updated_at) })}
                        {!p.analyzed && <span className="ml-2 text-amber-glow">{t('library.notAnalyzed')}</span>}
                      </div>
                    </div>
                    <div className="flex shrink-0 items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                      <Tooltip content={t('library.importVideo')}>
                        {/* 拦住冒泡，否则点导入会顺带触发卡片的「打开工程」 */}
                        <span onClick={(e) => e.stopPropagation()}>
                          <ImportVideoButton
                            variant="ghost"
                            size="icon"
                            label=""
                            projectId={p.id}
                            onDone={() => refresh()}
                          />
                        </span>
                      </Tooltip>
                      <Button
                        variant="ghost"
                        size="icon"
                        title={t('library.duplicate')}
                        onClick={async (e) => {
                          // 别让点击冒泡到卡片本身：那会顺手把工程打开，跳进剪辑台
                          e.stopPropagation()
                          try {
                            await api.duplicateProject(p.id)
                            refresh()
                            toast({ kind: 'success', title: t('library.duplicated') })
                          } catch (err) {
                            toast({ kind: 'error', title: t('library.duplicateFailed'), detail: String(err) })
                          }
                        }}
                      >
                        <Copy size={13} />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        title={t('library.delete')}
                        onClick={async (e) => {
                          e.stopPropagation()
                          const ok = await confirm({
                            title: t('library.deleteTitle', { name: p.name }),
                            desc: t('library.deleteDesc'),
                            danger: true,
                          })
                          if (ok) {
                            try {
                              await deleteProject(p.id)
                              toast({ kind: 'info', title: t('library.deleted') })
                            } catch (err) {
                              toast({ kind: 'error', title: t('library.deleteFailed'), detail: String(err) })
                            }
                          }
                        }}
                      >
                        <Trash2 size={13} className="text-rose-hot/80" />
                      </Button>
                    </div>
                  </div>
                </Card>
              </motion.div>
            ))}
          </div>
        )}
      </div>

      {/* 新建工程 */}
      <Modal
        open={creating}
        onClose={() => setCreating(false)}
        title={t('library.modalTitle')}
        subtitle={t('library.modalSubtitle')}
        width={460}
        footer={
          <div className="flex justify-end gap-2">
            <Button variant="ghost" onClick={() => setCreating(false)}>
              {t('common.cancel')}
            </Button>
            <Button variant="primary" disabled={!name.trim()} loading={saving} onClick={() => void handleCreate()}>
              {t('library.createAndEnter')}
            </Button>
          </div>
        }
      >
        <label className="mb-1.5 block text-[12px] text-ink-300">{t('library.nameLabel')}</label>
        <input
          autoFocus
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => {
            // 中文/日文输入法按回车确认候选词时也会触发 keydown，必须避开，
            // 否则候选词还没上屏就把工程建出来了。
            if (e.key === 'Enter' && !e.nativeEvent.isComposing && name.trim()) {
              void handleCreate()
            }
          }}
          className="field"
          placeholder={t('library.namePlaceholder')}
        />
      </Modal>
    </div>
  )
}
